"""Test D2 -- mirror descent, corrected.

Two fixes over Test D:
  1. AdamW on the dual variable Theta (weight decay 0) instead of a fixed-eta
     step, so per-coordinate second-moment normalisation absorbs the 139-525x
     within-layer gradient range that stalled Variant H.
  2. Initialise near the hypercube corners:
       'conf'   Theta0 = c * proxy / rms(proxy), c solved per factor so that
                mean|tanh(Theta0)| = 0.9   (keeps confidence ordering)
       'corner' Theta0 = atanh(0.95) * sign(proxy)   (corners only, ablation)
Both preserve sign(proxy) exactly, which is asserted against the cached pre.
"""
import argparse, json, math, os, time

import torch
import torch.nn as nn

import testD as D
import instrument as I
from nanoquant.core.compress_block import fused_weighted_mse
from nanoquant.optimi import AdamW
from nanoquant.utils.utils import cleanup_memory, set_seed

CACHED_PRE = 0.14366140267175126
OUT = "/home/work/exp/artifacts/testD2"


def solve_c(z, target=0.9, lo=1e-3, hi=1e4, iters=80):
    """c such that mean|tanh(c*z)| = target, by bisection (monotone in c)."""
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        if torch.tanh(mid * z).abs().mean().item() < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def to_theta2(blk, init_mode):
    """Replace bf16 latent proxies with fp32 dual variables."""
    init_signs, init_bin, cinfo = {}, {}, {}
    for name, m in D.nq_modules(blk):
        for attr in ["V_latent", "U_latent"]:
            proxy = getattr(m, attr).data.float()
            if init_mode == "conf":
                z = proxy / proxy.square().mean().sqrt().clamp(min=1e-12)
                c = solve_c(z)
                theta = c * z
                cinfo[f"{name}.{attr}"] = c
            elif init_mode == "corner":
                theta = math.atanh(0.95) * D._sign(proxy)
            else:
                raise ValueError(init_mode)
            new = nn.Parameter(theta, requires_grad=True)
            new.optim_group = "binary"
            del m._parameters[attr]
            setattr(m, attr, new)
            init_signs[f"{name}.{attr}"] = D._sign(theta).to(torch.int8).cpu()
            init_bin[f"{name}.{attr}"] = D._sign(proxy).to(torch.bfloat16).cpu()
            assert torch.equal(D._sign(theta), D._sign(proxy)), f"sign changed at {name}.{attr}"
    return init_signs, init_bin, cinfo


def run(args):
    os.makedirs(f"{OUT}/runs", exist_ok=True)
    model, qd, kwargs, cal_in, cal_out, ho_in, ho_out = D.load_harness(args.seed)
    blob = torch.load(f"{D.DCACHE}/block0_postadmm.pt", weights_only=False)
    blk = blob["block"].to("cuda")
    W_refs = blob["W_refs"]
    sub = dict(D.nq_modules(blk))
    imp = sub['mlp.down_proj'].o_norm.to("cuda")
    cal_in_d, cal_out_d = cal_in.to("cuda"), cal_out.to("cuda")

    rec = {"tag": args.tag, "variant": args.variant, "init": args.init, "lr": args.lr,
           "seed": args.seed, "pre_cached": CACHED_PRE}
    rec["pre_loaded"] = I.block_err(blk, ho_in, ho_out, kwargs)

    J = {}
    for n in D.NAMES:
        H = I.prep_hin(f"model.layers.0.{n}", f"{D.CACHE}/hin", shrinkage=D.PAPER_GAMMA, device="cuda")
        j, f = I.recon_objectives(W_refs[n].cuda(), sub[n], H)
        J[n] = {"J_pre": j, "fro_pre": f}
        del H
    cleanup_memory()

    set_seed(qd['seed'])
    for _, m in D.nq_modules(blk):
        for n_, p in m.named_parameters():
            if "latent" in n_ or "scale" in n_ or "bias" in n_:
                p.requires_grad_(True)

    init_signs, init_bin, cinfo = to_theta2(blk, args.init)
    rec["c_per_factor"] = cinfo

    # assert: reparameterisation must not move a single sign
    D.MIRROR["on"] = True
    D.MIRROR["hard"] = True
    D.MIRROR["beta"] = 1.0
    pre_theta = I.block_err(blk, ho_in, ho_out, kwargs)
    rec["pre_sign_theta0"] = pre_theta
    rec["pre_assert_ok"] = bool(pre_theta == rec["pre_loaded"])
    print(f"[D2] pre: cached={CACHED_PRE:.9f} loaded={rec['pre_loaded']:.9f} "
          f"sign(Theta0)={pre_theta:.9f} -> {'OK' if rec['pre_assert_ok'] else 'MISMATCH'}", flush=True)

    # optimizers: Theta on AdamW(lr), scales/bias on AdamW 1e-5 exactly as control
    groups = {"scale": [], "bias": [], "binary": []}
    for _, m in D.nq_modules(blk):
        for n_, p in m.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            tag = getattr(p, "optim_group", None)
            if tag is None and p.ndim == 1 and "bias" in n_:
                tag = "bias"
            if tag in groups:
                groups[tag].append(p)
    total_steps = math.ceil(qd['num_calib_samples'] / qd['fact_batch_size']) * qd['fact_epochs']
    cfg = []
    if groups["scale"]:
        cfg.append({"params": groups["scale"], "lr": qd['fact_scale_lr']})
    if groups["bias"]:
        cfg.append({"params": groups["bias"], "lr": qd['fact_bias_lr']})
    opt = AdamW(cfg, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps,
                                                       eta_min=1e-4 * qd['fact_scale_lr'])
    optM = AdamW(groups["binary"], lr=args.lr, weight_decay=0)
    schedM = torch.optim.lr_scheduler.CosineAnnealingLR(optM, T_max=total_steps,
                                                        eta_min=1e-4 * args.lr)

    D.MIRROR["hard"] = (args.variant == "H2")
    t0, step = time.time(), 0
    nsamp, bs = qd['num_calib_samples'], qd['fact_batch_size']
    for epoch in range(qd['fact_epochs']):
        idx = torch.randperm(nsamp, device="cpu", dtype=torch.long)
        ep = torch.zeros(1, device="cuda")
        for i in range(nsamp):
            if args.variant == "R2":
                D.MIRROR["beta"] = 1.0 + 9.0 * (step / max(total_steps - 1, 1))
            j = idx[i].item()
            y = blk(cal_in_d[j:j + 1], **kwargs)[0]
            loss = fused_weighted_mse(y, cal_out_d[j:j + 1], imp)
            (loss / bs).backward()
            if (i + 1) % bs == 0 or (i + 1) == nsamp:
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                optM.step(); schedM.step(); optM.zero_grad(set_to_none=True)
            ep += loss.detach()
            step += 1
        cleanup_memory()
        print(f"\t\t(Epoch {epoch+1:02d}/08) Block Loss: {(ep/cal_out_d.numel()).item():.4e}"
              f"  beta={D.MIRROR['beta']:.2f}", flush=True)
    rec["step3_s"] = time.time() - t0

    final_signs = {k: D._sign(getattr(sub[k.rsplit('.', 1)[0]], k.rsplit('.', 1)[1]).data).to(torch.int8).cpu()
                   for k in init_signs}
    tuned_bin = {k: v.to(torch.bfloat16) for k, v in final_signs.items()}
    D.harden(blk)
    D.MIRROR["on"] = False
    rec["post"] = I.block_err(blk, ho_in, ho_out, kwargs)
    rec["flips"] = {k: (final_signs[k] != init_signs[k]).float().mean().item() for k in init_signs}
    rec["flip_mean"] = sum(rec["flips"].values()) / len(rec["flips"])

    for n in D.NAMES:
        H = I.prep_hin(f"model.layers.0.{n}", f"{D.CACHE}/hin", shrinkage=D.PAPER_GAMMA, device="cuda")
        j, f = I.recon_objectives(W_refs[n].cuda(), sub[n], H)
        J[n].update({"J_post": j, "fro_post": f})
        del H
    rec["layers"] = J

    # cumulative per-layer post: layers 1..k tuned, k+1..7 back at their ADMM signs
    curve = {}
    with torch.no_grad():
        for k, n in enumerate(D.NAMES):
            for kk, nn_ in enumerate(D.NAMES):
                m = sub[nn_]
                src = tuned_bin if kk <= k else init_bin
                m.V.data.copy_(src[f"{nn_}.V_latent"].to("cuda"))
                m.U.data.copy_(src[f"{nn_}.U_latent"].to("cuda"))
            curve[n] = I.block_err(blk, ho_in, ho_out, kwargs)
        for nn_ in D.NAMES:
            m = sub[nn_]
            m.V.data.copy_(tuned_bin[f"{nn_}.V_latent"].to("cuda"))
            m.U.data.copy_(tuned_bin[f"{nn_}.U_latent"].to("cuda"))
    rec["layer_post_curve"] = curve

    json.dump(rec, open(f"{OUT}/runs/{args.tag}.json", "w"), indent=1)
    print("[D2] FINAL " + json.dumps({k: rec[k] for k in
                                      ["tag", "variant", "init", "lr", "post", "flip_mean",
                                       "pre_assert_ok", "step3_s"]}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variant", choices=["R2", "H2"], required=True)
    ap.add_argument("--init", choices=["conf", "corner"], required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())
