"""Test D -- mirror descent replacing STE in NanoQuant's Step 3. Block 0 only.

setup : run Step 1 (tune_nonfact) + Step 2 (ADMM) for all seven layers of block 0
        and bank the post-ADMM block, every NanoQuantLinear still holding live
        latent proxies. Nothing is finalised.
run   : load that exact state and apply one Step 3 variant, then measure the
        held-out block error AFTER hard sign() has been written in and the
        relaxed forward disabled.
"""
import argparse, json, math, os, time

import torch
import torch.nn as nn

from nanoquant.modules.hub import NanoQuantConfigDataclass  # noqa: F401  import-order guard
from nanoquant.core.compress_block import (factorize_and_replace, fused_weighted_mse,
                                           get_param_group_config, tune_nonfact)
from nanoquant.core.importance import register_stats
from nanoquant.modules.linear import NanoQuantLinear
from nanoquant.optimi import AdamW
from nanoquant.utils.load_utils import load_model
from nanoquant.utils.utils import (calculate_ranks, cleanup_memory, find_layers, get_decoder_layers,
                                   get_layers_to_factorize, set_seed)

import instrument as I

CACHE = "/home/work/exp/artifacts/sbh"
DCACHE = "/home/work/exp/artifacts/testD"
MID = ("/home/work/hf_cache/hub/models--unsloth--Llama-3.2-1B/snapshots/"
       "9535bd9b1d1dea6acafbdc4813b728796aeb28da")
PAPER_GAMMA = 0.2
NAMES = ['self_attn.q_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'self_attn.k_proj',
         'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']


def qcfg(seed=0):
    return NanoQuantConfigDataclass(model_id=MID, seed=seed, calib_shrinkage=PAPER_GAMMA,
                                    admm_penalty_scheduler="linear").to_dict()


# ---------------------------------------------------------------------------
# mirror map:  primal U = tanh(beta * Theta);  dL/dU is applied to Theta directly
# ---------------------------------------------------------------------------
MIRROR = {"on": False, "beta": 1.0, "hard": False}


class _MirrorMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, beta, hard, out_dtype):
        ctx.tdtype = theta.dtype
        u = torch.tanh(beta * theta)
        if hard:
            u = torch.where(u >= 0, torch.ones_like(u), -torch.ones_like(u))
        return u.to(out_dtype)

    @staticmethod
    def backward(ctx, g):
        # mirror step: the gradient wrt U goes straight to Theta, NOT through tanh'
        return g.to(ctx.tdtype), None, None, None


_orig_quantize = NanoQuantLinear.quantize


def _quantize(self, x):
    # Theta parameters are the only fp32 tensors on these modules; scales are bf16
    if MIRROR["on"] and not self._binarized and x.dtype == torch.float32:
        return _MirrorMap.apply(x, MIRROR["beta"], MIRROR["hard"], self.dtype)
    return _orig_quantize(self, x)


NanoQuantLinear.quantize = _quantize


def nq_modules(blk):
    return [(n, m) for n, m in blk.named_modules() if isinstance(m, NanoQuantLinear)]


def _sign(x):
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


# ---------------------------------------------------------------------------
def load_harness(seed=0):
    qd = qcfg(seed)
    model = load_model(MID, qd['seqlen'], device_map="cpu")
    st = torch.load(f"{CACHE}/stats.pt")
    st['stats_device'] = 'cpu'
    model = register_stats(model, st)
    model.eval()
    model.config.use_cache = False
    kwargs = torch.load(f"{CACHE}/kwargs.pt")
    kwargs = {k: (v.to("cuda") if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
    x = torch.load(f"{CACHE}/in_b0.pt").detach().requires_grad_(False)
    y = torch.load(f"{CACHE}/out_b0.pt").detach().requires_grad_(False)
    n = qd['num_calib_samples']
    return model, qd, kwargs, x[:n], y[:n], x[n:], y[n:]


def do_setup(args):
    os.makedirs(DCACHE, exist_ok=True)
    model, qd, kwargs, cal_in, cal_out, ho_in, ho_out = load_harness(args.seed)
    blk = get_decoder_layers(model)[0].to("cuda")
    sub = find_layers(blk)
    ranks = calculate_ranks(model, NAMES, qd)
    imp = sub['mlp.down_proj'].o_norm.to("cuda")
    cal_in_d, cal_out_d = cal_in.to("cuda"), cal_out.to("cuda")

    set_seed(qd['seed'])
    W_refs, t0 = {}, time.time()
    for name in NAMES:
        print(f"  (1/2) {name} Step 1 ...", flush=True)
        tune_nonfact(blk, cal_in_d, cal_out_d, imp, kwargs, qd)
        cleanup_memory()
        W_refs[name] = sub[name].weight.data.clone().cpu()
        print(f"  (2/2) {name} ADMM  rank={ranks[f'0.{name}']} ...", flush=True)
        nano, _ = factorize_and_replace(blk, name, ranks[f"0.{name}"], qd)
        # freeze until Step 3: in the released order the earlier layers are already
        # finalised and frozen by the time the next Step 1 runs
        for p in nano.parameters():
            p.requires_grad_(False)
        cleanup_memory()

    pre = I.block_err(blk, ho_in, ho_out, kwargs)
    print(f"[setup] post-ADMM block pre error = {pre:.6f}   ({time.time()-t0:.0f}s)")
    torch.save({"block": blk.cpu(), "W_refs": W_refs, "ranks": ranks, "pre": pre},
               f"{DCACHE}/block0_postadmm.pt")
    blk.to("cuda")
    re_pre = I.block_err(blk, ho_in, ho_out, kwargs)
    reloaded = torch.load(f"{DCACHE}/block0_postadmm.pt", weights_only=False)["block"].to("cuda")
    ld_pre = I.block_err(reloaded, ho_in, ho_out, kwargs)
    print(f"[setup] cache round-trip: in-memory {re_pre:.9f}  reloaded {ld_pre:.9f}  "
          f"{'IDENTICAL' if re_pre == ld_pre else 'DIFFERS'}")
    json.dump({"pre": pre, "reload_pre": ld_pre, "ranks": ranks}, open(f"{DCACHE}/setup.json", "w"), indent=1)


# ---------------------------------------------------------------------------
def to_theta(blk):
    """Replace each bf16 latent proxy with an fp32 dual variable
    Theta = atanh(clamp(proxy / max|proxy|, +-0.99))."""
    init_signs = {}
    for name, m in nq_modules(blk):
        for attr in ["V_latent", "U_latent"]:
            p = getattr(m, attr)
            proxy = p.data.float()
            t = (proxy / proxy.abs().max().clamp(min=1e-12)).clamp(-0.99, 0.99)
            theta = torch.atanh(t)
            new = nn.Parameter(theta, requires_grad=True)
            new.optim_group = "binary"
            del m._parameters[attr]
            setattr(m, attr, new)
            init_signs[f"{name}.{attr}"] = _sign(theta).to(torch.int8).cpu()
    return init_signs


def harden(blk):
    """Write the hard sign() in and disable the relaxed forward."""
    with torch.no_grad():
        for _, m in nq_modules(blk):
            for attr in ["V_latent", "U_latent"]:
                base = attr.replace("_latent", "")
                v = _sign(getattr(m, attr).data).to(torch.bfloat16)
                del m._parameters[attr]
                setattr(m, base, nn.Parameter(v, requires_grad=False))
            m.do_train = False
            m._binarized = True
            for p in m.parameters():
                p.requires_grad_(False)


def step3(blk, qd, cal_in_d, cal_out_d, imp, kwargs, mode, eta, beta_end):
    """One Step 3 pass. Everything but the binary update rule matches the release."""
    set_seed(qd['seed'])
    numel = cal_out_d.numel()
    bs = qd['fact_batch_size']
    epochs = qd['fact_epochs']
    nsamp = qd['num_calib_samples']
    total_steps = math.ceil(nsamp / bs) * epochs

    for _, m in nq_modules(blk):
        for n_, p in m.named_parameters():
            if "latent" in n_ or "scale" in n_ or "bias" in n_:
                p.requires_grad_(True)

    init_signs = {}
    if mode == "ste":
        for name, m in nq_modules(blk):
            for attr in ["V_latent", "U_latent"]:
                init_signs[f"{name}.{attr}"] = _sign(getattr(m, attr).data).to(torch.int8).cpu()
        cfg = get_param_group_config(blk, binary_lr=qd['fact_binary_lr'],
                                     scale_lr=qd['fact_scale_lr'], bias_lr=qd['fact_bias_lr'])
        opt = AdamW(cfg, weight_decay=0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps,
                                                           eta_min=1e-4 * qd['fact_scale_lr'])
        mirror_opt = None
    else:
        init_signs = to_theta(blk)
        groups = {"scale": [], "bias": [], "binary": []}
        for _, m in nq_modules(blk):
            for n_, p in m.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue
                tag = getattr(p, "optim_group", None)
                if tag is None and p.ndim == 1 and "bias" in n_:
                    tag = "bias"
                if tag in groups:
                    groups[tag].append(p)
        cfg = []
        if groups["scale"]:
            cfg.append({"params": groups["scale"], "lr": qd['fact_scale_lr']})
        if groups["bias"]:
            cfg.append({"params": groups["bias"], "lr": qd['fact_bias_lr']})
        opt = AdamW(cfg, weight_decay=0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps,
                                                           eta_min=1e-4 * qd['fact_scale_lr'])
        mirror_opt = torch.optim.SGD(groups["binary"], lr=eta)
        MIRROR["on"] = True
        MIRROR["hard"] = (mode == "hard")
        MIRROR["beta"] = 1.0

    step = 0
    for epoch in range(epochs):
        idx = torch.randperm(nsamp, device="cpu", dtype=torch.long)
        ep_loss = torch.zeros(1, device="cuda")
        for i in range(nsamp):
            j = idx[i].item()
            if mirror_opt is not None and mode == "relaxed":
                MIRROR["beta"] = 1.0 + (beta_end - 1.0) * (step / max(total_steps - 1, 1))
            y = blk(cal_in_d[j:j + 1], **kwargs)[0]
            loss = fused_weighted_mse(y, cal_out_d[j:j + 1], imp)
            (loss / bs).backward()
            if (i + 1) % bs == 0 or (i + 1) == nsamp:
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                if mirror_opt is not None:
                    mirror_opt.step()
                    mirror_opt.zero_grad(set_to_none=True)
            ep_loss += loss.detach()
            step += 1
        cleanup_memory()
        print(f"\t\t(Epoch {epoch+1:02d}/{epochs:02d}) Block Loss: {(ep_loss/numel).item():.4e}"
              + (f"  beta={MIRROR['beta']:.2f}" if mirror_opt is not None else ""), flush=True)

    final_signs = {}
    for name, m in nq_modules(blk):
        for attr in ["V_latent", "U_latent"]:
            final_signs[f"{name}.{attr}"] = _sign(getattr(m, attr).data).to(torch.int8).cpu()
    harden(blk)
    MIRROR["on"] = False
    flips = {k: (final_signs[k] != init_signs[k]).float().mean().item() for k in init_signs}
    return flips


def do_run(args):
    model, qd, kwargs, cal_in, cal_out, ho_in, ho_out = load_harness(args.seed)
    blob = torch.load(f"{DCACHE}/block0_postadmm.pt", weights_only=False)
    blk = blob["block"].to("cuda")
    W_refs = blob["W_refs"]
    sub = dict(nq_modules(blk))
    imp = sub['mlp.down_proj'].o_norm.to("cuda")
    cal_in_d, cal_out_d = cal_in.to("cuda"), cal_out.to("cuda")

    rec = {"tag": args.tag, "mode": args.mode, "eta": args.eta, "beta_end": args.beta_end,
           "seed": args.seed}
    rec["pre"] = I.block_err(blk, ho_in, ho_out, kwargs)
    J = {}
    for name in NAMES:
        H = I.prep_hin(f"model.layers.0.{name}", f"{CACHE}/hin", shrinkage=PAPER_GAMMA, device="cuda")
        j, f = I.recon_objectives(W_refs[name].cuda(), sub[name], H)
        J[name] = {"J_pre": j, "fro_pre": f}
        del H
    cleanup_memory()

    t0 = time.time()
    flips = step3(blk, qd, cal_in_d, cal_out_d, imp, kwargs, args.mode, args.eta, args.beta_end)
    rec["step3_s"] = time.time() - t0
    rec["post"] = I.block_err(blk, ho_in, ho_out, kwargs)
    for name in NAMES:
        H = I.prep_hin(f"model.layers.0.{name}", f"{CACHE}/hin", shrinkage=PAPER_GAMMA, device="cuda")
        j, f = I.recon_objectives(W_refs[name].cuda(), sub[name], H)
        J[name].update({"J_post": j, "fro_post": f})
        del H
    rec["layers"] = J
    rec["flips"] = flips
    rec["flip_mean"] = sum(flips.values()) / len(flips)
    os.makedirs(f"{DCACHE}/runs", exist_ok=True)
    json.dump(rec, open(f"{DCACHE}/runs/{args.tag}.json", "w"), indent=1)
    print(f"[D] FINAL {json.dumps({k: rec[k] for k in ['tag','mode','eta','pre','post','flip_mean','step3_s']})}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode_", choices=["setup", "run"])
    ap.add_argument("--tag", default="d")
    ap.add_argument("--mode", default="ste", choices=["ste", "relaxed", "hard"])
    ap.add_argument("--eta", type=float, default=1e-2)
    ap.add_argument("--beta_end", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    (do_setup if a.mode_ == "setup" else do_run)(a)
