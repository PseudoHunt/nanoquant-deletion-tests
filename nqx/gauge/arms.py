"""Gauge-NQ Stage 1 arms (brief sections 9-12), down_proj only.

Every arm produces a *starting state* and then hands it to the identical common
Step 3.  The only thing that differs between arms is that starting state.
"""
import argparse
import json
import math
import os
import time

import torch

from nanoquant.modules.hub import NanoQuantConfigDataclass  # noqa: F401  import-order guard

import instrument as I
import testD as D
from nanoquant.core.compress_block import fused_weighted_mse, get_param_group_config
from nanoquant.optimi import AdamW
from nanoquant.utils.utils import cleanup_memory

from . import core as C
from . import harness as H
from . import state as S

LAYER = "mlp.down_proj"
BLOCK = 32
ARM3_STEPS = 200
ARM3_CKPTS = (100, 200)
ARM3_LR = 3e-3
ITQ_ROUNDS = 20
ARM1_SAMPLES = 16
ARM1_SEED = 777
ITQ_INIT_SEED = 4242
DELTA = 0.0031          # determinism floor, block 0


# ---------------------------------------------------------------------------
class Runner:
    def __init__(self, seed=0, layers=(LAYER,), b=BLOCK):
        self.st = H.load_post_admm(seed=seed)
        self.qd = self.st["qd"]
        self.kwargs = self.st["kwargs"]
        self.ho_in, self.ho_out = self.st["ho_in"], self.st["ho_out"]
        self.W_refs = self.st["W_refs"]
        self.layers = list(layers)
        self.b = b
        self.cal_in_d = self.st["cal_in"].to("cuda")
        self.cal_out_d = self.st["cal_out"].to("cuda")
        C.GUARD.assert_not_heldout(self.cal_in_d, self.cal_out_d, where="Runner.calibration")
        self.snapB = torch.load(f"{H.GCACHE}/snapshot_B.pt", weights_only=False)
        self.snapA = torch.load(f"{H.GCACHE}/snapshot_A.pt", weights_only=False)
        self.arm3_order = H.gauge_batch_order(ARM3_STEPS)

    # -- state construction ------------------------------------------------
    def new_block(self):
        blk = H.fresh_block()
        sub = H.nq_sub(blk)
        bases = {n: C.LayerBase(n, sub[n]).to("cuda") for n in self.layers}
        imp = sub["mlp.down_proj"].o_norm.to("cuda")
        for _, m in D.nq_modules(blk):
            for p in m.parameters():
                p.requires_grad_(False)
        return blk, sub, bases, imp

    # -- measurement -------------------------------------------------------
    def block_err(self, blk):
        return I.block_err(blk, self.ho_in, self.ho_out, self.kwargs)

    def calib_func_loss(self, blk):
        C.GUARD.assert_not_heldout(self.cal_in_d, where="calib_func_loss")
        return H.func_loss_over(blk, self.cal_in_d, self.cal_out_d, self.kwargs)

    # -- the identical common Step 3 --------------------------------------
    def finish(self, rec, blk, sub, imp, tag):
        """E_gauge -> common Step 3 -> E_final, plus every diagnostic."""
        rec["E_gauge"] = self.block_err(blk)
        rec["J_pre"] = H.J_table(blk, self.W_refs, key="pre")
        init_signs = H.snapshot_signs(blk)
        init_bin = H.binary_of(blk)
        rec["calib_func_loss_at_gauge"] = self.calib_func_loss(blk)

        s3 = H.common_step3(blk, self.qd, self.cal_in_d, self.cal_out_d, imp,
                            self.kwargs, self.snapB, tag=tag)
        rec["step3"] = {k: v for k, v in s3.items() if k != "flips"}
        rec["step3_flips"] = s3["flips"]
        rec["E_final"] = self.block_err(blk)
        Jp = H.J_table(blk, self.W_refs, key="post")
        for n in Jp:
            rec["J_pre"][n].update(Jp[n])
        rec["J"] = rec.pop("J_pre")
        final_signs = H.snapshot_signs(blk)
        rec["step3_sign_delta"] = H.sign_delta(init_signs, final_signs)
        tuned_bin = H.binary_of(blk)
        rec["layer_post_curve"] = H.layer_post_curve(blk, init_bin, tuned_bin,
                                                     self.ho_in, self.ho_out, self.kwargs)
        return rec

    # -- arms --------------------------------------------------------------
    def arm0a(self, tag="0a"):
        t0 = time.time()
        blk, sub, bases, imp = self.new_block()
        base = bases[LAYER]
        rec = {"arm": tag, "desc": "baseline: R = I through the identical materialise/export path",
               "layer": LAYER, "E_ADMM": self.block_err(blk)}
        Is = C.identity_Rs(base.rank, self.b, "cuda")
        rec["materialize"] = H.materialize_gauge(sub[LAYER], base, Is, check_identity=True)
        rec["E_gauge_pre_export"] = rec_pre_export(self, blk, sub, base, Is)
        # restore the true identity state after the pre-export probe
        H.materialize_gauge(sub[LAYER], base, Is, check_identity=False)
        rec["gauge_sign_delta_U"] = rec["materialize"]["sign_delta_U"]
        rec["gauge_sign_delta_V"] = rec["materialize"]["sign_delta_V"]
        self.finish(rec, blk, sub, imp, tag)
        rec["wall_s"] = time.time() - t0
        H.save_run(rec, f"arm{tag}")
        return rec

    def arm0b(self, n_steps):
        """Step/data-matched extra-STE control.  Ordinary NanoQuant latent update
        (hard-sign STE, NanoQuant's own importance-weighted loss and AdamW at
        fact_binary_lr / fact_scale_lr), on down_proj only, over exactly the
        calibration batches Arm 3 uses for its first n_steps steps.

        This is NOT compute matched: the gauge path may cost more per step."""
        t0 = time.time()
        blk, sub, bases, imp = self.new_block()
        base = bases[LAYER]
        m = sub[LAYER]
        rec = {"arm": f"0b_{n_steps}", "layer": LAYER, "extra_ste_steps": n_steps,
               "desc": "ordinary NanoQuant hard-sign STE, step/data matched to Arm 3",
               "E_ADMM": self.block_err(blk), "batch_order": self.arm3_order[:n_steps]}

        for n_, p in m.named_parameters():
            if "latent" in n_ or "scale" in n_ or "bias" in n_:
                p.requires_grad_(True)
        cfg = get_param_group_config(m, binary_lr=self.qd["fact_binary_lr"],
                                     scale_lr=self.qd["fact_scale_lr"],
                                     bias_lr=self.qd["fact_bias_lr"])
        opt = AdamW(cfg, weight_decay=0)
        C.GUARD.assert_not_heldout(self.cal_in_d, self.cal_out_d, where=f"arm0b_{n_steps}")
        losses = []
        for s in range(n_steps):
            j = self.arm3_order[s]
            y = blk(self.cal_in_d[j:j + 1], **self.kwargs)[0]
            loss = fused_weighted_mse(y, self.cal_out_d[j:j + 1], imp)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()) / self.cal_out_d[0].numel())
        rec["extra_ste_loss"] = {"first": losses[0], "last": losses[-1],
                                 "trace": losses[::max(1, n_steps // 20)]}
        del opt                                     # extra-phase optimiser discarded
        blk.zero_grad(set_to_none=True)
        for _, mm in D.nq_modules(blk):
            for p in mm.parameters():
                p.requires_grad_(False)
        cleanup_memory()

        # sign movement of the extra STE phase, against the original ADMM signs
        cur_U = C.pos_sign(m.U_latent.data.float())
        cur_V = C.pos_sign(m.V_latent.data.float())
        rec["gauge_sign_delta_U"] = (cur_U != C.pos_sign(base.U0)).float().mean().item()
        rec["gauge_sign_delta_V"] = (cur_V != C.pos_sign(base.V0)).float().mean().item()
        rec["E_gauge_pre_export"] = None    # no export re-extraction happens in this arm
        self.finish(rec, blk, sub, imp, f"0b_{n_steps}")
        rec["wall_s"] = time.time() - t0
        H.save_run(rec, f"arm0b_{n_steps}")
        return rec

    def arm1(self, n=ARM1_SAMPLES, seed=ARM1_SEED):
        """Random gauge sensitivity control -- a diagnostic, no Step 3."""
        t0 = time.time()
        blk, sub, bases, imp = self.new_block()
        base = bases[LAYER]
        rec = {"arm": "1", "layer": LAYER, "n_samples": n, "seed": seed,
               "block_size": self.b, "E_ADMM": self.block_err(blk), "samples": []}
        g = torch.Generator(device="cuda"); g.manual_seed(seed)
        Is = C.identity_Rs(base.rank, self.b, "cuda")
        rec["E_gauge_identity"] = None
        H.materialize_gauge(sub[LAYER], base, Is, check_identity=True)
        rec["E_gauge_identity"] = self.block_err(blk)
        for i in range(n):
            Rs = [C.haar_so(s, g) for s in C.block_sizes(base.rank, self.b)]
            d = H.materialize_gauge(sub[LAYER], base, Rs)
            e = self.block_err(blk)
            H.apply_gauge_signs_only(sub[LAYER], base, Rs)
            e_pre = self.block_err(blk)
            H.materialize_gauge(sub[LAYER], base, Rs)
            rec["samples"].append({"id": i, "E_gauge": e, "E_gauge_pre_export": e_pre,
                                   "sign_delta_U": d["sign_delta_U"], "sign_delta_V": d["sign_delta_V"]})
            print(f"[arm1] {i:02d}  E_gauge = {e:.6e}  pre_export = {e_pre:.6e}  "
                  f"dU = {d['sign_delta_U']:.4f}  dV = {d['sign_delta_V']:.4f}", flush=True)
        vals = torch.tensor([s["E_gauge"] for s in rec["samples"]])
        rec["stats"] = {"min": vals.min().item(), "median": vals.median().item(),
                        "max": vals.max().item(), "std": vals.std().item(),
                        "best_id": int(vals.argmin().item())}
        rec["wall_s"] = time.time() - t0
        H.save_run(rec, "arm1")
        return rec

    def arm2(self, variant, rounds=ITQ_ROUNDS):
        """Joint-ITQ-inspired latent alignment, absolute gauge (section 11)."""
        t0 = time.time()
        blk, sub, bases, imp = self.new_block()
        base = bases[LAYER]
        m = sub[LAYER]
        rec = {"arm": f"2_{variant}", "layer": LAYER, "variant": variant, "rounds": rounds,
               "block_size": self.b, "E_ADMM": self.block_err(blk), "trace": []}
        sizes = C.block_sizes(base.rank, self.b)
        if variant.endswith("_rand"):
            # Classic ITQ starts from a *random* rotation.  Starting at R = I is a
            # trivial fixed point of the Procrustes step (the sign-magnitude target
            # is built from X0 in its own basis), so this cell asks whether the
            # alignment finds anything when it is not started at the answer.
            gi = torch.Generator(device="cuda"); gi.manual_seed(ITQ_INIT_SEED)
            Rs = [C.haar_so(sz, gi) for sz in sizes]
            rec["init"] = f"haar_so seed {ITQ_INIT_SEED}"
        else:
            Rs = C.identity_Rs(base.rank, self.b, "cuda")
            rec["init"] = "identity"
        V0m = base.V0.transpose(0, 1).contiguous()          # math V0 : [in, rank]
        with torch.no_grad():
            for t in range(rounds):
                U_R, V_R = C.gauge(base, Rs)
                B_U, B_V, sp, so = C.q_nq(U_R, V_R, base)
                if variant.startswith("frozen"):
                    M_U, M_V = base.so0, base.sp0           # ADMM magnitude targets, fixed
                else:
                    M_U, M_V = so, sp                       # refreshed through Q_NQ
                T_U = M_U.unsqueeze(1) * B_U                # [out, rank]
                T_V = M_V.unsqueeze(1) * B_V.transpose(0, 1)  # [in, rank]
                new, off = [], 0
                with C.no_tf32():
                    for b_ in sizes:
                        X0 = torch.cat([base.U0[:, off:off + b_], V0m[:, off:off + b_]], dim=0)
                        Tk = torch.cat([T_U[:, off:off + b_], T_V[:, off:off + b_]], dim=0)
                        M = X0.transpose(0, 1) @ Tk
                        P, _, Qh = torch.linalg.svd(M.double(), full_matrices=False)
                        new.append((P @ Qh).float())
                        off += b_
                Rs = new
                orth = max((R.T @ R - torch.eye(R.shape[0], device=R.device)).norm().item() for R in Rs)
                assert orth < 1e-4, f"ITQ rotation lost orthogonality: {orth:.2e}"
                H.materialize_gauge(m, base, Rs)
                e = self.block_err(blk)
                rec["trace"].append({"round": t + 1, "E_gauge": e, "orth_err": orth})
                print(f"[arm2/{variant}] round {t+1:02d}  E_gauge = {e:.6e}", flush=True)
                del U_R, V_R, B_U, B_V, T_U, T_V
                torch.cuda.empty_cache()

        H.apply_gauge_signs_only(m, base, Rs)
        rec["E_gauge_pre_export"] = self.block_err(blk)
        d = H.materialize_gauge(m, base, Rs)
        rec["materialize"] = d
        rec["gauge_sign_delta_U"] = d["sign_delta_U"]
        rec["gauge_sign_delta_V"] = d["sign_delta_V"]
        self.finish(rec, blk, sub, imp, f"2_{variant}")
        rec["wall_s"] = time.time() - t0
        H.save_run(rec, f"arm2_{variant}")
        return rec

    def arm3(self, lr=ARM3_LR, tag="3", probe_every=25):
        """Functional latent gauge -- the proposed method (section 12)."""
        t0 = time.time()
        blk, sub, bases, imp = self.new_block()
        base = bases[LAYER]
        m = sub[LAYER]
        rec = {"arm": tag, "layer": LAYER, "steps": ARM3_STEPS, "lr": lr,
               "block_size": self.b, "optimizer": "Adam",
               "E_ADMM": self.block_err(blk), "trace": [], "checkpoints": {},
               "calib_probe": []}
        cay = C.BlockCayley(base.rank, b=self.b, device="cuda")
        gf = H.GaugedForward(m, base, cay)
        opt = torch.optim.Adam(cay.parameters(), lr=lr)
        C.GUARD.assert_not_heldout(self.cal_in_d, self.cal_out_d, where="arm3_gauge_search")
        saved = {}
        for s in range(ARM3_STEPS):
            j = self.arm3_order[s]
            loss = H.func_loss(blk, self.cal_in_d[j:j + 1], self.cal_out_d[j:j + 1], self.kwargs)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            orth = cay.orthogonality_error()
            assert orth < 1e-5, f"Cayley lost orthogonality at step {s+1}: {orth:.2e}"
            rec["trace"].append({"step": s + 1, "loss": float(loss.detach()), "orth_err": orth})
            if probe_every and (s + 1) % probe_every == 0:
                with torch.no_grad():
                    cl = self.calib_func_loss(blk)
                rec["calib_probe"].append({"step": s + 1, "calib_L_func": cl})
                print(f"[{tag}] step {s+1:03d}  batch L = {float(loss.detach()):.6e}  "
                      f"calib L_func = {cl:.6e}  orth = {orth:.2e}", flush=True)
            if (s + 1) in ARM3_CKPTS:
                saved[s + 1] = [R.detach().clone() for R in cay.Rs()]
                # section 4.2 logging: functional loss before / after export re-extraction
                with torch.no_grad(), H._FixedScaleSwap(gf):
                    pre_ex = self.calib_func_loss(blk)
                with torch.no_grad():
                    post_ex = self.calib_func_loss(blk)
                rec["checkpoints"][str(s + 1)] = {
                    "calib_func_loss_pre_export_stats": pre_ex,
                    "calib_func_loss_post_export_stats": post_ex,
                }
                print(f"[{tag}] ckpt {s+1}: calib L_func pre-export {pre_ex:.6e} -> "
                      f"post-export {post_ex:.6e}", flush=True)
        gf.remove()
        del opt, cay
        cleanup_memory()
        rec["gauge_search_s"] = time.time() - t0
        H.save_run(rec, f"arm{tag}_search")

        out = {"search": rec, "checkpoints": {}}
        for k, Rs in saved.items():
            t1 = time.time()
            blk, sub, bases, imp = self.new_block()
            base = bases[LAYER]
            r = {"arm": f"{tag}_{k}", "layer": LAYER, "gauge_steps": k, "lr": lr,
                 "block_size": self.b, "E_ADMM": self.block_err(blk)}
            r.update(rec["checkpoints"][str(k)])
            H.apply_gauge_signs_only(sub[LAYER], base, Rs)
            r["E_gauge_pre_export"] = self.block_err(blk)
            d = H.materialize_gauge(sub[LAYER], base, Rs)
            r["materialize"] = d
            r["gauge_sign_delta_U"] = d["sign_delta_U"]
            r["gauge_sign_delta_V"] = d["sign_delta_V"]
            self.finish(r, blk, sub, imp, f"3_{k}")
            r["wall_s"] = time.time() - t1
            H.save_run(r, f"arm{tag}_{k}")
            out["checkpoints"][str(k)] = r
        torch.save({str(k): [R.cpu() for R in v] for k, v in saved.items()},
                   f"{H.GCACHE}/arm{tag}_R.pt")
        return out


    def arm4(self, tag="4", fixed_scales=False, strategy="cyclic", suffix=""):
        """Discrete Givens coordinate-descent gauge -> the identical common Step 3.

        The rotation comes from `nqx/gauge/givens.py descent`, which searched the
        true binary objective at its sign-pattern breakpoints -- no STE anywhere.
        With `fixed_scales` the ADMM export scales are kept instead of being
        recomputed by Q_NQ; Stage 1 showed the recomputation costs the gauge arms,
        and Step 3 re-learns both scale vectors regardless, so both conventions
        are run and reported.
        """
        t0 = time.time()
        blob = torch.load(f"{H.GCACHE}/givens_R_{strategy}{suffix}.pt", weights_only=False)
        R = blob["R"].to("cuda").float()
        blk, sub, bases, imp = self.new_block()
        base = bases[LAYER]
        m = sub[LAYER]
        with torch.no_grad(), C.no_tf32():
            I_ = torch.eye(R.shape[0], device=R.device)
            orth = float((R.T @ R - I_).norm())
        assert orth < 1e-4, f"accumulated Givens rotation is not orthogonal: {orth:.2e}"
        rec = {"arm": tag, "layer": LAYER, "desc": "discrete Givens coordinate descent",
               "rank_blocks": blob.get("blocks"), "strategy": strategy,
               "sweeps_done": blob.get("sweeps_done"),
               "fixed_scales": fixed_scales, "R_orth_err": orth,
               "E_ADMM": self.block_err(blk)}
        H.apply_gauge_signs_only(m, base, [R])
        rec["E_gauge_pre_export"] = self.block_err(blk)
        if fixed_scales:
            d = {"sign_delta_U": float((C.pos_sign(base.U0 @ R) != C.pos_sign(base.U0))
                                       .float().mean()),
                 "sign_delta_V": float((C.pos_sign(R.T @ base.V0) != C.pos_sign(base.V0))
                                       .float().mean())}
        else:
            d = H.materialize_gauge(m, base, [R])
        rec["materialize"] = d
        rec["gauge_sign_delta_U"] = d["sign_delta_U"]
        rec["gauge_sign_delta_V"] = d["sign_delta_V"]
        self.finish(rec, blk, sub, imp, tag)
        rec["wall_s"] = time.time() - t0
        H.save_run(rec, f"arm{tag}")
        return rec


def rec_pre_export(runner, blk, sub, base, Rs):
    H.apply_gauge_signs_only(sub[LAYER], base, Rs)
    return runner.block_err(blk)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(f"{H.GCACHE}/runs", exist_ok=True)
    r = Runner(seed=a.seed)
    for arm in a.arms:
        print(f"\n===== ARM {arm} =====", flush=True)
        if arm == "0a":
            out = r.arm0a()
        elif arm.startswith("0a_r"):
            out = r.arm0a(tag=arm)
        elif arm == "0a_dup":
            out = r.arm0a(tag="0a_dup")
        elif arm.startswith("0b_"):
            out = r.arm0b(int(arm.split("_")[1]))
        elif arm == "1":
            out = r.arm1()
        elif arm.startswith("2_"):
            out = r.arm2(arm.split("_", 1)[1])
        elif arm == "3":
            out = r.arm3()
        elif arm.startswith("4"):
            st_ = "greedy" if "greedy" in arm else "cyclic"
            fs_ = arm.endswith("fs")
            sx_ = ""
            for k in ["b0123", "all50", "rand1", "fp64", "spread"]:
                if k in arm:
                    sx_ = "_" + k
            out = r.arm4(tag=f"4_{st_}{sx_}" + ("_fixedscale" if fs_ else ""),
                         fixed_scales=fs_, strategy=st_, suffix=sx_)
        elif arm.startswith("3lr"):
            lr = float(arm[3:])
            out = r.arm3(lr=lr, tag=f"3lr{arm[3:]}")
        else:
            raise SystemExit(f"unknown arm {arm}")
        keys = ["arm", "E_ADMM", "E_gauge_pre_export", "E_gauge", "E_final", "wall_s"]
        if isinstance(out, dict) and "E_final" in out:
            print("[arm] FINAL " + json.dumps({k: out.get(k) for k in keys}), flush=True)
        elif arm == "3":
            for k, v in out["checkpoints"].items():
                print("[arm] FINAL " + json.dumps({kk: v.get(kk) for kk in keys}), flush=True)
        else:
            print("[arm] FINAL " + json.dumps(out.get("stats", {})), flush=True)


if __name__ == "__main__":
    main()
