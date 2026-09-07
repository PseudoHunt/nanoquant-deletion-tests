"""Mechanism check for Arm 3: is R = I a local minimum of the binary functional
error within the gauge class, or is the STE gradient simply being over-stepped?

Accumulates the full-calibration functional gradient at R = I, then evaluates the
*true* calibration functional loss along the steepest-descent ray and along
random tangent directions.  Calibration data only; the held-out split is never
touched.
"""
import argparse
import json
import time

import torch

from nanoquant.modules.hub import NanoQuantConfigDataclass  # noqa: F401  import-order guard

from . import core as C
from . import harness as H
from . import state as S

LAYER = "mlp.down_proj"
TS = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_grad", type=int, default=128, help="calibration sequences for the gradient")
    ap.add_argument("--n_rand", type=int, default=3)
    ap.add_argument("--b", type=int, default=32)
    a = ap.parse_args()

    st = H.load_post_admm(seed=0)
    blk, kwargs = st["blk"], st["kwargs"]
    sub = H.nq_sub(blk)
    base = C.LayerBase(LAYER, sub[LAYER]).to("cuda")
    cal_in = st["cal_in"].to("cuda"); cal_out = st["cal_out"].to("cuda")
    C.GUARD.assert_not_heldout(cal_in, cal_out, where="linesearch")
    for _, m in blk.named_modules():
        for p in m.parameters(recurse=False):
            p.requires_grad_(False)

    cay = C.BlockCayley(base.rank, b=a.b, device="cuda")
    gf = H.GaugedForward(sub[LAYER], base, cay)

    t0 = time.time()
    for j in range(a.n_grad):
        loss = H.func_loss(blk, cal_in[j:j + 1], cal_out[j:j + 1], kwargs)
        (loss / a.n_grad).backward()
    g = [p.grad.detach().clone() for p in cay.groups]
    gnorm = torch.sqrt(sum((x * x).sum() for x in g)).item()
    for p in cay.groups:
        p.grad = None
    print(f"[ls] full-calibration gradient accumulated in {time.time()-t0:.0f}s, "
          f"||g||_F = {gnorm:.6e}", flush=True)

    def eval_at(dirs, t):
        with torch.no_grad():
            for p, d in zip(cay.groups, dirs):
                p.copy_(t * d)
        with torch.no_grad():
            post = H.func_loss_over(blk, cal_in, cal_out, kwargs)
            with H._FixedScaleSwap(gf):
                pre = H.func_loss_over(blk, cal_in, cal_out, kwargs)
            orth = cay.orthogonality_error()
        return post, pre, orth

    desc = [-x / gnorm for x in g]           # unit steepest-descent direction
    rec = {"grad_norm": gnorm, "n_grad": a.n_grad, "block_size": a.b,
           "descent": [], "random": []}
    for t in TS:
        post, pre, orth = eval_at(desc, t)
        rec["descent"].append({"t": t, "calib_L_post_export": post,
                               "calib_L_pre_export": pre, "orth_err": orth})
        print(f"[ls] descent t = {t:<8g}  L(post-export) = {post:.6f}  "
              f"L(pre-export) = {pre:.6f}", flush=True)

    gg = torch.Generator(device="cuda"); gg.manual_seed(31337)
    for r in range(a.n_rand):
        dirs = [torch.randn(p.shape, generator=gg, device=p.device) for p in cay.groups]
        n = torch.sqrt(sum((x * x).sum() for x in dirs)).item()
        dirs = [x / n for x in dirs]
        row = []
        for t in TS:
            post, pre, _ = eval_at(dirs, t)
            row.append({"t": t, "calib_L_post_export": post, "calib_L_pre_export": pre})
        rec["random"].append(row)
        print(f"[ls] random dir {r}: " + "  ".join(f"{x['t']:g}:{x['calib_L_post_export']:.4f}"
                                                   for x in row), flush=True)

    gf.remove()
    S.atomic_json(rec, f"{H.GCACHE}/linesearch.json")
    print("[ls] done")


if __name__ == "__main__":
    main()
