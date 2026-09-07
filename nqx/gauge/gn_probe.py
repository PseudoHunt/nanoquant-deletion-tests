"""Directional Gauss-Newton probe on one layer.

For a perturbation dW, the true change in block error is exactly

    dE(t) = ( 2<R, dY(t)> + ||dY(t)||^2 ) / ||Y_fp||^2,   dY(t) = Y(W + t dW) - Y(W)

with R = Y_q - Y_fp.  The Gauss-Newton prediction replaces the true dY by its
linearisation J dW.  So instead of forming J at all, walk the ray W + t dW and
fit

    E(t) = E(0) + a t + b t^2 + ...

The linear coefficient `a` IS the S3 term (2<R, J dW>/||Y_fp||^2) and the
quadratic coefficient `b` IS the directional GN curvature (||J dW||^2/||Y_fp||^2),
both read off the small-t behaviour where the linearisation is valid.  The GN
prediction for the full endpoint is then a + b, against the measured E(1) - E(0).

This answers two things at once: whether GN gets the sign right, and -- by
comparing the fit to the measured curve at t = 1 -- whether a single local
quadratic survives the full accumulated move or only small bundles.

The layer is temporarily swapped for a dense nn.Linear so intermediate t are
well defined; t = 0 reproduces the deployed weight, and every number is quoted
relative to that dense baseline so the ray is a smooth function of t.
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn

from nanoquant.modules.hub import NanoQuantConfigDataclass  # noqa: F401  import-order guard

import instrument as I

from . import core as C
from . import harness as H
from . import state as S

OUT = "/home/work/exp/results/nanoquant/gauge/curvature"


class DenseSwap:
    """Temporarily replace a NanoQuantLinear with a dense nn.Linear."""

    def __init__(self, blk, name):
        self.parent = blk
        parts = name.split(".")
        for p in parts[:-1]:
            self.parent = getattr(self.parent, p)
        self.attr = parts[-1]
        self.orig = getattr(self.parent, self.attr)

    def set_weight(self, W):
        lin = nn.Linear(W.shape[1], W.shape[0], bias=False,
                        device=W.device, dtype=torch.bfloat16)
        lin.weight.data.copy_(W.to(torch.bfloat16))
        lin.weight.requires_grad_(False)
        setattr(self.parent, self.attr, lin)

    def restore(self):
        setattr(self.parent, self.attr, self.orig)


def main(a):
    os.makedirs(OUT, exist_ok=True)
    st = H.load_post_admm(seed=a.seed)
    blk, kwargs = st["blk"], st["kwargs"]
    sub = H.nq_sub(blk)
    cal_in, cal_out = st["cal_in"].to("cuda"), st["cal_out"].to("cuda")
    C.GUARD.assert_not_heldout(cal_in, cal_out, where="gn_probe")
    name = a.layer
    base = C.LayerBase(name, sub[name]).to("cuda")

    def err():
        return H.func_loss_over(blk, cal_in, cal_out, kwargs)

    E_nq = err()
    W_base = I.effective_weight(sub[name]).double()
    swap = DenseSwap(blk, name)

    from . import givens as GV
    from . import layer_oracle as LO
    t0 = time.time()
    if a.R_from:
        # probe an explicit saved rotation (e.g. a KEEP from the gated run)
        blob = torch.load(a.R_from, weights_only=False)
        R0 = (blob[name] if name in blob else blob["R"]).to("cuda").double()
        doses, snaps = [0], {0: R0}
        print(f"[gn] using saved rotation from {os.path.basename(a.R_from)} "
              f"(|R-I| = {float((R0 - torch.eye(R0.shape[0], device=R0.device, dtype=R0.dtype)).norm()):.3e})",
              flush=True)
        o = None
    else:
        o = LO.build_all(st, n_seq=a.n_seq)[name].to("cuda")
        ps = GV.PlaneSearcher(o, base)
        surro0 = o.error(W_base.float())
        doses, snaps = a.doses, {}
    for gi in (range(max(doses)) if o is not None else []):
        gp = list(range(gi * 32, gi * 32 + 32))
        for x in range(32):
            for y in range(x + 1, 32):
                r = ps.sweep(gp[x], gp[y])
                if GV._accept_ok(r, o, max(surro0, 1e-12), a):
                    ps.apply(r["i"], r["j"], r["theta"])
        if (gi + 1) in doses:
            snaps[gi + 1] = ps.R.clone()
            print(f"[gn] snapshot at {gi+1} rank block(s)  ({time.time()-t0:.0f}s)", flush=True)
    if o is not None:
        del ps

    rec = {"layer": name, "E_nanoquant": E_nq, "doses": doses,
           "ts": a.ts, "rows": []}
    for d in doses:
        R = snaps[d]
        H.apply_gauge_signs_only(sub[name], base, [R.float()])
        W_g = I.effective_weight(sub[name]).double()
        H.materialize_gauge(sub[name], base, C.identity_Rs(base.rank, 32, "cuda"))
        dW = W_g - W_base

        curve = {}
        for t in a.ts:
            swap.set_weight(W_base + t * dW)
            curve[t] = err()
        swap.restore()

        E0d = curve[0.0]
        # fit a, b from the two smallest non-zero t (local quadratic)
        t1, t2 = a.ts[1], a.ts[2]
        y1, y2 = curve[t1] - E0d, curve[t2] - E0d
        b = (y2 / t2 - y1 / t1) / (t2 - t1)
        aa = y1 / t1 - b * t1
        true_d = curve[1.0] - E0d
        rec["rows"].append({
            "dose_rank_blocks": d, "E_dense_t0": E0d, "curve": {str(k): v for k, v in curve.items()},
            "a_linear": aa, "b_gn_curvature": b, "gn_prediction": aa + b,
            "true_delta": true_d,
            "gn_correct_sign": bool((aa + b > 0) == (true_d > 0)),
            "linear_correct_sign": bool((aa > 0) == (true_d > 0)),
            "gn_rel_err": abs(aa + b - true_d) / max(abs(true_d), 1e-30)})
        r = rec["rows"][-1]
        print(f"[gn] dose {d:2d} blk  a(linear) {aa:+.4e}  b(GN curv) {b:+.4e}  "
              f"a+b {aa+b:+.4e}  true {true_d:+.4e}  "
              f"GN {'OK' if r['gn_correct_sign'] else 'WRONG'}  "
              f"lin {'OK' if r['linear_correct_sign'] else 'WRONG'}  "
              f"relerr {r['gn_rel_err']:.2f}", flush=True)
        S.atomic_json(rec, f"{OUT}/gn_probe_{name.replace('.','_')}.json")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="self_attn.v_proj")
    ap.add_argument("--doses", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--ts", type=float, nargs="+", default=[0.0, 0.125, 0.25, 0.5, 1.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_seq", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--noise_k", type=float, default=10.0)
    ap.add_argument("--R_from", default="", help="probe an explicit saved rotation")
    a = ap.parse_args()
    a.n_seq = a.n_seq or None
    main(a)
