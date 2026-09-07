"""Do genuinely block-beneficial single gauge moves exist in a nonlinear layer?
And does the cheap linear score have ranking power over them?

For each of N Givens planes we take the candidate the layer surrogate would
choose, apply that ONE move from the original NanoQuant state, and measure the
TRUE calibration block error.  Every move is measured against the same baseline
and rolled back, so they never compound.

For each move we record the exact decomposition of the true change,

    dE = ( 2<R, dY> + ||dY||^2 ) / ||Y_fp||^2,   R = Y_q - Y_fp,  dY = Y - Y_q

-- so `lin_exact` is the true compensation term and `quad_exact` the true
curvature term, with no linearisation anywhere -- alongside the cheap scores
S3 = <G_W, dW> (the linearised compensation term) and S1.

That answers two things at once:
  * how many single moves are genuinely beneficial (the existence question), and
  * whether S3 ranks them, even though its absolute error floor is too large to
    use as an accept/reject rule.
"""
import argparse
import json
import os
import time

import torch

from nanoquant.modules.hub import NanoQuantConfigDataclass  # noqa: F401  import-order guard

import instrument as I

from . import block_surrogate as BS
from . import core as C
from . import curvature as CV
from . import givens as GV
from . import harness as H
from . import layer_oracle as LO
from . import state as S

OUT = "/home/work/exp/results/nanoquant/gauge/curvature"


def main(a):
    os.makedirs(OUT, exist_ok=True)
    st = H.load_post_admm(seed=a.seed)
    blk, kwargs = st["blk"], st["kwargs"]
    sub = H.nq_sub(blk)
    cal_in, cal_out = st["cal_in"].to("cuda"), st["cal_out"].to("cuda")
    C.GUARD.assert_not_heldout(cal_in, cal_out, where="topk_probe")
    name = a.layer
    base = C.LayerBase(name, sub[name]).to("cuda")
    m = sub[name]

    # baseline block outputs, cached once
    Yb, Y_sq, num0 = [], 0.0, 0.0
    with torch.no_grad():
        for j in range(cal_in.shape[0]):
            y = blk(cal_in[j:j + 1], **kwargs)[0]
            Yb.append(y.clone())
            t = cal_out[j:j + 1]
            num0 += float((y.double() - t.double()).square().sum())
            Y_sq += float(t.double().square().sum())
    E0 = num0 / Y_sq
    print(f"[topk] baseline block error {E0:.8f}   (Y_sq {Y_sq:.4e})", flush=True)

    curv = CV.collect(st, layers=[name], n_seq=a.n_seq)[name].to("cuda")
    o = LO.build_all(st, n_seq=a.n_seq)[name].to("cuda")
    W_base = I.effective_weight(m).double()
    surro0 = o.error(W_base.float())

    rank = base.rank
    g = torch.Generator(device="cpu"); g.manual_seed(a.seed)
    planes = []
    while len(planes) < a.n_planes:
        i = int(torch.randint(0, rank, (1,), generator=g))
        j = int(torch.randint(0, rank, (1,), generator=g))
        if i != j and (min(i, j), max(i, j)) not in planes:
            planes.append((min(i, j), max(i, j)))

    ps = GV.PlaneSearcher(o, base)
    rec = {"layer": name, "E0": E0, "Y_sq": Y_sq, "n_planes": a.n_planes, "moves": []}
    t0 = time.time()
    for n_, (i, j) in enumerate(planes):
        r = ps.sweep(i, j)
        if not r["improves"]:
            continue
        R_one = ps.R.clone()
        ps.apply(i, j, r["theta"])
        H.apply_gauge_signs_only(m, base, [ps.R.float()])
        W_c = I.effective_weight(m).double()
        dW = W_c - W_base

        num, lin, quad = 0.0, 0.0, 0.0
        with torch.no_grad():
            for k in range(cal_in.shape[0]):
                y = blk(cal_in[k:k + 1], **kwargs)[0]
                t = cal_out[k:k + 1].double()
                dY = y.double() - Yb[k].double()
                Rres = Yb[k].double() - t
                num += float((y.double() - t).square().sum())
                lin += float(2.0 * (Rres * dY).sum())
                quad += float(dY.square().sum())
        true_d = num / Y_sq - E0
        ent = {"plane": [i, j], "theta": r["theta"], "n_flip": r["n_flip"],
               "surrogate_delta": r["delta_f"],
               "true_delta": true_d, "lin_exact": lin / Y_sq, "quad_exact": quad / Y_sq,
               "S3": BS.s3_block_linear(dW, curv.G_W),
               "S1": BS.s1_input_quadratic(dW, curv.H_in),
               "beneficial": bool(true_d < 0)}
        rec["moves"].append(ent)
        # roll back
        with torch.no_grad():
            ps.R = R_one
            ps.U = base.U0.double() @ ps.R
            ps.Vm = base.V0.transpose(0, 1).contiguous().double() @ ps.R
        ps.refresh()
        H.materialize_gauge(m, base, C.identity_Rs(rank, 32, "cuda"))
        if (len(rec["moves"])) % 10 == 0:
            nb = sum(x["beneficial"] for x in rec["moves"])
            print(f"[topk] {len(rec['moves'])} moves  beneficial {nb}  "
                  f"best {min(x['true_delta'] for x in rec['moves']):+.3e}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            S.atomic_json(rec, f"{OUT}/topk_{name.replace('.','_')}.json")
    rec["wall_s"] = time.time() - t0
    S.atomic_json(rec, f"{OUT}/topk_{name.replace('.','_')}.json")
    nb = sum(x["beneficial"] for x in rec["moves"])
    print(f"[topk] DONE {len(rec['moves'])} moves, {nb} beneficial "
          f"({100*nb/max(len(rec['moves']),1):.1f}%)", flush=True)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="self_attn.v_proj")
    ap.add_argument("--n_planes", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_seq", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--noise_k", type=float, default=10.0)
    a = ap.parse_args()
    a.n_seq = a.n_seq or None
    main(a)
