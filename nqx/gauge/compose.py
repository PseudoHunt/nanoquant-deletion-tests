"""Sequential composition: does beneficial gauge capacity harvest?

    exact Givens enumeration -> cheap S3 ranking -> true block loss on top-K
                             -> accept only a real improvement

S3 is used only as a *ranker* (Spearman +0.62 against true dE on single moves);
every accept/reject decision is made by the true calibration block error, so a
mis-ranked candidate costs an evaluation and nothing else.  The block-residual
gradient G_W is recomputed after every accepted move, since accepting changes the
residual the next move's compensation term is measured against.

S3 for every candidate in a plane is one batched matmul, because for a Givens
move only rank coordinates i and j change:

    <G_W, dW> = sum_k [ b_k' G~ c_k' - b_k G~ c_k ],    G~ = diag(so) G_W diag(sp)
"""
import argparse
import json
import math
import os
import time

import torch

from nanoquant.modules.hub import NanoQuantConfigDataclass  # noqa: F401  import-order guard

import instrument as I

from . import core as C
from . import curvature as CV
from . import givens as GV
from . import harness as H
from . import layer_oracle as LO
from . import state as S

OUT = "/home/work/exp/results/nanoquant/gauge/curvature"


@torch.no_grad()
def s3_candidates(ps, i, j, Gt):
    """S3 for every distinct binary state this plane can reach, plus the angles."""
    u_i, u_j = ps.U[:, i], ps.U[:, j]
    v_i, v_j = ps.Vm[:, i], ps.Vm[:, j]
    bp = torch.cat([GV.breakpoints(u_i, u_j), GV.breakpoints(v_i, v_j)])
    bp = torch.sort(torch.unique(bp)).values
    mids = torch.cat([(bp[:-1] + bp[1:]) / 2,
                      torch.tensor([(bp[-1] + bp[0] + GV.HALF_PI) / 2 % GV.HALF_PI],
                                   device=bp.device)])
    mids = torch.cat([torch.zeros(1, device=bp.device), mids])
    out = []
    for s in range(0, mids.numel(), ps.chunk):
        th = mids[s:s + ps.chunk]
        cs, sn = torch.cos(th).unsqueeze(1), torch.sin(th).unsqueeze(1)
        with C.no_tf32():
            Bi = C.pos_sign(cs * u_i + sn * u_j)
            Bj = C.pos_sign(-sn * u_i + cs * u_j)
            Ci = C.pos_sign(cs * v_i + sn * v_j)
            Cj = C.pos_sign(-sn * v_i + cs * v_j)
            val = (Bi * (Gt @ Ci.transpose(0, 1)).transpose(0, 1)).sum(1) \
                + (Bj * (Gt @ Cj.transpose(0, 1)).transpose(0, 1)).sum(1)
        out.append(val)
    v = torch.cat(out)
    return mids, v - v[0]          # S3 relative to the current state


def main(a):
    os.makedirs(OUT, exist_ok=True)
    st = H.load_post_admm(seed=a.seed)
    blk, kwargs = st["blk"], st["kwargs"]
    sub = H.nq_sub(blk)
    cal_in, cal_out = st["cal_in"].to("cuda"), st["cal_out"].to("cuda")
    C.GUARD.assert_not_heldout(cal_in, cal_out, where="compose")
    name = a.layer
    base = C.LayerBase(name, sub[name]).to("cuda")
    m = sub[name]
    rank = base.rank

    def err():
        return H.func_loss_over(blk, cal_in, cal_out, kwargs)

    o = LO.build_all(st, n_seq=a.n_seq)[name].to("cuda")
    ps = GV.PlaneSearcher(o, base)
    E0 = err()
    print(f"[cmp] start block error {E0:.8f}", flush=True)

    rec = {"layer": name, "E0": E0, "K": a.K, "pool": a.pool,
           "accepts": [], "iters": []}
    E_cur = E0
    g = torch.Generator(device="cpu"); g.manual_seed(a.seed)
    t0 = time.time()
    n_eval = 0

    for it in range(a.max_iters):
        # refresh the block-residual gradient at the CURRENT state
        cu = CV.collect(st, layers=[name], n_seq=a.n_seq)[name].to("cuda")
        so = base.so0.double(); sp = base.sp0.double()
        with C.no_tf32():
            Gt = so.unsqueeze(1) * cu.G_W * sp.unsqueeze(0)

        # rank candidates across a random pool of planes by S3
        cands = []
        for _ in range(a.pool):
            i = int(torch.randint(0, rank, (1,), generator=g))
            j = int(torch.randint(0, rank, (1,), generator=g))
            if i == j:
                continue
            i, j = min(i, j), max(i, j)
            mids, s3 = s3_candidates(ps, i, j, Gt)
            k = int(torch.argmin(s3))
            if float(s3[k]) < 0:
                th = float(mids[k])
                if th >= math.pi / 4:
                    th -= GV.HALF_PI
                cands.append((float(s3[k]), i, j, th))
        cands.sort()

        best = None
        R_save = ps.R.clone(); U_save = ps.U.clone(); V_save = ps.Vm.clone()
        for s3v, i, j, th in cands[:a.K]:
            ps.apply(i, j, th)
            H.apply_gauge_signs_only(m, base, [ps.R.float()])
            e = err(); n_eval += 1
            if best is None or e < best[0]:
                best = (e, i, j, th, s3v)
            with torch.no_grad():
                ps.R, ps.U, ps.Vm = R_save.clone(), U_save.clone(), V_save.clone()
            ps.refresh()
        H.apply_gauge_signs_only(m, base, [ps.R.float()])

        if best is None or best[0] >= E_cur:
            rec["iters"].append({"iter": it, "accepted": False, "n_cands": len(cands),
                                 "best_seen": None if best is None else best[0],
                                 "E_cur": E_cur, "n_eval": n_eval})
            print(f"[cmp] iter {it:3d}: no improvement among top-{a.K} "
                  f"({len(cands)} cands)  E {E_cur:.8f}", flush=True)
            continue

        e, i, j, th, s3v = best
        ps.apply(i, j, th)
        H.apply_gauge_signs_only(m, base, [ps.R.float()])
        gain = e - E_cur
        E_cur = e
        with torch.no_grad(), C.no_tf32():
            dU = float((C.pos_sign(base.U0.double() @ ps.R) != C.pos_sign(base.U0.double())).sum())
            dV = float((C.pos_sign(ps.R.T @ base.V0.double()) != C.pos_sign(base.V0.double())).sum())
        rec["accepts"].append({"k": len(rec["accepts"]) + 1, "iter": it,
                               "plane": [i, j], "theta": th, "S3": s3v,
                               "E_after": E_cur, "gain": gain,
                               "cum_gain": E_cur - E0,
                               "cum_gain_pct": 100 * (E_cur - E0) / E0,
                               "sign_changes_U": dU, "sign_changes_V": dV,
                               "n_true_evals": n_eval, "seconds": time.time() - t0})
        print(f"[cmp] accept {len(rec['accepts']):3d} (iter {it}) plane ({i},{j}) "
              f"gain {gain:+.3e}  E {E_cur:.8f}  cum {100*(E_cur-E0)/E0:+.5f}%  "
              f"evals {n_eval}  ({time.time()-t0:.0f}s)", flush=True)
        S.atomic_json(rec, f"{OUT}/compose_{name.replace('.','_')}.json")
        S.atomic_save({"R": ps.R.cpu()}, f"{H.GCACHE}/compose_R_{name.replace('.','_')}.pt")
        if len(rec["accepts"]) >= a.max_accepts:
            break

    rec["E_final"] = E_cur
    rec["total_gain_pct"] = 100 * (E_cur - E0) / E0
    rec["n_true_evals"] = n_eval
    rec["wall_s"] = time.time() - t0
    S.atomic_json(rec, f"{OUT}/compose_{name.replace('.','_')}.json")
    print(f"[cmp] DONE {len(rec['accepts'])} accepts, {n_eval} true evals, "
          f"E {E0:.8f} -> {E_cur:.8f} ({rec['total_gain_pct']:+.5f}%)", flush=True)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="self_attn.v_proj")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--pool", type=int, default=24)
    ap.add_argument("--max_accepts", type=int, default=30)
    ap.add_argument("--max_iters", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_seq", type=int, default=0)
    a = ap.parse_args()
    a.n_seq = a.n_seq or None
    main(a)
