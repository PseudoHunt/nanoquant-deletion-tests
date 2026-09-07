"""Whole-block discrete gauge search: all seven projections, with a true
block-error gate on every layer.

    cheap discrete search within layer  ->  TRUE block-error gate  ->  keep/revert

The gate is what makes this different from optimising `J`.  Tests A, B and A+B in
this repository all improved layer-wise reconstruction and all finished with a
worse post-Step-3 block error, so a layer surrogate cannot be trusted on its own.
Here every layer's gauge is accepted only if the *measured* calibration block
error improves; otherwise that layer's rotation is rolled back and the search
moves on.  A layer can therefore never make the block worse, and the number of
reverts is itself the Test-A/B signal.

`mlp.down_proj` uses the exact block-output oracle (its output is affine in the
block output).  The other six use the layer surrogate from `layer_oracle.py`.
"""
import argparse
import json
import os
import time

import torch

from nanoquant.modules.hub import NanoQuantConfigDataclass  # noqa: F401  import-order guard

import instrument as I

from . import core as C
from . import givens as GV
from . import harness as H
from . import layer_oracle as LO
from . import quad as Q
from . import state as S

ORDER = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
         'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']


def run(args):
    st = H.load_post_admm(seed=args.seed)
    blk, kwargs = st["blk"], st["kwargs"]
    sub = H.nq_sub(blk)
    cal_in, cal_out = st["cal_in"].to("cuda"), st["cal_out"].to("cuda")
    C.GUARD.assert_not_heldout(cal_in, cal_out, where="wholeblock")

    def block_err_calib():
        return H.func_loss_over(blk, cal_in, cal_out, kwargs)

    bases = {n: C.LayerBase(n, sub[n]).to("cuda") for n in ORDER}
    oracles = LO.build_all(st, n_seq=args.n_seq)
    oracles['mlp.down_proj'] = Q.Oracle(
        torch.load(f"{H.GCACHE}/{args.down_oracle}", weights_only=False))

    E_start = block_err_calib()
    rec = {"mode": "wholeblock", "order": ORDER, "blocks_per_layer": args.nb,
           "sweeps": args.sweeps, "n_seq_gram": args.n_seq,
           "E_ADMM_calib": E_start,
           "E_ADMM_heldout": I.block_err(blk, st["ho_in"], st["ho_out"], kwargs),
           "layers": [], "Rs": {}}
    print(f"[wb] start: calib {E_start:.8f}  held-out {rec['E_ADMM_heldout']:.6f}",
          flush=True)

    Rs = {n: torch.eye(bases[n].rank, device="cuda", dtype=torch.float64) for n in ORDER}
    t0 = time.time()
    for sw in range(args.sweeps):
        for name in ORDER:
            base, o = bases[name], oracles[name]
            ps = GV.PlaneSearcher(o, base)
            with torch.no_grad(), C.no_tf32():
                ps.R = Rs[name].clone()
                ps.U = base.U0.double() @ ps.R
                ps.Vm = base.V0.transpose(0, 1).contiguous().double() @ ps.R
            ps.refresh()

            E_before = block_err_calib()
            surro_before = o.error(ps.deployed_W())
            nb = min(args.nb, base.rank // 32)
            state = {"accepted": 0, "visited": 0}
            acc = 0
            for gi in range(nb):
                gp = list(range(gi * 32, gi * 32 + 32))
                pl = [(gp[a], gp[b]) for a in range(32) for b in range(a + 1, 32)]
                for (i, j) in pl:
                    r = ps.sweep(i, j)
                    state["visited"] += 1
                    if GV._accept_ok(r, o, max(surro_before, 1e-12), args):
                        ps.apply(r["i"], r["j"], r["theta"])
                        acc += 1
            surro_after = o.error(ps.deployed_W())

            # materialise this layer and take the TRUE block-error decision
            d = H.materialize_gauge(sub[name], base, [ps.R.float()])
            E_after = block_err_calib()
            keep = E_after < E_before
            ent = {"sweep": sw + 1, "layer": name, "rank": base.rank,
                   "blocks_searched": nb, "planes_visited": state["visited"],
                   "accepted": acc,
                   "surrogate_before": surro_before, "surrogate_after": surro_after,
                   "surrogate_gain_pct": 100 * (surro_after - surro_before) / max(abs(surro_before), 1e-30),
                   "block_err_before": E_before, "block_err_after": E_after,
                   "block_gain_pct": 100 * (E_after - E_before) / E_before,
                   "kept": bool(keep),
                   "sign_delta_U": d["sign_delta_U"], "sign_delta_V": d["sign_delta_V"],
                   "seconds": time.time() - t0}
            if keep:
                Rs[name] = ps.R.clone()
            else:
                H.materialize_gauge(sub[name], base, [Rs[name].float()])
                ent["block_err_after_revert"] = block_err_calib()
            rec["layers"].append(ent)
            print(f"[wb] sw{sw+1} {name:<18} {acc:4d} acc  surrogate "
                  f"{ent['surrogate_gain_pct']:+.4f}%  block {ent['block_gain_pct']:+.5f}%  "
                  f"{'KEEP' if keep else 'REVERT'}  ({time.time()-t0:.0f}s)", flush=True)
            del ps
            torch.cuda.empty_cache()
            S.atomic_json(rec, f"{H.GCACHE}/wholeblock{args.tag}.json")
            S.atomic_save({n: Rs[n].cpu() for n in ORDER},
                          f"{H.GCACHE}/wholeblock_R{args.tag}.pt")

    rec["E_gauge_calib"] = block_err_calib()
    rec["E_gauge_heldout"] = I.block_err(blk, st["ho_in"], st["ho_out"], kwargs)
    rec["n_kept"] = sum(1 for e in rec["layers"] if e["kept"])
    rec["n_revert"] = sum(1 for e in rec["layers"] if not e["kept"])
    rec["wall_s"] = time.time() - t0
    S.atomic_json(rec, f"{H.GCACHE}/wholeblock{args.tag}.json")
    print(f"[wb] FINAL calib {E_start:.8f} -> {rec['E_gauge_calib']:.8f} "
          f"({100*(rec['E_gauge_calib']-E_start)/E_start:+.4f}%)   held-out "
          f"{rec['E_ADMM_heldout']:.6f} -> {rec['E_gauge_heldout']:.6f} "
          f"({100*(rec['E_gauge_heldout']-rec['E_ADMM_heldout'])/rec['E_ADMM_heldout']:+.4f}%)   "
          f"kept {rec['n_kept']} / reverted {rec['n_revert']}", flush=True)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", type=int, default=8, help="rank blocks searched per layer")
    ap.add_argument("--sweeps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_seq", type=int, default=0, help="0 = all calibration sequences")
    ap.add_argument("--down_oracle", default="quad_oracle_fp64.pt")
    ap.add_argument("--tag", default="")
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--noise_k", type=float, default=10.0)
    a = ap.parse_args()
    a.n_seq = a.n_seq or None
    os.makedirs(H.GCACHE, exist_ok=True)
    run(a)
