"""Whole-block discrete gauge search with a true block-error gate.

    cheap surrogate PROPOSES moves  ->  TRUE block loss DECIDES which survive

The first version gated once per layer, after ~1800 accepted moves.  All six
surrogate layers reverted, but that verdict is coarse: a subset of those moves
could improve the block while the bundle as a whole does not.  The gate now
operates per **rank block** (~30-60 moves each), so a layer can keep the rank
blocks that help and roll back only the ones that hurt.  The surrogate is
demoted from objective to proposal mechanism.

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


def _materialize(module, base, R, how):
    if how == "fixed":
        H.apply_gauge_signs_only(module, base, [R])
        with torch.no_grad():
            U_R = base.U0 @ R
            V_R = R.transpose(0, 1) @ base.V0
            return {"sign_delta_U": float((C.pos_sign(U_R) != C.pos_sign(base.U0))
                                          .float().mean()),
                    "sign_delta_V": float((C.pos_sign(V_R) != C.pos_sign(base.V0))
                                          .float().mean())}
    return H.materialize_gauge(module, base, [R])


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

            E_layer_start = block_err_calib()
            surro_layer_start = o.error(ps.deployed_W())
            nb = min(args.nb, base.rank // 32)
            state = {"accepted": 0, "visited": 0}
            acc, E_cur = 0, E_layer_start
            blocks_kept, per_block = 0, []

            for gi in range(nb):
                # snapshot so a rejected rank block rolls back only itself
                R_s, U_s, V_s = ps.R.clone(), ps.U.clone(), ps.Vm.clone()
                sb = o.error(ps.deployed_W())
                gp = list(range(gi * 32, gi * 32 + 32))
                acc_g = 0
                for a_ in range(32):
                    for b_ in range(a_ + 1, 32):
                        r = ps.sweep(gp[a_], gp[b_])
                        state["visited"] += 1
                        if GV._accept_ok(r, o, max(surro_layer_start, 1e-12), args):
                            ps.apply(r["i"], r["j"], r["theta"])
                            acc_g += 1
                sa = o.error(ps.deployed_W())
                dd = _materialize(sub[name], base, ps.R.float(), args.export)
                E_g = block_err_calib()
                keep_g = E_g < E_cur
                pb = {"rank_block": gi, "accepted": acc_g,
                      "surrogate_before": sb, "surrogate_after": sa,
                      "surrogate_gain_pct": 100 * (sa - sb) / max(abs(sb), 1e-30),
                      "block_err_before": E_cur, "block_err_after": E_g,
                      "block_gain_pct": 100 * (E_g - E_cur) / E_cur,
                      "sign_delta_U": dd["sign_delta_U"],
                      "sign_delta_V": dd["sign_delta_V"], "kept": bool(keep_g)}
                if keep_g:
                    E_cur = E_g; acc += acc_g; blocks_kept += 1
                else:
                    with torch.no_grad():
                        ps.R, ps.U, ps.Vm = R_s, U_s, V_s
                    ps.refresh()
                    _materialize(sub[name], base, ps.R.float(), args.export)
                pb["cum_accepted_in_layer"] = acc
                pb["block_err_after_decision"] = E_cur
                per_block.append(pb)
                print(f"[wb/blk] {name:<18} rb{gi} +{acc_g:4d} acc  surrogate "
                      f"{pb['surrogate_gain_pct']:+.4f}%  block {pb['block_gain_pct']:+.5f}%  "
                      f"{'KEEP' if keep_g else 'revert'}  ({time.time()-t0:.0f}s)", flush=True)

            surro_after = o.error(ps.deployed_W())
            E_after, keep = E_cur, blocks_kept > 0
            d = {"sign_delta_U": per_block[-1]["sign_delta_U"] if per_block else 0.0,
                 "sign_delta_V": per_block[-1]["sign_delta_V"] if per_block else 0.0}
            ent = {"sweep": sw + 1, "layer": name, "rank": base.rank,
                   "blocks_searched": nb, "planes_visited": state["visited"],
                   "accepted": acc, "rank_blocks_kept": blocks_kept,
                   "rank_blocks_tried": nb, "per_block": per_block,
                   "surrogate_before": surro_layer_start, "surrogate_after": surro_after,
                   "surrogate_gain_pct": 100 * (surro_after - surro_layer_start) / max(abs(surro_layer_start), 1e-30),
                   "block_err_before": E_layer_start, "block_err_after": E_after,
                   "block_gain_pct": 100 * (E_after - E_layer_start) / E_layer_start,
                   "kept": bool(keep), "sign_delta_U": d["sign_delta_U"],
                   "sign_delta_V": d["sign_delta_V"], "seconds": time.time() - t0}
            Rs[name] = ps.R.clone()
            print(f"[wb] sw{sw+1} {name:<18} {acc:4d} acc  {blocks_kept}/{nb} rank blocks "
                  f"kept  surrogate {ent['surrogate_gain_pct']:+.4f}%  block "
                  f"{ent['block_gain_pct']:+.5f}%  ({time.time()-t0:.0f}s)", flush=True)
            rec["layers"].append(ent)
            del ps
            torch.cuda.empty_cache()
            S.atomic_json(rec, f"{H.GCACHE}/wholeblock{args.tag}.json")
            S.atomic_save({n: Rs[n].cpu() for n in ORDER},
                          f"{H.GCACHE}/wholeblock_R{args.tag}.pt")

    rec["E_gauge_calib"] = block_err_calib()
    rec["E_gauge_heldout"] = I.block_err(blk, st["ho_in"], st["ho_out"], kwargs)
    # report the other export convention too
    other = "qnq" if args.export == "fixed" else "fixed"
    for n in ORDER:
        _materialize(sub[n], bases[n], Rs[n].float(), other)
    rec[f"E_gauge_heldout_{other}"] = I.block_err(blk, st["ho_in"], st["ho_out"], kwargs)
    rec[f"E_gauge_calib_{other}"] = block_err_calib()
    for n in ORDER:
        _materialize(sub[n], bases[n], Rs[n].float(), args.export)
    rec["export"] = args.export
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
    ap.add_argument("--export", default="fixed", choices=["fixed", "qnq"])
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--noise_k", type=float, default=10.0)
    a = ap.parse_args()
    a.n_seq = a.n_seq or None
    os.makedirs(H.GCACHE, exist_ok=True)
    run(a)
