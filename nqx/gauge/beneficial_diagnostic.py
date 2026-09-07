"""Phase A, part 2: score endpoints whose true effect is BENEFICIAL.

Part 1 scored six harmful endpoints.  On an all-harmful set the criterion is
degenerate: S1 and S2 are positive semi-definite, so they score n/n by
construction, and S4 = S3 + lambda.S2 reaches n/n for any lambda above a finite
threshold simply by drowning S3.  A surrogate can only be shown to work if it can
also be wrong in the other direction.

Beneficial endpoints already on disk, at zero extra search cost:

  * `self_attn.k_proj` rank-block 6 -- the single KEEP out of 43 rank-block
    decisions in the per-rank-block gated run (true block -0.007%)
  * `mlp.down_proj` at gauge doses of 1, 4, 9 and 16 rank blocks
    (true block -0.026% to -0.414%)

`k_proj` appears on both sides of the set, which removes the layer confound for
at least one layer.

`down_proj` needs no gradient pass: the block output is affine in its weight, so
E(W) = (T_sq - 2<G,W> + tr(W H W^T))/Y_sq exactly, giving

    G_W        = 2 (W H - G) / Y_sq                (exact gradient)
    H_in       = H / n_tokens
    g_n        = 2 r_n / Y_sq,  r = block output - FP output
    H_out      = mean_n g_n g_n^T                  (same empirical Fisher as elsewhere)

so its H_in comes straight from the existing quad oracle instead of a 15-minute
8192^2 re-accumulation.
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
from . import harness as H
from . import layer_oracle as LO
from . import quad as Q
from . import state as S

OUT = "/home/work/exp/results/nanoquant/gauge/curvature"
DOSES = [("down_1blk", "givens_R_cyclic.pt"), ("down_4blk", "givens_R_cyclic_b0123.pt"),
         ("down_9blk", "givens_R_cyclic_spread.pt"), ("down_16blk", "givens_R_cyclic_d16.pt")]


@torch.no_grad()
def down_curvature(st, oracle, blk, kwargs, cal_in, cal_out, sub, device="cuda"):
    """Exact G_W / H_in and the empirical-Fisher H_out for down_proj."""
    o = oracle.to(device)
    m = sub["mlp.down_proj"]
    W = I.effective_weight(m).double()
    with C.no_tf32():
        G_W = 2.0 * (W @ o.H - o.G) / o.Y_sq
    Ho = torch.zeros(m.out_features, m.out_features, device=device, dtype=torch.float64)
    ntok = 0
    for j in range(cal_in.shape[0]):
        y = blk(cal_in[j:j + 1], **kwargs)[0]
        r = (y.double() - cal_out[j:j + 1].double()).flatten(0, -2)
        g = 2.0 * r / o.Y_sq
        with C.no_tf32():
            Ho.add_(g.T @ g)
        ntok += r.shape[0]
    lc = CV.LayerCurvature("mlp.down_proj", G_W, o.H / ntok, Ho / ntok,
                           ntok, m.in_features, m.out_features)
    lc.n_hout = ntok
    lc.h_out_stride = 1
    return lc


def main(a):
    os.makedirs(OUT, exist_ok=True)
    st = H.load_post_admm(seed=a.seed)
    blk, kwargs = st["blk"], st["kwargs"]
    sub = H.nq_sub(blk)
    cal_in, cal_out = st["cal_in"].to("cuda"), st["cal_out"].to("cuda")
    C.GUARD.assert_not_heldout(cal_in, cal_out, where="beneficial_diagnostic")

    def block_err():
        return H.func_loss_over(blk, cal_in, cal_out, kwargs)

    E0 = block_err()
    rec = {"E_block_ADMM": E0, "endpoints": []}
    print(f"[ben] block error at original NanoQuant state = {E0:.8f}", flush=True)

    t0 = time.time()
    curv = CV.collect(st, layers=["self_attn.k_proj"], n_seq=a.n_seq)
    oracle = Q.Oracle(torch.load(f"{H.GCACHE}/quad_oracle_fp64.pt", weights_only=False))
    curv["mlp.down_proj"] = down_curvature(st, oracle, blk, kwargs, cal_in, cal_out, sub)
    print(f"[ben] curvature ready in {time.time()-t0:.0f}s", flush=True)

    oracles = LO.build_all(st, n_seq=a.n_seq)
    oracles["mlp.down_proj"] = oracle

    jobs = [("k_proj_rb6_KEEP", "self_attn.k_proj",
             f"{H.GCACHE}/wholeblock_R_blkgate.pt", "self_attn.k_proj")]
    for tag, f in DOSES:
        jobs.append((tag, "mlp.down_proj", f"{H.GCACHE}/{f}", None))

    for tag, name, path, key in jobs:
        t1 = time.time()
        base = C.LayerBase(name, sub[name]).to("cuda")
        blob = torch.load(path, weights_only=False)
        R = (blob[key] if key else blob["R"]).to("cuda").double()
        cu = curv[name].to("cuda")
        o = oracles[name].to("cuda")

        W_cur = I.effective_weight(sub[name]).double()
        W_fp = st["W_refs"][name].to("cuda").double()
        recon_before = o.error(W_cur.float())
        with torch.no_grad(), C.no_tf32():
            P0 = base.U0.double() @ base.V0.double()
            PR = (base.U0.double() @ R) @ (R.T @ base.V0.double())
            inv = float((PR - P0).norm() / max(float(P0.norm()), 1e-30))

        H.apply_gauge_signs_only(sub[name], base, [R.float()])
        W_cand = I.effective_weight(sub[name]).double()
        E_after = block_err()
        recon_after = o.error(W_cand.float())
        dW = W_cand - W_cur
        true_d = E_after - E0

        H_in_sum = cu.H_in * cu.n_tokens
        s0 = BS.s0_layer_recon_delta(W_cur, W_cand, W_fp, H_in_sum)
        s1 = BS.s1_input_quadratic(dW, cu.H_in)
        s2 = BS.s2_kfac_quadratic(dW, cu.H_in, cu.H_out, cu.n_tokens)
        s3 = BS.s3_block_linear(dW, cu.G_W)
        ent = {"tag": tag, "layer": name, "beneficial": bool(true_d < 0),
               "recon_before": recon_before, "recon_after": recon_after,
               "recon_delta_pct": 100 * (recon_after - recon_before) / abs(recon_before),
               "E_block_after": E_after, "true_delta": true_d,
               "true_delta_pct": 100 * true_d / E0,
               "S0": s0, "S1": s1, "S2": s2, "S3": s3,
               "continuous_product_rel_change": inv,
               "correct_S0": bool((s0 > 0) == (true_d > 0)),
               "correct_S1": bool((s1 > 0) == (true_d > 0)),
               "correct_S2": bool((s2 > 0) == (true_d > 0)),
               "correct_S3": bool((s3 > 0) == (true_d > 0)),
               "seconds": time.time() - t1}
        rec["endpoints"].append(ent)
        print(f"[ben] {tag:<18} true {ent['true_delta_pct']:+.4f}%  recon "
              f"{ent['recon_delta_pct']:+.3f}%  S0 {s0:+.3e} S3 {s3:+.3e} S2 {s2:+.3e}  "
              f"S3 {'OK' if ent['correct_S3'] else 'WRONG'}  inv {inv:.1e}", flush=True)

        H.materialize_gauge(sub[name], base, C.identity_Rs(base.rank, 32, "cuda"))
        back = block_err()
        assert abs(back - E0) < 1e-9, f"{tag}: rollback failed ({back:.10f} vs {E0:.10f})"
        S.atomic_json(rec, f"{OUT}/beneficial_diagnostic.json")
        torch.cuda.empty_cache()

    print(f"[ben] done in {time.time()-t0:.0f}s")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_seq", type=int, default=0)
    a = ap.parse_args()
    a.n_seq = a.n_seq or None
    main(a)
