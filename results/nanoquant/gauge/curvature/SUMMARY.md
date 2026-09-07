# Block-aware discrete gauge search — from a failed surrogate to a working method

This directory covers the arc after Stage 1D, in which whole-block gauge search
using layer-wise reconstruction failed on all six non-`down_proj` projections,
and the search for a block-aware criterion that fixed it.

## 1. The failure that started it

Whole-block gauge search, gated once per layer on the true block error
(`wholeblock.json`). Every layer improved its own reconstruction objective and
made the block worse; all six reverted.

| layer | layer surrogate | true block error | verdict |
|---|---|---|---|
| q_proj | **−9.90%** | +0.350% | REVERT |
| k_proj | **−19.36%** | +0.290% | REVERT |
| v_proj | **−7.10%** | +1.861% | REVERT |
| o_proj | **−11.38%** | +2.102% | REVERT |
| gate_proj | −0.72% | +0.262% | REVERT |
| up_proj | −0.75% | +0.262% | REVERT |

Under an *exactly function-preserving* intervention (`UVᵀ` unchanged to 8.5e-7,
identical rank/bpw/representation), driving layer reconstruction down drives block
error up. Refining the gate to per-rank-block bundles of ~250 moves gave **1 keep
in 43** (`wholeblock_blkgate.json`) — better, not enough.

## 2. Phase A: which cheap surrogate predicts the true block direction?

Scored on 4 known-harmful endpoints plus 5 known-beneficial ones
(`endpoint_diagnostic.json`, `beneficial_diagnostic.json`).

| surrogate | correct | note |
|---|---|---|
| S0 layer reconstruction | 5/9 | = the majority-class baseline |
| S1 input quadratic | — | PSD: trivially "correct" on harmful sets |
| S2 K-FAC empirical Fisher | — | PSD: same |
| S3 block-residual linear | 6/9 | one better than guessing |

**A criterion built only on harmful endpoints is degenerate.** S1 and S2 are
positive semi-definite, so they score n/n by construction, and
`S4 = S3 + λ·S2` reaches n/n for any λ above a threshold simply by drowning S3.
Adding beneficial endpoints made the test falsifiable, and both surrogates
collapsed to the base rate.

**Why S3 fails.** `down_proj` is exactly quadratic in its weight, so
`ΔE = S3 + (n_tok/Y_sq)·S1` holds identically — verified to 0.8–4%. The linear
term **overshoots by 4.3×** and curvature cancels 77% of it. Curvature is the
same order as the signal, not a correction.

**Why the proposed curvature could not supply it.** The empirical Fisher is
**5.6e8–7.7e8×** smaller than the exact Gauss-Newton term, and the required λ
varies from ~2e5 to ~6e8 across layers — three decades — so no single λ exists.
`H_out` built from gradient outer products is the wrong object in kind, not scale.

## 3. Directional Gauss-Newton probe

Instead of forming `JᵀJ`, walk the ray `W + t·ΔW` and fit `E(t) = E(0) + at + bt²`:
`a` is the compensation term, `b` the directional GN curvature, with **no fitted
λ** (`gn_probe_*.json`).

| layer | dose | a (linear) | b (GN curv) | a+b | true ΔE | GN | lin |
|---|---|---|---|---|---|---|---|
| v_proj | 8 blk | −1.575e-03 | +3.675e-03 | **+2.100e-03** | +2.312e-03 | OK (9%) | ✗ |
| o_proj | 8 blk | −5.108e-04 | +3.058e-03 | **+2.547e-03** | +2.650e-03 | OK (4%) | ✗ |
| k_proj | 8 blk | −1.034e-03 | +7.835e-04 | −2.510e-04 | +3.557e-04 | **✗** | ✗ |

**GN 11/13, linear 2/13.** But GN has a roughly constant absolute error floor —
median **2.4e-04**, range 1.0e-04 to 6.1e-04 — while true effects span 6e-06 to
2.6e-03. It resolves large moves (v/o, 4–9% error) and fails at or below its
floor (k_proj, and the one beneficial endpoint at 6e-06).

The single beneficial non-`down_proj` endpoint we had (`k_proj` rb6, the lone KEEP
of 43) is **not trustworthy**: its sign flips between two mathematically
equivalent evaluation paths (−8.7e-06 two-stage vs +6.0e-06 dense). It was below
representation noise all along.

## 4. The reframe: S3 is a bad classifier and a good ranker

120 single Givens moves in `v_proj`, each applied alone and measured against the
true block error (`topk_self_attn_v_proj.json`):

| | |
|---|---|
| genuinely beneficial | **22 / 120 (18.3%)** |
| best true ΔE | −3.95e-05 |
| **Spearman(S3, true ΔE)** | **+0.618** |
| Spearman(S1, true ΔE) | +0.323 |
| **Spearman(layer surrogate, true ΔE)** | **−0.232** |

Two results here matter independently of any method.

**The capacity exists.** 18% of single moves in `v_proj` improve the block. The
earlier 0/8 rank-block verdict was not absence of capacity — each bundle mixed
~18% good with ~82% bad and the net was harmful. Every gate granularity tried was
too coarse: 1800-move bundles 0/6, 250-move bundles 1/43.

**The layer objective is anti-correlated, at the level of individual moves.**
ρ = −0.232: it ranks harmful moves *above* beneficial ones. That is the
quantitative statement of the failure in section 1.

Exact decomposition of the true change: median compensation **−1.81e-05**
(helping) against median curvature **+3.34e-05** (hurting).

## 5. The method, and that it composes

    exact Givens enumeration -> cheap S3 ranking -> true block loss on top-K
                             -> accept only a real improvement

S3 is never the judge, so a mis-ranked candidate costs an evaluation and cannot
damage the solution. `G_W` is recomputed after every accept
(`compose_self_attn_v_proj.json`).

| # | plane | gain | E after | cumulative |
|---|---|---|---|---|
| 1 | (37, 281) | −3.33e-05 | 0.12507944 | −0.0266% |
| 2 | (172, 344) | −2.89e-05 | 0.12505052 | −0.0497% |
| 3 | (194, 345) | −1.38e-05 | 0.12503674 | −0.0607% |
| 4 | (206, 343) | −3.35e-05 | 0.12500329 | −0.0875% |
| 5 | (1, 162) | **−6.28e-05** | 0.12494047 | **−0.1377%** |

**0.12511274 → 0.12494047, −0.1377% from five accepted moves, with no decay** —
accept 5 is the largest of the five.

| approach | moves | pre-Step-3 result |
|---|---|---|
| layer-surrogate search, `down_proj` | 378 | −0.041% (∝ m^0.51) |
| layer-surrogate search, `v_proj` | ~1600 | **+1.86%** (harmful) |
| **S3-ranked + true-verified, `v_proj`** | **5** | **−0.138%** |

Five verified moves beat 378 unverified ones by 3.4×, in the layer where the
unverified search did its worst damage.

## 6. Step 3 on the five-move checkpoint — and the generalisation gap

The 5-move `v_proj` checkpoint was run through the identical common Step 3, five
replicates against the 9-run baseline distribution (`runs/arm6_v_rep*.json`).

    calibration block error : 0.12511274 -> 0.12494047   (-0.1377%)
    HELD-OUT block error    : 0.14376577 -> 0.14375736   (-0.0059%)   <- 24x smaller

| | E_final | |
|---|---|---|
| baseline (n = 9) | 0.11523577 +- 8.2e-05 | |
| gauge (n = 5) | 0.115373 / 0.115193 / 0.115381 / 0.115334 / 0.115334 | mean **0.11532299** |
| delta | **+0.0757%** | Welch t = +2.00, df = 9, **p = 0.077** |

**No significant change in `E_final`, trending slightly worse.**

**The generalisation gap is the finding that matters, and it undercuts the method
as built.** The accept rule is "does the true *calibration* block error improve?",
and the moves available are worth ~3e-05 each -- inside the calibration set's own
sampling noise as an estimator of the true objective. Taking the best of 8
candidates by calibration loss therefore reliably improves calibration loss and
transfers almost nothing: 24x attenuation onto the held-out split.

This retrospectively qualifies section 4 as well. The "22/120 genuinely
beneficial" single moves were beneficial **on calibration**; held-out was not
measured for them, and on this evidence most of that is probably selection noise.
That measurement should have been taken at the time.

Consequences:

* **This is a selection problem, not a surrogate problem.** A better ranker (GN)
  would make it *worse* -- more efficient selection against a noisy objective
  overfits harder.
* The dose curve at 10/20/40 accepts will most likely show calibration improving
  and held-out flat.
* The fix to test first is a **validation split inside the accept rule**: accept
  only moves that improve on a search slice *and* on a slice held out from the
  search (both drawn from calibration; the 32-sequence held-out set stays
  untouched for final reporting). That directly measures whether any of this
  capacity is real, and costs nothing beyond splitting the 128 sequences.

## 7. Status and what is not yet established

* discrete gauge capacity outside `down_proj` — **demonstrated**
* cheap ranking signal — **demonstrated** (ρ = +0.62)
* safe acceptance rule — **demonstrated** (true objective is always the judge)
* composition without interference — **demonstrated on calibration** (5 moves, no decay)
* **generalisation — FAILS as built.** The calibration gain attenuates 24x onto
  held-out data; the accept rule selects on an objective whose noise exceeds the
  effect size.
* **post-Step-3 benefit — not obtained.** `E_final` +0.076%, p = 0.077 against a
  9-run baseline: no significant change, trending worse.

Efficiency is the open engineering problem: **480 true evaluations for 5 accepts**,
and the accept rate falls through the run — ranking by most-negative S3 surfaces
large perturbations, whose curvature cost grows faster than their compensation
benefit. That is a ranking problem, not a validity one.

## Files

| file | contents |
|---|---|
| `wholeblock.json` | per-layer gated whole-block search, 0/6 kept |
| `wholeblock_blkgate.json`, `per_rank_block_gate_partial.json` | per-rank-block gate, 1/43 |
| `endpoint_diagnostic.json` | S0–S4 on 4 harmful endpoints |
| `beneficial_diagnostic.json` | S0–S4 on 5 beneficial endpoints |
| `gn_probe_*.json` | directional GN, v/o/k, 4 doses each |
| `topk_self_attn_v_proj.json` | 120 single moves: existence + ranking power |
| `compose_self_attn_v_proj.json` | sequential composition, 5 accepts |
