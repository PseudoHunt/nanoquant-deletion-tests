# Handoff — Gauge-NQ, state at VM shutdown

Everything below is reproducible from this repo plus `SamsungLabs/NanoQuant @ a9e0a43`.
No result depends on anything left on the VM.

## Environment to rebuild

| | |
|---|---|
| GPU | any 24 GB+ (was 1x A30) |
| torch | 2.11.0+cu130 |
| transformers | **4.57.1** (the 4.51.3 pin in `pyproject.toml` is unsatisfiable — see `repro.md`) |
| datasets | 5.0.1 |
| not installed | `gemlite`, `optimum`, `zeus-ml` (none reachable on this path) |
| model | `unsloth/Llama-3.2-1B`, snapshot `9535bd9b1d1dea6acafbdc4813b728796aeb28da` |
| config | bits 1.0, gamma **0.2** (Appendix C, not the repo default 0.4), 128 wikitext2 calib seqs, seed 0 |

Repo must live at `/home/work/exp` (paths are absolute in `nqx/`), with
`/home/work/hf_cache` pointing at the HF cache. `env.sh` sets `HF_HOME`,
`PYTHONPATH=nqx`, and `MID`.

## Rebuild order (~25 min GPU)

```
python3 nqx/sbh.py cache --model_id "$MID" --calib_seed 0     # FP caches, blocks 0 and 8
python3 nqx/sbh.py run   --model_id "$MID" --tag base --block 0 --no_eval
        # gate: pre 0.1416, post 0.1233 (tolerance 1e-3) -- see repro_gate.md
python3 nqx/testD.py setup --seed 0                           # post-ADMM block-0 state
python3 -m nqx.gauge.mksnap                                   # Snapshots A/B + manifest
python3 -m nqx.gauge.stage0                                   # exactness gates, must all pass
python3 -m nqx.gauge.quad                                     # fp64 block-output oracle (down_proj)
```

**The ADMM is chaotic** (NOTES.md#2), so a fresh VM produces a *different*
post-ADMM state, not a replay. Expect `E_ADMM` within ~0.1% of 0.1437658 and
Arm 0a `E_final` within ~0.5% of 0.115388. All arms must branch from the same
freshly reproduced state; cross-VM comparisons of absolute numbers are invalid.

## Where things stand

**Continuous STE gauge: dead.** Stage 1 (`stage1_down.md`) — every arm fails the
gate; the STE gradient is anti-correlated with the true objective where signs move.

**Discrete gauge, `down_proj`: real but weak.** Stage 1D (`stage1d_givens.md`) —
97.2% of one-plane neighbourhoods contain a better binary model, but they are
near-perfectly redundant (gain ∝ m^0.51, cyclic-vs-greedy Jaccard 0.13), and
exhaustive search of one 32-block converges to −0.041%. Cross-block gains *are*
additive (~0.030%/block over 9 blocks, no trend).

**Layer-wise objectives are anti-correlated with block error.**
`curvature/SUMMARY.md` — under an exactly function-preserving intervention, all
six non-`down_proj` layers improve their reconstruction 0.7–19% and worsen the
block 0.26–2.10%. At single-move granularity, Spearman(layer surrogate, true ΔE)
= **−0.232**. This is the most reusable result in the directory.

**A working selection method, which then fails to generalise.**
S3 (block-residual linear term) ranks at ρ = +0.618 while being useless as a
classifier. Enumerate → rank by S3 → verify top-K against the true block error →
accept only real improvements: 5 accepted moves gave **−0.1377% on calibration
with no decay**, versus −0.041% from 378 unverified moves.

But: **−0.0059% on held-out (24x attenuation)** and `E_final` +0.076%
(p = 0.077, n = 5 vs 9). The accept rule selects on calibration block error at
effect sizes (~3e-05) below that estimator's own noise, so it selects noise.

## The one experiment to run first tomorrow

**Validation split inside the accept rule.** Split the 128 calibration sequences
into a search slice and a validation slice; accept a move only if it improves
*both*. The 32-sequence held-out set stays untouched for final reporting. This
directly tests whether any of the discrete gauge capacity is real or is entirely
selection noise, and costs nothing beyond the split.

Run it on `v_proj` with `nqx/gauge/compose.py` (add the split to the accept
rule). If gains survive on the validation slice, the method is alive and the next
question is efficiency (480 true evals for 5 accepts). If they vanish, discrete
gauge beyond `down_proj` is finished and the pivot is to Step 3 itself.

**Do not** spend time on GN reranking first. Better ranking against a noisy
objective overfits harder, not less.

## Traps recorded the hard way

See `NOTES.md` items 1–6. The two that cost the most time:

* **`E_final` has sd 0.075%** and a heavy tail; a single run against a single
  baseline draw produced a *sign error* in published numbers (item 6). Every
  `E_final` claim needs replicates on **both** sides.
* **fp32 is not enough** anywhere in the gauge algebra: `mean()` is not
  bit-reproducible across equal tensors at different addresses (item 4), and the
  Givens objective is a ~30x cancellation that leaves 1.2e-03 of fp32 noise —
  comparable to a median improving move. Everything is fp64 with TF32 off.
* The empirical Fisher is **5.6e8x** off the Gauss-Newton curvature and wrong by
  a layer-dependent factor spanning 3 decades, so no single λ can rescue it.

## Artifacts not in git (regenerable)

`*.pt` is gitignored. Mirrored to `/home/jl_fs/gauge_mirror/` if that volume
survives: `block0_postadmm.pt`, `snapshot_{A,B}.pt`, `quad_oracle_fp64.pt`,
`givens_R_*.pt`, `compose_R_*.pt`, `wholeblock_R*.pt`. All are reproducible from
the rebuild order above; only the ADMM state is expensive (~3 min).
