# Gauge-NQ Stage 1 — latent gauge on `mlp.down_proj`, block 0

Does an *exactly equivalent* latent basis, chosen either geometrically or by
downstream function error, improve NanoQuant's final ~1-bit solution at identical
rank, bpw, ADMM, final optimiser, data and deployed representation?

Everything below is one block (block 0) of **Llama-3.2-1B** at NanoQuant
**bits = 1.0** (measured 0.9857 bpw), gamma = **0.2** (Appendix C), 128 wikitext2
calibration sequences at seed 0, 32 held-out sequences from wikitext2
**validation**. `NanoQuant @ a9e0a43`, unmodified; every change is patched in at
run time from `nqx/gauge/`.

## What the gauge actually is here

`admm_nq.factorize_admm_nanoquant` returns **no `scale_mid`**, and
`NanoQuantLinear._setup_path` only creates one when the factor results carry it.
The reproduced post-ADMM state therefore has **no rank-wise middle scale (no
s3)**, so brief section 2.1 applies unchanged and section 2.2's
fold -> rotate -> unfold is not instantiated. Nothing manufactures an s3.

    stored U_latent : [out,  rank]        math U : [out, rank]  ==  stored U_latent
    stored V_latent : [rank, in ]         math V : [in,  rank]  ==  stored V_latent.T

    deployed:  W_eff = diag(scale_post) . sign(U_latent) . sign(V_latent) . diag(scale_pre)

    gauge:     U_R = U R,  V_R = V R   =>   U_R V_R^T = U R R^T V^T = U V^T

Block-orthogonal R along the rank axis, block size b = 32 (rank 1600 -> 50
blocks, 992 -> 31, 384 -> 12).

## Q_NQ, the single deterministic export operator

NanoQuant's own export rule is *mean magnitude over the rank axis*
(`scale_pre = B_final.abs().mean(dim=0)`, `scale_post = A_final.abs().mean(dim=1)`).
Those two vectors come from `A_final`/`B_final` — the rank-1-projected `A_z`/`B_z`
— and are not recoverable from the stored latents alone, so Q_NQ applies
NanoQuant's rule to the rotated latents *relative to the base latents*:

    scale_post(R) = scale_post_0 * rowmean|U R| / rowmean|U|
    scale_pre (R) = scale_pre_0  * colmean|V_R| / colmean|V|

This is section 4.2's requirement (export statistics recomputed from the current
rotated factors, then detached) while satisfying the section 4.3 identity gate
exactly: at R = I the ratio is 1.0 and the cached ADMM export is reproduced
bit-for-bit.

**There is no SVID on the export path.** `admm_nq.svid` / `rank1_approx` are used
only *inside* the ADMM outer loop; the post-ADMM -> deployed conversion draws no
random numbers at all. Q_NQ is therefore already a deterministic function of
(U_R, V_R), proved by bit-comparing repeated calls rather than by RNG replay.
Snapshot A is still banked so the claim is auditable.

One numerical trap had to be fixed to make the identity gate hold: fp32
`mean(dim=...)` on CUDA is **not** bit-reproducible across two tensors holding
identical values at different addresses — the reduction splits differently and a
handful of coordinates come out ~1e-7 apart, which is enough to move the export
ratio off 1.0 by an ulp. The export statistic is accumulated in fp64, which puts
the layout-dependent discrepancy at ~1e-16 so the ratio rounds to exactly 1.0f.

## Stage 0 — every assertion passed

| check | result |
|---|---|
| no `scale_mid` in the reproduced state | PASS (section 2.1 path) |
| `\|U_R V_R - U V\|_F / \|U V\|_F` for random block-orthogonal R, all 7 layers | PASS, max **8.5e-7** |
| `R = I` gives bit-identical U and V | PASS, all 7 layers |
| Cayley orthogonality `max_i \|R_i^T R_i - I\|_F` | PASS, **3.7e-6** at random A, **0.0** at A = 0 |
| Cayley at A = 0 is exactly I | PASS, max deviation 0.0 |
| `Q_NQ(U, V)` reproduces the cached export bit-for-bit (signs *and* scales) | PASS, all 7 layers |
| Q_NQ deterministic (repeat calls bit-identical) | PASS |
| functional loss reaches the Cayley parameters, finite and non-zero | PASS |
| frozen base `U`, `V` have `grad is None` | PASS |
| no export scale/magnitude tensor is an optimiser parameter | PASS |
| one Adam step moves R, and R stays orthogonal | PASS |
| held-out guard rejects the held-out split, accepts calibration | PASS |

Machine-readable: `stage0.json`.

## Reference points

| | value |
|---|---|
| released interleaved schedule, block 0 `pre` / `post` (section 0.3 gate) | 0.1415907 / 0.1232319 |
| reproduced D2 post-ADMM held-out block error `E_ADMM` | **0.1437658** |
| previously cached D2 `pre` / joint-Step-3 `post` | 0.1436614 / 0.1148516 |
| **Arm 0a `E_final`** (this VM's baseline) | **0.1153880** |
| **Arm 0a duplicate `E_final`** (replay floor) | **0.1151653**, i.e. **0.19%** |
| identity-gauge full-calibration functional loss | 0.125113 |

The ADMM was re-derived from scratch on this VM. NOTES.md#2 records that the
ADMM is chaotic — a relative 1e-7 perturbation of the input weight diverges to a
relative 0.87 by iteration 160, and the default runs 400 — so the reproduced
post-ADMM state is an *independent draw* from the same procedure, not a replay.
`E_ADMM` lands +0.073% from the cached value and Arm 0a's `E_final` +0.47% from
the cached 0.1148516. The 0.19% same-state replay floor measured here is inside
the 0.31% floor the repo established, so **δ = 0.0031 is kept as the gate**, and
every arm below branches from the *same* reproduced state, so the comparison is
internally controlled.

## Results

| arm | E_ADMM | E_gauge pre-export | E_gauge | E_final | Δ vs 0a | Δ vs matched 0b | sign Δ U | sign Δ V | wall s |
|---|---|---|---|---|---|---|---|---|---|
| **0a  baseline (R = I)** | 0.143766 | 0.143766 | 0.143766 | **0.115388** | +0.00% | — | 0.00% | 0.00% | 30 |
| **0a-dup  determinism replay** | 0.143766 | 0.143766 | 0.143766 | **0.115165** | -0.19% | — | 0.00% | 0.00% | 30 |
| **0b_100  extra STE x100** | 0.143766 | — | 0.133687 | **0.116673** | +1.11% | — | 0.04% | 1.26% | 31 |
| **0b_200  extra STE x200** | 0.143766 | — | 0.132425 | **0.118267** | +2.50% | — | 0.26% | 2.19% | 33 |
| **2  ITQ, refreshed magnitudes, R0 = I** | 0.143766 | 0.143791 | 0.143791 | **0.115075** | -0.27% | -2.70% | 0.00% | 0.06% | 37 |
| **2  ITQ, frozen magnitudes, R0 = I** | 0.143766 | 0.143791 | 0.143791 | **0.115109** | -0.24% | -2.67% | 0.00% | 0.06% | 36 |
| **2  ITQ, refreshed magnitudes, R0 random** | 0.143766 | 0.298349 | 0.341970 | **0.141624** | +22.74% | +19.75% | 50.04% | 50.00% | 38 |
| **2  ITQ, frozen magnitudes, R0 random** | 0.143766 | 0.298610 | 0.341778 | **0.141755** | +22.85% | +19.86% | 50.04% | 50.00% | 37 |
| **3  functional gauge lr=3e-3 @100** | 0.143766 | 0.215149 | 0.230205 | **0.130715** | +13.28% | +12.04% | 7.62% | 11.16% | 29 |
| **3  functional gauge lr=3e-3 @200** | 0.143766 | 0.273771 | 0.289858 | **0.140818** | +22.04% | +19.07% | 21.44% | 23.74% | 29 |
| **3  functional gauge lr=1e-3 @100** | 0.143766 | 0.160283 | 0.169900 | **0.118820** | +2.97% | +1.84% | 1.04% | 3.60% | 30 |
| **3  functional gauge lr=1e-3 @200** | 0.143766 | 0.184321 | 0.196208 | **0.123620** | +7.13% | +4.53% | 3.39% | 6.80% | 29 |
| **3  functional gauge lr=3e-4 @100** | 0.143766 | 0.146353 | 0.150818 | **0.116059** | +0.58% | -0.53% | 0.12% | 1.30% | 29 |
| **3  functional gauge lr=3e-4 @200** | 0.143766 | 0.151494 | 0.158784 | **0.117031** | +1.42% | -1.05% | 0.38% | 2.31% | 29 |
| **3  functional gauge lr=1e-4 @100** | 0.143766 | 0.144401 | 0.147720 | **0.115612** | +0.19% | -0.91% | 0.02% | 0.55% | 29 |
| **3  functional gauge lr=1e-4 @200** | 0.143766 | 0.145112 | 0.148725 | **0.115596** | +0.18% | -2.26% | 0.06% | 0.91% | 29 |
| **3  functional gauge lr=3e-5 @100** | 0.143766 | 0.143812 | 0.144126 | **0.115293** | -0.08% | -1.18% | 0.00% | 0.12% | 29 |
| **3  functional gauge lr=3e-5 @200** | 0.143766 | 0.144033 | 0.145652 | **0.115172** | -0.19% | -2.62% | 0.00% | 0.31% | 29 |
| **3  functional gauge lr=1e-5 @100** | 0.143766 | 0.143775 | 0.143783 | **0.115286** | -0.09% | -1.19% | 0.00% | 0.03% | 29 |
| **3  functional gauge lr=1e-5 @200** | 0.143766 | 0.143784 | 0.143978 | **0.115323** | -0.06% | -2.49% | 0.00% | 0.07% | 29 |

## Stage-1 success gate

A gauge checkpoint passes iff `E_final(gauge) < E_final(0a)·(1−δ)` **and** `E_final(gauge) < E_final(0b_matched)·(1−δ)`, with δ = 0.0031.

| checkpoint | E_final | 0a threshold | matched 0b | 0b threshold | verdict |
|---|---|---|---|---|---|
| 2  ITQ, refreshed magnitudes, R0 = I | 0.115075 | 0.115030 | 0b_200 = 0.118267 | 0.117901 | **FAIL** |
| 2  ITQ, frozen magnitudes, R0 = I | 0.115109 | 0.115030 | 0b_200 = 0.118267 | 0.117901 | **FAIL** |
| 2  ITQ, refreshed magnitudes, R0 random | 0.141624 | 0.115030 | 0b_200 = 0.118267 | 0.117901 | **FAIL** |
| 2  ITQ, frozen magnitudes, R0 random | 0.141755 | 0.115030 | 0b_200 = 0.118267 | 0.117901 | **FAIL** |
| 3  functional gauge lr=3e-3 @100 | 0.130715 | 0.115030 | 0b_100 = 0.116673 | 0.116311 | **FAIL** |
| 3  functional gauge lr=3e-3 @200 | 0.140818 | 0.115030 | 0b_200 = 0.118267 | 0.117901 | **FAIL** |
| 3  functional gauge lr=1e-3 @100 | 0.118820 | 0.115030 | 0b_100 = 0.116673 | 0.116311 | **FAIL** |
| 3  functional gauge lr=1e-3 @200 | 0.123620 | 0.115030 | 0b_200 = 0.118267 | 0.117901 | **FAIL** |
| 3  functional gauge lr=3e-4 @100 | 0.116059 | 0.115030 | 0b_100 = 0.116673 | 0.116311 | **FAIL** |
| 3  functional gauge lr=3e-4 @200 | 0.117031 | 0.115030 | 0b_200 = 0.118267 | 0.117901 | **FAIL** |
| 3  functional gauge lr=1e-4 @100 | 0.115612 | 0.115030 | 0b_100 = 0.116673 | 0.116311 | **FAIL** |
| 3  functional gauge lr=1e-4 @200 | 0.115596 | 0.115030 | 0b_200 = 0.118267 | 0.117901 | **FAIL** |
| 3  functional gauge lr=3e-5 @100 | 0.115293 | 0.115030 | 0b_100 = 0.116673 | 0.116311 | **FAIL** |
| 3  functional gauge lr=3e-5 @200 | 0.115172 | 0.115030 | 0b_200 = 0.118267 | 0.117901 | **FAIL** |
| 3  functional gauge lr=1e-5 @100 | 0.115286 | 0.115030 | 0b_100 = 0.116673 | 0.116311 | **FAIL** |
| 3  functional gauge lr=1e-5 @200 | 0.115323 | 0.115030 | 0b_200 = 0.118267 | 0.117901 | **FAIL** |

## Every arm ranked by `E_final` — where the noise band ends

`0a_dup` is the baseline re-run from a bit-identical state, so its distance from `0a` is pure kernel non-determinism. Any cell that lands between them is indistinguishable from doing nothing — and every such cell turns out to be a gauge that did not move (`sign Δ U = 0.00%`). The first cell that moves a U sign falls below the baseline, and everything after it is worse, monotonically in how far the gauge travelled.

| arm | E_final | Δ vs 0a | E_gauge | sign Δ U | sign Δ V |
|---|---|---|---|---|---|
| `2_refreshed` | 0.115075 | -0.27% | 0.143791 | 0.00% | 0.06% |
| `2_frozen` | 0.115109 | -0.24% | 0.143791 | 0.00% | 0.06% |
| `0a_dup` **<- baseline re-run** | 0.115165 | -0.19% | 0.143766 | 0.00% | 0.00% |
| `3lr3e-5_200` | 0.115172 | -0.19% | 0.145652 | 0.00% | 0.31% |
| `3lr1e-5_100` | 0.115286 | -0.09% | 0.143783 | 0.00% | 0.03% |
| `3lr3e-5_100` | 0.115293 | -0.08% | 0.144126 | 0.00% | 0.12% |
| `3lr1e-5_200` | 0.115323 | -0.06% | 0.143978 | 0.00% | 0.07% |
| `0a` **<- baseline** | 0.115388 | +0.00% | 0.143766 | 0.00% | 0.00% |
| `3lr1e-4_200` | 0.115596 | +0.18% | 0.148725 | 0.06% | 0.91% |
| `3lr1e-4_100` | 0.115612 | +0.19% | 0.147720 | 0.02% | 0.55% |
| `3lr3e-4_100` | 0.116059 | +0.58% | 0.150818 | 0.12% | 1.30% |
| `0b_100` | 0.116673 | +1.11% | 0.133687 | 0.04% | 1.26% |
| `3lr3e-4_200` | 0.117031 | +1.42% | 0.158784 | 0.38% | 2.31% |
| `0b_200` | 0.118267 | +2.50% | 0.132425 | 0.26% | 2.19% |
| `3lr1e-3_100` | 0.118820 | +2.97% | 0.169900 | 1.04% | 3.60% |
| `3lr1e-3_200` | 0.123620 | +7.13% | 0.196208 | 3.39% | 6.80% |
| `3_100` | 0.130715 | +13.28% | 0.230205 | 7.62% | 11.16% |
| `3_200` | 0.140818 | +22.04% | 0.289858 | 21.44% | 23.74% |
| `2_refreshed_rand` | 0.141624 | +22.74% | 0.341970 | 50.04% | 50.00% |
| `2_frozen_rand` | 0.141755 | +22.85% | 0.341778 | 50.04% | 50.00% |

## Arm 1 — random gauge sensitivity (16 Haar block-rotations, no Step 3)

Seed 777, block size 32, 16 samples on `mlp.down_proj`.

| | E_gauge |
|---|---|
| identity (R = I) | 0.143766 |
| min | 0.358381 |
| median | 0.361482 |
| max | 0.367272 |
| std | 2.909e-03 |
| best rotation id | 9 |

| id | E_gauge | E_gauge pre-export | sign Δ U | sign Δ V |
|---|---|---|---|---|
| 0 | 0.358888 | 0.312992 | 50.12% | 50.12% |
| 1 | 0.359932 | 0.312583 | 50.15% | 50.12% |
| 2 | 0.365355 | 0.315214 | 49.50% | 49.57% |
| 3 | 0.359221 | 0.311696 | 50.25% | 50.28% |
| 4 | 0.361482 | 0.313376 | 50.22% | 50.19% |
| 5 | 0.360613 | 0.312979 | 49.78% | 49.79% |
| 6 | 0.360104 | 0.311645 | 49.96% | 49.98% |
| 7 | 0.364138 | 0.315975 | 49.90% | 49.91% |
| 8 | 0.363129 | 0.314074 | 50.28% | 50.29% |
| 9 | 0.358381 | 0.312745 | 50.44% | 50.42% |
| 10 | 0.362131 | 0.315274 | 49.80% | 49.85% |
| 11 | 0.366679 | 0.315529 | 49.75% | 49.81% |
| 12 | 0.361371 | 0.313846 | 50.07% | 50.09% |
| 13 | 0.364149 | 0.318017 | 50.13% | 50.10% |
| 14 | 0.366201 | 0.316197 | 50.01% | 50.06% |
| 15 | 0.367272 | 0.317205 | 50.10% | 50.10% |

## Arm 2 — 2  ITQ, refreshed magnitudes, R0 = I, 20-round trace

| round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E_gauge | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 |

## Arm 2 — 2  ITQ, frozen magnitudes, R0 = I, 20-round trace

| round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E_gauge | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 | 0.14379 |

## Arm 2 — 2  ITQ, refreshed magnitudes, R0 random, 20-round trace

| round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E_gauge | 0.36138 | 0.36157 | 0.36063 | 0.36036 | 0.36004 | 0.35871 | 0.35848 | 0.35726 | 0.35641 | 0.35622 | 0.35500 | 0.35408 | 0.35425 | 0.35248 | 0.35172 | 0.35010 | 0.34872 | 0.34727 | 0.34460 | 0.34197 |

## Arm 2 — 2  ITQ, frozen magnitudes, R0 random, 20-round trace

| round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E_gauge | 0.36126 | 0.36119 | 0.36033 | 0.36043 | 0.35952 | 0.35865 | 0.35850 | 0.35710 | 0.35648 | 0.35595 | 0.35484 | 0.35360 | 0.35283 | 0.35125 | 0.35056 | 0.34968 | 0.34820 | 0.34692 | 0.34427 | 0.34178 |

## Arm 3 — functional latent gauge: calibration objective vs gauge step size

Adam on the block-Cayley parameters, 200 steps, calibration split only. `L_func` here is the **full-calibration** functional loss (all 128 sequences), probed every 25 steps. The identity gauge sits at **0.125113**; a cell that never goes below that never descended.

| gauge lr | step 25 | step 50 | step 75 | step 100 | step 125 | step 150 | step 175 | step 200 |
|---|---|---|---|---|---|---|---|---|
| 0.003 | — | — | — | — | — | — | — | — |
| 0.001 | 0.13168 | 0.13629 | 0.14224 | 0.15153 | 0.16390 | 0.16688 | 0.17274 | 0.17789 |
| 0.0003 | 0.12979 | 0.12970 | 0.13067 | 0.13240 | 0.13311 | 0.13458 | 0.13691 | 0.14041 |
| 0.0001 | 0.12536 | 0.12650 | 0.12864 | 0.12928 | 0.12921 | 0.12919 | 0.12985 | 0.13024 |
| 3e-05 | 0.12510 | 0.12523 | 0.12531 | 0.12548 | 0.12565 | 0.12603 | 0.12651 | 0.12708 |
| 1e-05 | 0.12511 | 0.12511 | 0.12510 | 0.12511 | 0.12512 | 0.12513 | 0.12523 | 0.12531 |

Section 4.2 logging — calibration functional loss at each checkpoint, before and after the export statistics are re-extracted:

| gauge lr | checkpoint | L_func pre-export stats | L_func post-export stats |
|---|---|---|---|
| 0.003 | 100 | 0.196140 | 0.211763 |
| 0.003 | 200 | 0.254537 | 0.272164 |
| 0.001 | 100 | 0.141552 | 0.151535 |
| 0.001 | 200 | 0.165563 | 0.177893 |
| 0.0003 | 100 | 0.127728 | 0.132401 |
| 0.0003 | 200 | 0.132865 | 0.140412 |
| 0.0001 | 100 | 0.125803 | 0.129280 |
| 0.0001 | 200 | 0.126542 | 0.130244 |
| 3e-05 | 100 | 0.125171 | 0.125482 |
| 3e-05 | 200 | 0.125442 | 0.127082 |
| 1e-05 | 100 | 0.125124 | 0.125105 |
| 1e-05 | 200 | 0.125134 | 0.125312 |

## Line search at `R = I` — is the ADMM basis a local optimum?

Functional gradient accumulated over **all 128** calibration sequences at `R = I` (so this is not mini-batch noise), `||g||_F = 4.186e-05`, normalised to a unit direction. `t` is the total Frobenius displacement of the Cayley parameters — on the same scale the sweep uses (Adam at lr 1e-5 for 200 steps travels about 0.45; at lr 3e-3, about 135). Calibration data only.

| t | 0 | 0.001 | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 | 10 | 30 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **−grad**, post-export | 0.125113 | 0.125111 | 0.125111 | 0.125110 | 0.125099 | 0.125331 | 0.131299 | 0.175430 | 0.294384 | 0.287309 | 0.294374 |
| **−grad**, pre-export | 0.125113 | 0.125111 | 0.125111 | 0.125110 | 0.125118 | 0.125161 | 0.126109 | 0.144304 | 0.226865 | 0.276912 | 0.286812 |
| random dir 0, post-export | 0.125113 | 0.125110 | 0.125110 | 0.125109 | 0.125111 | 0.125123 | 0.125501 | 0.132576 | 0.271278 | 0.348763 | 0.341674 |
| random dir 1, post-export | 0.125113 | 0.125112 | 0.125112 | 0.125112 | 0.125116 | 0.125132 | 0.125604 | 0.132935 | 0.274568 | 0.348076 | 0.339573 |
| random dir 2, post-export | 0.125113 | 0.125113 | 0.125113 | 0.125114 | 0.125110 | 0.125130 | 0.125514 | 0.132579 | 0.272783 | 0.353191 | 0.341620 |

Best point anywhere on the steepest-descent ray: **0.125099** against the identity's **0.125113** — an improvement of **0.011%**, twenty times below the 0.19% replay floor. Across the whole range where the curve is actually moving (t = 0.3 to t = 10) the gradient direction rises **faster than a random one** — at t = 1, 0.1313 against 0.1255/0.1256/0.1255; at t = 3, 0.1754 against 0.1326/0.1329/0.1326 — so the STE gradient through the hard sign is not merely uninformative about the gauge, it is anti-correlated with the true objective at any step size large enough to move anything. (By t = 30 every direction has saturated near the random-gauge error and the ordering stops meaning anything.)

## Per-layer post-Step-3 diagnostic (cumulative curve)

Layers 1..k at their tuned signs, the rest back at their pre-Step-3 signs. Mechanism analysis only — the joint Step 3 may redistribute compensation across layers, so the final gain is not required to stay localised to `down_proj`.

Columns are a readable subset; every arm is in `runs/*.json`.

| layer | `0a` | `0a_dup` | `0b_100` | `0b_200` | `2_refreshed` | `2_refreshed_rand` | `3_100` | `3_200` | `3lr3e-4_200` | `3lr3e-5_200` | `3lr1e-5_200` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `self_attn.q_proj` | 0.14256 | 0.14303 | 0.13814 | 0.13721 | 0.14283 | 0.30241 | 0.21495 | 0.26712 | 0.15270 | 0.14369 | 0.14278 |
| `self_attn.v_proj` | 0.14403 | 0.14443 | 0.13887 | 0.13713 | 0.14431 | 0.29811 | 0.21502 | 0.26596 | 0.15375 | 0.14490 | 0.14407 |
| `self_attn.o_proj` | 0.14899 | 0.14956 | 0.14285 | 0.13888 | 0.14965 | 0.28765 | 0.21511 | 0.26370 | 0.15761 | 0.14961 | 0.14866 |
| `self_attn.k_proj` | 0.14892 | 0.14951 | 0.14258 | 0.13872 | 0.14957 | 0.28709 | 0.21501 | 0.26357 | 0.15747 | 0.14948 | 0.14854 |
| `mlp.gate_proj` | 0.13906 | 0.13931 | 0.13607 | 0.13464 | 0.13920 | 0.26967 | 0.20455 | 0.24995 | 0.14797 | 0.13916 | 0.13869 |
| `mlp.up_proj` | 0.12780 | 0.12761 | 0.12908 | 0.13050 | 0.12754 | 0.25060 | 0.19323 | 0.23521 | 0.13740 | 0.12789 | 0.12754 |
| `mlp.down_proj` | 0.11539 | 0.11517 | 0.11667 | 0.11827 | 0.11507 | 0.14162 | 0.13072 | 0.14082 | 0.11703 | 0.11517 | 0.11532 |

## J diagnostic (H-weighted layer reconstruction objective), pre / post Step 3

| layer | `0a` pre / post | `0a_dup` pre / post | `0b_100` pre / post | `0b_200` pre / post | `2_refreshed` pre / post | `2_refreshed_rand` pre / post | `3_100` pre / post | `3_200` pre / post | `3lr3e-4_200` pre / post | `3lr3e-5_200` pre / post | `3lr1e-5_200` pre / post |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `self_attn.q_proj` | 0.0079 / 0.0098 | 0.0079 / 0.0099 | 0.0079 / 0.0098 | 0.0079 / 0.0099 | 0.0079 / 0.0099 | 0.0079 / 0.0106 | 0.0079 / 0.0103 | 0.0079 / 0.0104 | 0.0079 / 0.0099 | 0.0079 / 0.0099 | 0.0079 / 0.0098 |
| `self_attn.v_proj` | 0.2630 / 0.2550 | 0.2630 / 0.2552 | 0.2630 / 0.2555 | 0.2630 / 0.2558 | 0.2630 / 0.2554 | 0.2630 / 0.2551 | 0.2630 / 0.2548 | 0.2630 / 0.2552 | 0.2630 / 0.2549 | 0.2630 / 0.2552 | 0.2630 / 0.2551 |
| `self_attn.o_proj` | 0.0821 / 0.0865 | 0.0821 / 0.0867 | 0.0821 / 0.0853 | 0.0821 / 0.0844 | 0.0821 / 0.0867 | 0.0821 / 0.0895 | 0.0821 / 0.0872 | 0.0821 / 0.0884 | 0.0821 / 0.0868 | 0.0821 / 0.0869 | 0.0821 / 0.0867 |
| `self_attn.k_proj` | 0.0190 / 0.0189 | 0.0190 / 0.0189 | 0.0190 / 0.0189 | 0.0190 / 0.0190 | 0.0190 / 0.0189 | 0.0190 / 0.0193 | 0.0190 / 0.0191 | 0.0190 / 0.0192 | 0.0190 / 0.0189 | 0.0190 / 0.0189 | 0.0190 / 0.0190 |
| `mlp.gate_proj` | 0.1917 / 0.2201 | 0.1917 / 0.2214 | 0.1917 / 0.2099 | 0.1917 / 0.2042 | 0.1917 / 0.2218 | 0.1917 / 0.2287 | 0.1917 / 0.2152 | 0.1917 / 0.2220 | 0.1917 / 0.2175 | 0.1917 / 0.2215 | 0.1917 / 0.2201 |
| `mlp.up_proj` | 0.2501 / 0.2748 | 0.2501 / 0.2758 | 0.2501 / 0.2724 | 0.2501 / 0.2709 | 0.2501 / 0.2760 | 0.2501 / 0.2890 | 0.2501 / 0.2712 | 0.2501 / 0.2800 | 0.2501 / 0.2718 | 0.2501 / 0.2759 | 0.2501 / 0.2747 |
| `mlp.down_proj` | 0.1665 / 0.1700 | 0.1665 / 0.1704 | 0.1595 / 0.1914 | 0.1649 / 0.2077 | 0.1666 / 0.1696 | 0.4755 / 0.2955 | 0.3305 / 0.2257 | 0.4408 / 0.2769 | 0.1888 / 0.1743 | 0.1676 / 0.1697 | 0.1666 / 0.1700 |

## Step-3 sign-flip fraction on `mlp.down_proj`

| arm | U flips | V flips | block mean |
|---|---|---|---|
| `0a` | 0.98% | 2.55% | 1.01% |
| `0a_dup` | 0.98% | 2.58% | 1.03% |
| `0b_100` | 1.29% | 3.05% | 0.98% |
| `0b_200` | 1.59% | 3.42% | 0.91% |
| `2_refreshed` | 0.99% | 2.55% | 1.03% |
| `2_frozen` | 0.99% | 2.57% | 1.03% |
| `2_refreshed_rand` | 8.09% | 7.22% | 1.84% |
| `2_frozen_rand` | 8.11% | 7.15% | 1.83% |
| `3_100` | 5.38% | 5.64% | 1.46% |
| `3_200` | 7.88% | 6.94% | 1.76% |
| `3lr1e-3_100` | 1.99% | 3.81% | 1.13% |
| `3lr1e-3_200` | 3.51% | 4.75% | 1.27% |
| `3lr3e-4_100` | 1.22% | 3.04% | 1.07% |
| `3lr3e-4_200` | 1.50% | 3.40% | 1.09% |
| `3lr1e-4_100` | 1.05% | 2.81% | 1.05% |
| `3lr1e-4_200` | 1.13% | 2.92% | 1.07% |
| `3lr3e-5_100` | 0.99% | 2.66% | 1.03% |
| `3lr3e-5_200` | 1.01% | 2.74% | 1.05% |
| `3lr1e-5_100` | 0.99% | 2.56% | 1.02% |
| `3lr1e-5_200` | 0.99% | 2.58% | 1.02% |

## What the arms say

### 1. Binary error varies enormously across the latent gauge class

Sixteen Haar SO(32) block-rotations of `down_proj`'s rank space leave the
continuous product `U V^T` unchanged to a relative **8.5e-7** and move the
held-out block error from **0.14377** to **0.358 – 0.367**, a factor of **2.5**,
flipping **~50%** of both U and V signs. So the gauge is emphatically not a
nuisance parameter: `sign(U R) sign(V R)^T` is wildly sensitive to R even though
`U R (V R)^T` is not.

The direction is uniform, though. **Not one of the sixteen beats the identity**,
and the spread across them (std 2.9e-3, a 2.4% band) is tiny next to their common
2.5x distance from it. The ADMM latent basis is not an arbitrary point in its own
gauge orbit; it is a sharply distinguished one. A generic gauge destroys the
alignment between the two sign patterns that makes the binary product work at all.

### 2. Geometric alignment has nothing to align

Both Joint-ITQ variants started at `R = I` move `E_gauge` from 0.1437658 to
0.1437914 in round 1 and then sit there for all twenty rounds (**+0.018%**, three
orders of magnitude below the effect Arm 1 shows exists). `R = I` is a fixed
point of the Procrustes step, and structurally so: with no `scale_mid`, the
magnitude field NanoQuant's export produces is **rank-1** — `M_U` is one scalar
per output row, broadcast across the rank axis — so the target
`M ⊙ sign(X0 R)` agrees with `X0 R` in sign *by construction*, and
`tr(R^T X0^T T)` is already stationary at whatever R built it.

Started from a random rotation instead — which is how ITQ is classically
initialised — the alignment does exactly what it is supposed to do. It descends
steadily and monotonically, 0.3614 -> 0.3420 over twenty rounds, and it is still
**2.4x above the identity** when it gets there, with 50.0% of U signs and 50.0%
of V signs different from the ADMM ones. `E_final` 0.14159, **+22.7%** against
Arm 0a. So the geometric criterion works as an algorithm and converges to a basin
that has nothing to do with the one NanoQuant's ADMM already found. That is Arm 1's
message from the other side: the ADMM basis is not somewhere ITQ can get to.

The two `R0 = I` ITQ arms land at `E_final` 0.115075 / 0.115109 against Arm 0a's
0.115388 and the 0a-duplicate's 0.115165. That is not a result — a gauge that
does not move *should* reproduce the baseline, and the fact that it does, to
within the 0.19% replay floor, is a consistency check on the whole materialise /
export / Step-3 path.

### 3. The functional gauge never descends, at any step size

At the pre-registered `lr = 3e-3` the gauge search **diverges**: the
full-calibration functional loss climbs monotonically from the identity's
**0.125113** to 0.2118 at step 100 and 0.2722 at step 200, and `E_final` follows
it to 0.1307 / 0.1408 — 13% and 22% *worse* than Arm 0a.

Running a proposed method only at a step size where it provably ascends is not a
test of the method, so the gauge learning rate was swept over five values. The
sweep is selected on the **calibration** objective alone; the held-out split
takes no part in it. The result is unambiguous:

**every learning rate increases the calibration objective above the identity
value from the very first probe**, and smaller learning rates simply move less.
There is no step size at which the STE gradient buys a better gauge.

The line search settles what that means. The functional gradient was accumulated
over the **whole** calibration set at `R = I` (so this is not mini-batch noise),
normalised to a unit direction, and the *true* calibration loss evaluated along
the ray. `t` is the total Frobenius displacement of the Cayley parameters, on the
same scale the sweep uses: Adam at lr 1e-5 for 200 steps travels about 0.45, at
lr 3e-3 about 135.

| t | 0 | 0.001 | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 | 10 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|
| `L_func` post-export | 0.125113 | 0.125111 | 0.125111 | 0.125110 | **0.125099** | 0.125331 | 0.131299 | 0.175430 | 0.294384 | 0.287309 |
| `L_func` pre-export | 0.125113 | 0.125111 | 0.125111 | 0.125110 | 0.125118 | 0.125161 | 0.126109 | 0.144304 | 0.226865 | 0.276912 |

The curve is **flat to four decimal places out to t = 0.1** and then climbs
monotonically. The best point anywhere on the full-batch steepest-descent ray is
0.125099 against the identity's 0.125113 — an improvement of **0.011%**, twenty
times smaller than the 0.19% replay floor and nearly thirty times smaller than
the 0.31% gate. So this is not an over-stepping artefact and not a sign error in
the STE path (Stage 0 already proved the gradient is finite, non-zero, reaches
only the Cayley parameters, and moves R): **`R = I` sits at the bottom of a flat
basin of the binary functional error within the gauge class**, and every step
size large enough to matter walks out of it.

### 4. The extra-STE control reproduces this repository's central negative result

`0b_100` and `0b_200` improve the pre-Step-3 point substantially — `E_gauge`
0.14377 -> **0.13369** / **0.13242**, i.e. −7.0% and −7.9% — and then finish
**worse**: `E_final` +1.11% and +2.50% against Arm 0a. More STE before the common
Step 3 actively harms the final answer, and the harm grows with the number of
extra steps.

This is the same shape as Tests A and B in `SUMMARY.md`, now measured on a
starting-point intervention that is *pure optimisation of the same objective*
rather than a different initialisation rule. It sharpens the earlier conclusion:
it is not that "better initialisations happen not to help", it is that moving the
starting point toward a lower pre-Step-3 error *spends* something Step 3 needs.
The `J` and sign-flip columns say what: from the extra-STE starting points Step 3
flips **more** `down_proj` signs (U 1.29% / 1.59% vs the baseline's 0.98%) and
still arrives higher, so the extra STE has already consumed the easy sign flips
and left Step 3 a worse-conditioned remainder.

### 5. The export statistics lose ground, and the logs show it

Section 4.2 asks that a gain which disappears during export be visible. It is,
and it goes the other way from a gain: in every gauged arm the *post*-export
number is worse than the *pre*-export one.

| | pre-export | post-export |
|---|---|---|
| Arm 1, median of 16 random gauges (held-out) | 0.3140 | 0.3615 |
| Arm 3 @100, lr 3e-3 (held-out) | 0.2151 | 0.2302 |
| Arm 3 @100, lr 3e-3 (calibration `L_func`) | 0.19614 | 0.21176 |

The mechanism is straightforward. NanoQuant's export scale is the *mean
magnitude* of the continuous factor along the rank axis. Rotating a heavy-tailed
row of `U` toward a more Gaussian one lowers its mean |·| (toward
`sqrt(2/pi)·||u||/sqrt(b)`) while `sign(U R)` still has entries of magnitude
exactly 1, so the honestly recomputed scale systematically **under-scales** the
rotated reconstruction. Keeping the ADMM scales is better than re-deriving them —
but keeping them would not be NanoQuant's export rule, and section 4.2 asks for
the rule. Both numbers are reported for every arm.

## Verdict

**Stage 1 fails its gate, in all four cells, on the 0a leg.** No latent gauge
tested — geometric or function-selected — improves NanoQuant's final ~1-bit
block error. Stage 2 is therefore not entered (brief section 14: "only if Stage 1
passes").

Against the brief's outcome menu this is none of A–D. It is a fifth outcome the
menu did not anticipate, and it is more informative than a flat null:

* Gauge freedom **is** a real and large degree of freedom for the binary
  solution (Arm 1: 2.5x, ~50% of signs).
* The ADMM latent basis is **already a local optimum** within that freedom
  (Arm 2's fixed point at `R = I`, Arm 3's uniform ascent at every step size, the
  line search rising out of `t = 0` along both the gradient and random directions).
* And the binding constraint on the final answer is still Step 3, not the
  starting point — now demonstrated by a control that improves the starting point
  by 7.9% on the *same* objective and finishes 2.5% worse.

The scientific hypothesis in section 20 is well posed and its premise is
confirmed: `sign(U R) sign(V R)^T` genuinely differs from `sign(U) sign(V)^T`,
by a lot. What is not confirmed is that a better binary solution is reachable
from the ADMM basis by moving R — because the ADMM basis is where the reachable
optimum already is.

## Caveats

* One block, one seed, one layer. Stage 1 was scoped to `down_proj` on block 0
  and the gate was not passed, so seeds 1–2, block 8 and the full model were not
  spent (the same promotion rule the rest of this repository uses).
* The reproduced post-ADMM state is an *independent draw*, not a replay of the
  cached D2 state: the ADMM is chaotic (NOTES.md#2). Arm 0a lands 0.47% from the
  cached 0.1148516 for that reason. Every arm branches from the same reproduced
  state, so all within-table comparisons are controlled; only the comparison to
  the historical cell carries that extra 0.47%.
* Arm 0b uses NanoQuant's own importance-weighted Step-3 loss, because it is by
  definition the *ordinary NanoQuant* update; Arm 3 uses the unweighted relative
  functional loss the brief specifies. The two controls are step- and
  data-matched, not loss-matched, and — as the brief insists — not compute
  matched either.
* Block size b = 32 throughout, as specified. b ∈ {16, 64} was a Stage 2 item and
  was not reached. A dense rank x rank rotation was explicitly excluded by the
  brief and is the one part of the gauge class this run does not probe; Arm 1's
  result (a random *block* rotation is 2.5x worse) makes a dense random rotation
  a strictly worse bet, but a dense *optimised* rotation is not ruled out by
  anything measured here.
