# Stage 1D — discrete gauge search on `mlp.down_proj`, **rank block 0 only**

> **Scope, stated up front.** `down_proj` has rank 1600. At b = 32 that is 50 rank
> blocks x C(32,2) = 496 planes = **24,800** within-block Givens planes, and
> C(1600,2) = **1,279,200** planes without the block restriction. This experiment
> searched **496** of them — rank block 0. That is **2% of the block-diagonal
> gauge** and **0.04% of all pairwise planes**. Latent coordinates are
> permutation-symmetric, so there is no reason coordinates 0-31 are special or
> representative. Every conclusion below is therefore about **rank block 0**, and
> the whole-layer question is open and being run separately.
>
> **What this is, precisely:** exact discrete enumeration of every sign pattern a
> plane can reach, + *approximate* quadratic ranking of those patterns, + exact
> bf16 validation of the accumulated result. It is not "exact optimisation of the
> true binary objective": the oracle scores a dense equivalent weight `W` in fp64
> while deployment computes `((x s_pre) B_V^T) B_U^T s_post` in bf16, and the
> per-move disagreement is 38% (see `eps_Delta`). Aggregate conclusions rest on
> the repeated bf16 evaluations, not on the oracle.

Stage 1 searched the latent gauge with a straight-through gradient and concluded
that `R = I` was a local optimum. **That conclusion was too strong**, and this
experiment is the correction.

The binary objective is *piecewise constant in the sign pattern*: rotating along
any ray out of `R = I` changes nothing until some entry of `U R` or `V R` crosses
zero. Stage 1's "flat basin out to t = 0.1" was therefore not evidence about the
quality of `R = I` — it was the signature of a region containing **no sign flips
at all**, which is what a piecewise-constant objective must look like near any
point. What Stage 1 established is that **STE fails to search the gauge**, not
that the gauge contains nothing.

So this experiment throws the surrogate away and optimises the true binary
objective directly.

## The move class and why it is exactly enumerable

A Givens rotation `R_ij(theta)` in the rank space touches only columns i, j of U
and rows i, j of V, and leaves `U R (V R)^T = U V^T` exact. The deployed sign
patterns are therefore piecewise constant in theta and change only at zeros:

    cos.u_i[m] + sin.u_j[m] = 0   =>   theta = atan2(-u_i[m], u_j[m])

**The objective is pi/2-periodic, not pi-periodic.** `R(theta + pi/2)` swaps rank
coordinates i and j (with a sign) and `b_i c_i + b_j c_j` is invariant under that,
so the model is bit-identical — verified to the last bit:

| theta | f(theta) | f(theta + pi/2) | naive flip count at theta+pi/2 |
|---|---|---|---|
| 0.10 | −386.190002 | −386.190002 | 10320 |
| 0.70 | −238.139374 | −238.139374 | 14319 |

Two consequences. The two crossings of a coordinate are exactly pi/2 apart, so
mod pi/2 each coordinate contributes **one** breakpoint and `[0, pi/2)` is
exhaustive — 10240 raw breakpoints per plane, which after de-duplication leave
**median 9136 (min 9051, max 9237)** distinct binary models, half the naive
search. And a naive column-wise flip count reports ~10240 "flips" for a
model that has not changed at all, so flip counts must be measured modulo the
relabelling and accepted angles canonicalised to `[-pi/4, pi/4)`.

## The oracle: exact, and forward-pass-free

`down_proj`'s output reaches the block output through a plain residual add and
nothing else, so with every other projection frozen the block output is **affine**
in its weight and the calibration block error is exactly

    E(W) = ( ||T||^2 - 2<G, W> + tr(W H W^T) ) / ||Y_fp||^2,
    G = T^T Z,  H = Z^T Z,  Z = down_proj's input activations

Candidate weights are scored with no forward pass at all. Exploiting the low-rank
structure of a two-coordinate change reduces a whole plane to six batched
matmuls. This is what makes exhaustive enumeration affordable: ~0.5 s to evaluate
every one of the 9489 distinct binary models a plane can reach, against ~9489
block forwards over 128x2048 tokens otherwise.

**fp64 is required, not optional.** `f` is a difference of terms of order 1e4
(`c^T H~ c` over 8192 entries of +-1) cancelling down to order 1e2, so fp32 leaves
**1.2e-3** of absolute noise — 0.25% of a median improving move. Harmless when
judging one move, unacceptable when accumulating several hundred, because banked
arithmetic noise is indistinguishable from a banked gain. fp64 puts the residual
at **4.5e-13** (median exactly 0) for about 2x the cost, and acceptance
additionally requires a move to clear 10x its own plane's measured residual.

## Result 1 — identity is NOT a discrete local optimum

| | |
|---|---|
| planes in rank block 0 | 496 |
| distinct binary models per plane | min 9051, median 9136, max 9237 |
| **planes containing a strictly better binary model** | **482 (97.2%)** |
| sign flips at the best angle (relabel-aware) | median **44** of 20480 touched (0.21%) |
| gain per plane | median −0.00035%, best −0.0055% |

97.2% of one-plane neighbourhoods contain a better binary model, reachable by a
median of 44 coordinated sign flips. This is exactly the regime Stage 1 never
sampled: every gauge it evaluated either flipped ~0.0% of signs — and so *was*
the baseline — or had its flips chosen by the discredited STE gradient.

**So the Stage 1 verdict is now split in two.** The gauge class does contain
better binary points, densely. STE simply cannot find them.

## Result 2 — but those improvements are near-perfectly redundant

Running them is a different matter from counting them.

| sweep | accepted | cumulative real bf16 gain | increment |
|---|---|---|---|
| 1 | 213 | −0.0320% | −0.0320% |
| 2 | 103 | −0.0396% | −0.0076% |
| 3 | 46 | −0.0415% | −0.0019% |
| 4 | 16 | −0.0407% | ~0 (within `eps_Delta`) |

Only **213 of the 482** pilot-improving planes survived to be accepted in
sequence: applying earlier moves destroys later ones' improvements. Cumulative
gain scales as

    gain  ~  m^0.51

— **sqrt(m) to two digits**, the signature of a random walk rather than additive
capacity — with **2817 reversals** (signs flipped and later flipped back) against
3008 net changes, i.e. 33% wasted motion.

**Best-improvement reaches the same fixed point 4.4x faster**, which is the
control that matters: the saturation is a property of the landscape, not of a
badly ordered descent.

| | accepts | planes visited | calib gain | held-out `E_gauge` | net U flips | net V flips | flip events | reversals |
|---|---|---|---|---|---|---|---|---|
| cyclic | 378 | 1984 | −0.0407% | 0.143728 | 61 | 2947 | 8642 | 2817 |
| greedy | **85** | 1418 | −0.0418% | 0.143730 | 57 | 2927 | 4774 | 895 |

Two strategies 4.4x apart in move count converge to the same *error* to five
decimals. They do **not** converge to the same *solution*. Similar counts of
changed signs do not mean the same signs changed, and they do not:

| | cyclic | greedy | Hamming(cyc, greedy) | shared | union | Jaccard |
|---|---|---|---|---|---|---|
| U signs changed vs ADMM | 61 | 57 | **90** | 14 | 104 | **0.13** |
| V signs changed vs ADMM | 2947 | 2927 | **4290** | 792 | 5082 | **0.16** |

The two searches agree on only ~14% of the sign changes they make, and the
Hamming distance between their solutions (90 / 4290) is *larger* than either
one's distance from the ADMM point (61 / 2947). So they reach nearly identical
error through **largely disjoint** sets of sign flips.

That is a sharper statement of the redundancy than the sweep counts give. The
landscape is not one improving basin found by two routes; it is massively
degenerate -- many different small sign-change sets buy the same ~0.04%, and none
of them buys more. After 378 accumulated rotations the tracked `R` has
orthogonality error **3e-15** and reproduces the scored factors to **6e-17**, so
the exact-equivalence premise held throughout.

Net sign change over the whole layer: **0.002% of U, 0.022% of V**. Step 3 flips
0.98% and 2.55%. The discrete gauge search moves ~100x fewer signs than Step 3.

## Result 3 — the three numbers, scored against the baseline *distribution*

    E_ADMM  0.143766  ->  E_gauge  0.143728  ->  E_final  0.115354

`E_final` is the only stochastic quantity in this pipeline (NOTES.md#6):
everything through `Q_NQ` export is bit-identical across processes, while Step 3
diverges from a last-bit difference because it flips ~1% of the signs. Re-running
the **identical** `0a` state eight times gives

    mean 0.115231804   sd 8.69e-05 (0.0754%)   spread 0.2329%

so a single-run difference below **2 sd = 0.15%** carries no information. The
single `0a` draw originally used as "the baseline" is the **worst of the eight**.

| arm | E_final | vs baseline mean | z | verdict |
|---|---|---|---|---|
| baseline distribution (n = 8) | 0.115232 +- 0.000087 | — | — | — |
| 4_cyclic | 0.115354 | **+0.106%** | +1.41 | within noise |
| 4_cyclic, fixed ADMM scales | 0.115342 | **+0.096%** | +1.27 | within noise |
| 4_greedy | 0.115356 | **+0.107%** | +1.43 | within noise |
| gauge, 4 rank blocks | 0.115399 | **+0.145%** | +1.92 | within noise |
| `0b_200` extra STE (for contrast) | 0.118267 | +2.634% | +34.9 | **significant** |

Every discrete-gauge arm is *worse* than the baseline mean, not better, and none
is significant on its own. All four sit at z = +1.3 to +1.9 — consistently on the
worse side, and the arm with the largest `E_gauge` gain (4 blocks) is the worst,
which is the Outcome-C signature. At 1.9 sd that is suggestive and not
established; distinguishing it needs replicates of the gauge arm too, not only of
the baseline.

For contrast, the extra-STE control sits at z = +17 and +35. That regression was
never noise-limited and is unaffected by any of this.

## What kind of failure this is — and what it is not

This is **not** the interesting failure. A "large pre-Step-3 gain destroyed by
Step 3" result would have implicated Step 3's path dependence and pointed
straight at compensation-aware rounding. That is not what happened: the
pre-Step-3 gain was itself microscopic (−0.026% held-out) and Step 3 neither
destroyed nor amplified it.

**The supported claim is narrow:** *rank block 0 contains dense but almost
entirely redundant discrete gauge capacity, saturating near −0.04%.* The
tempting generalisation — "the latent gauge does not contain the capacity" — is
**not supported by this experiment** and should not be made. 496 of 24,800
within-block planes were searched. One block yielding −0.042% would be −2.09% if
the 50 blocks were additive and −0.30% under sqrt composition, which straddles
the 0.31% gate; the observed magnitude is precisely in the range where a
single-block result cannot decide the layer.

That distinguishes it sharply from the extra-STE control, where a **real** 7.9%
pre-Step-3 gain was converted into a 2.5% final regression. Those are different
mechanisms and only the latter says anything about Step 3.

## Methodological results worth keeping

* **`eps_Delta`, not the absolute oracle gap, is the quantity that matters** for
  ranking candidates. Measuring it inverted the expected picture: the oracle is
  **unreliable for a single move** (`eps_Delta` = 4.2e-6 against a 1.1e-5 move,
  38% error) and **reliable in aggregate** — `eps_Delta` stays flat at ~2e-6 while
  the accumulated signal grows monotonically, so it behaves as a constant offset
  rather than a compounding bias. Exactly backwards from the natural assumption.
* **fp32 cancellation** at 1.2e-3 absolute would have let hundreds of accepted
  moves bank numerical noise as gain. See NOTES.md.
* **The quadratic oracle** is reusable: any layer whose output reaches the block
  output affinely admits it, which covers `down_proj` and `o_proj` in a Llama
  block.


---

# Generated tables

## The oracle

| case | real bf16 forward | quadratic oracle | relative gap |
|---|---|---|---|
| identity | 0.12511274 | 0.12510566 | 5.66e-05 |
| haar_a | 0.35037689 | 0.35019520 | 5.19e-04 |
| haar_b | 0.35143047 | 0.35124217 | 5.36e-04 |

The absolute gap is largely a common offset and is **not** the quantity that matters for ranking candidates; what matters is the accuracy of the *difference* `eps_Delta = |dE_oracle - dE_real|`, which is measured against the real forward at every checkpoint of every descent run below.

## Pilot — one 32-block, every plane, every reachable binary model

| | |
|---|---|
| planes (all pairs in rank block 0) | 496 |
| distinct binary models per plane (mod pi/2) | 9201 |
| **planes containing an improving model** | **482 (97.2%)** |
| sign flips at the best angle (relabel-aware) | median **44**, min 1, max 798 |
| — as a fraction of the 20480 touched entries | 0.21% |
| gain per plane | median -0.00035%, best -0.00549% |
| incremental-vs-direct gate | 7.89e-07 |
| fp64 residual arithmetic noise per plane | median 0.0e+00, max 9.1e-13 |

## Coordinate descent — `cyclic`

Rank blocks [0], 496 planes, strategy `cyclic`. Calibration data only.

| sweep | accepted | E oracle | E real bf16 | cum vs start (oracle) | cum vs start (real) |
|---|---|---|---|---|---|
| 1 | 213 | 0.12506714 | 0.12507264 | -0.0302% | **-0.0320%** |
| 2 | 103 | 0.12505901 | 0.12506320 | -0.0367% | **-0.0396%** |
| 3 | 46 | 0.12505652 | 0.12506087 | -0.0387% | **-0.0415%** |
| 4 | 16 | 0.12505600 | 0.12506187 | -0.0391% | **-0.0407%** |

| accepts | E oracle | E real bf16 | cum ΔE oracle | cum ΔE real | eps_Delta (cum) | sign Δ U | sign Δ V | reversals |
|---|---|---|---|---|---|---|---|---|
| 25 | 0.12509575 | 0.12509924 | -9.149e-06 | -1.350e-05 | 4.35e-06 | 0.000% | 0.005% | 376 |
| 50 | 0.12509069 | 0.12509425 | -1.422e-05 | -1.849e-05 | 4.27e-06 | 0.000% | 0.006% | 643 |
| 75 | 0.12508491 | 0.12508897 | -2.000e-05 | -2.377e-05 | 3.77e-06 | 0.001% | 0.010% | 936 |
| 100 | 0.12508124 | 0.12508600 | -2.366e-05 | -2.673e-05 | 3.07e-06 | 0.001% | 0.012% | 1232 |
| 125 | 0.12507811 | 0.12508275 | -2.679e-05 | -2.999e-05 | 3.20e-06 | 0.001% | 0.013% | 1336 |
| 150 | 0.12507464 | 0.12507960 | -3.026e-05 | -3.314e-05 | 2.88e-06 | 0.001% | 0.015% | 1566 |
| 175 | 0.12507193 | 0.12507884 | -3.298e-05 | -3.390e-05 | 9.22e-07 | 0.001% | 0.017% | 1676 |
| 200 | 0.12506838 | 0.12507411 | -3.652e-05 | -3.863e-05 | 2.10e-06 | 0.002% | 0.021% | 1919 |
| 225 | 0.12506638 | 0.12507209 | -3.852e-05 | -4.064e-05 | 2.12e-06 | 0.002% | 0.021% | 2062 |
| 250 | 0.12506407 | 0.12506981 | -4.083e-05 | -4.293e-05 | 2.10e-06 | 0.002% | 0.022% | 2216 |
| 275 | 0.12506236 | 0.12506800 | -4.254e-05 | -4.474e-05 | 2.20e-06 | 0.002% | 0.022% | 2319 |
| 300 | 0.12505995 | 0.12506532 | -4.496e-05 | -4.742e-05 | 2.46e-06 | 0.002% | 0.022% | 2493 |
| 325 | 0.12505865 | 0.12506282 | -4.625e-05 | -4.992e-05 | 3.66e-06 | 0.002% | 0.022% | 2635 |
| 350 | 0.12505729 | 0.12506170 | -4.761e-05 | -5.103e-05 | 3.43e-06 | 0.002% | 0.023% | 2746 |
| 375 | 0.12505624 | 0.12506206 | -4.867e-05 | -5.068e-05 | 2.02e-06 | 0.002% | 0.022% | 2816 |

Log-log fit of cumulative real gain against accepted moves: **gain ∝ m^0.50** (1.0 = independent/additive, 0.5 = random-walk/interfering).

| | held-out block error |
|---|---|
| `E_ADMM` (start) | 0.143766 |
| `E_gauge`, fixed ADMM scales | 0.143728 |
| `E_gauge`, full `Q_NQ` export | 0.143728 |
| planes visited / moves accepted | 1984 / 378 |

## Coordinate descent — `cyclic_b0123`

Rank blocks [0, 1, 2, 3], 1984 planes, strategy `cyclic`. Calibration data only.

| sweep | accepted | E oracle | E real bf16 | cum vs start (oracle) | cum vs start (real) |
|---|---|---|---|---|---|
| 1 | 914 | 0.12494355 | 0.12494706 | -0.1290% | **-0.1324%** |

| accepts | E oracle | E real bf16 | cum ΔE oracle | cum ΔE real | eps_Delta (cum) | sign Δ U | sign Δ V | reversals |
|---|---|---|---|---|---|---|---|---|
| 25 | 0.12509575 | 0.12509924 | -9.149e-06 | -1.350e-05 | 4.35e-06 | 0.000% | 0.005% | 376 |
| 50 | 0.12509069 | 0.12509425 | -1.422e-05 | -1.849e-05 | 4.27e-06 | 0.000% | 0.006% | 643 |
| 75 | 0.12508491 | 0.12508897 | -2.000e-05 | -2.377e-05 | 3.77e-06 | 0.001% | 0.010% | 936 |
| 100 | 0.12508124 | 0.12508600 | -2.366e-05 | -2.673e-05 | 3.07e-06 | 0.001% | 0.012% | 1232 |
| 125 | 0.12507811 | 0.12508275 | -2.679e-05 | -2.999e-05 | 3.20e-06 | 0.001% | 0.013% | 1336 |
| 150 | 0.12507464 | 0.12507960 | -3.026e-05 | -3.314e-05 | 2.88e-06 | 0.001% | 0.015% | 1566 |
| 175 | 0.12507193 | 0.12507884 | -3.298e-05 | -3.390e-05 | 9.22e-07 | 0.001% | 0.017% | 1676 |
| 200 | 0.12506838 | 0.12507411 | -3.652e-05 | -3.863e-05 | 2.10e-06 | 0.002% | 0.021% | 1919 |
| 225 | 0.12506236 | 0.12506791 | -4.254e-05 | -4.483e-05 | 2.29e-06 | 0.002% | 0.024% | 2174 |
| 250 | 0.12505776 | 0.12506224 | -4.715e-05 | -5.050e-05 | 3.35e-06 | 0.002% | 0.027% | 2454 |
| 275 | 0.12505299 | 0.12505658 | -5.192e-05 | -5.616e-05 | 4.24e-06 | 0.002% | 0.031% | 2782 |
| 300 | 0.12504815 | 0.12505188 | -5.676e-05 | -6.086e-05 | 4.10e-06 | 0.003% | 0.035% | 3063 |
| 325 | 0.12504178 | 0.12504569 | -6.312e-05 | -6.705e-05 | 3.93e-06 | 0.003% | 0.039% | 3579 |
| 350 | 0.12503764 | 0.12504186 | -6.726e-05 | -7.087e-05 | 3.61e-06 | 0.003% | 0.041% | 3827 |
| 375 | 0.12503381 | 0.12503784 | -7.109e-05 | -7.489e-05 | 3.80e-06 | 0.003% | 0.043% | 4102 |
| 400 | 0.12503128 | 0.12503395 | -7.363e-05 | -7.879e-05 | 5.16e-06 | 0.003% | 0.043% | 4226 |
| 425 | 0.12502733 | 0.12503055 | -7.758e-05 | -8.219e-05 | 4.62e-06 | 0.004% | 0.046% | 4475 |
| 450 | 0.12502460 | 0.12502900 | -8.030e-05 | -8.374e-05 | 3.44e-06 | 0.004% | 0.047% | 4597 |
| 475 | 0.12501553 | 0.12502263 | -8.938e-05 | -9.011e-05 | 7.34e-07 | 0.004% | 0.052% | 4952 |
| 500 | 0.12501098 | 0.12501513 | -9.392e-05 | -9.761e-05 | 3.69e-06 | 0.004% | 0.056% | 5224 |
| 525 | 0.12500720 | 0.12501127 | -9.771e-05 | -1.015e-04 | 3.76e-06 | 0.004% | 0.058% | 5454 |
| 550 | 0.12500373 | 0.12500822 | -1.012e-04 | -1.045e-04 | 3.34e-06 | 0.005% | 0.059% | 5776 |
| 575 | 0.12500077 | 0.12500515 | -1.041e-04 | -1.076e-04 | 3.46e-06 | 0.005% | 0.060% | 5964 |
| 600 | 0.12499700 | 0.12500133 | -1.079e-04 | -1.114e-04 | 3.50e-06 | 0.005% | 0.062% | 6097 |
| 625 | 0.12499340 | 0.12499822 | -1.115e-04 | -1.145e-04 | 3.02e-06 | 0.005% | 0.066% | 6408 |
| 650 | 0.12498974 | 0.12499552 | -1.152e-04 | -1.172e-04 | 2.05e-06 | 0.005% | 0.067% | 6602 |
| 675 | 0.12498756 | 0.12499313 | -1.173e-04 | -1.196e-04 | 2.26e-06 | 0.005% | 0.068% | 6776 |
| 700 | 0.12498349 | 0.12498967 | -1.214e-04 | -1.231e-04 | 1.66e-06 | 0.005% | 0.071% | 6990 |
| 725 | 0.12497487 | 0.12498178 | -1.300e-04 | -1.310e-04 | 9.26e-07 | 0.006% | 0.074% | 7281 |
| 750 | 0.12496939 | 0.12497578 | -1.355e-04 | -1.370e-04 | 1.44e-06 | 0.006% | 0.077% | 7508 |
| 775 | 0.12496649 | 0.12497303 | -1.384e-04 | -1.397e-04 | 1.30e-06 | 0.006% | 0.078% | 7610 |
| 800 | 0.12496313 | 0.12496989 | -1.418e-04 | -1.428e-04 | 1.08e-06 | 0.006% | 0.081% | 7774 |
| 825 | 0.12495942 | 0.12496623 | -1.455e-04 | -1.465e-04 | 1.03e-06 | 0.006% | 0.084% | 8002 |
| 850 | 0.12495346 | 0.12495799 | -1.514e-04 | -1.548e-04 | 3.30e-06 | 0.006% | 0.086% | 8258 |
| 875 | 0.12494897 | 0.12495357 | -1.559e-04 | -1.592e-04 | 3.24e-06 | 0.007% | 0.090% | 8602 |
| 900 | 0.12494537 | 0.12495010 | -1.595e-04 | -1.626e-04 | 3.11e-06 | 0.007% | 0.092% | 8895 |

Log-log fit of cumulative real gain against accepted moves: **gain ∝ m^0.76** (1.0 = independent/additive, 0.5 = random-walk/interfering).

| | held-out block error |
|---|---|
| `E_ADMM` (start) | 0.143766 |
| `E_gauge`, fixed ADMM scales | 0.143605 |
| `E_gauge`, full `Q_NQ` export | 0.143606 |
| planes visited / moves accepted | 1984 / 914 |

## Coordinate descent — `greedy`

Rank blocks [0], 496 planes, strategy `greedy`. Calibration data only.

| sweep | accepted | E oracle | E real bf16 | cum vs start (oracle) | cum vs start (real) |
|---|---|---|---|---|---|
| 1 | 85 | 0.12505883 | 0.12506038 | -0.0368% | **-0.0418%** |

| accepts | E oracle | E real bf16 | cum ΔE oracle | cum ΔE real | eps_Delta (cum) | sign Δ U | sign Δ V | reversals |
|---|---|---|---|---|---|---|---|---|
| 25 | 0.12507004 | 0.12507149 | -3.487e-05 | -4.125e-05 | 6.38e-06 | 0.001% | 0.020% | 300 |
| 50 | 0.12506196 | 0.12506280 | -4.295e-05 | -4.994e-05 | 6.99e-06 | 0.002% | 0.022% | 706 |
| 75 | 0.12505929 | 0.12505993 | -4.561e-05 | -5.280e-05 | 7.19e-06 | 0.002% | 0.022% | 865 |

Log-log fit of cumulative real gain against accepted moves: **gain ∝ m^0.23** (1.0 = independent/additive, 0.5 = random-walk/interfering).

| | held-out block error |
|---|---|
| `E_ADMM` (start) | 0.143766 |
| `E_gauge`, fixed ADMM scales | 0.143730 |
| `E_gauge`, full `Q_NQ` export | 0.143730 |
| planes visited / moves accepted | 1418 / 85 |

## Coordinate descent — `screen_k16`

Rank blocks 0, 496 planes, strategy `screen`. Calibration data only.

| sweep | accepted | E oracle | E real bf16 | cum vs start (oracle) | cum vs start (real) |
|---|---|---|---|---|---|
| 1 | 13 | 0.12509359 | 0.12509589 | -0.0096% | **-0.0135%** |

| | held-out block error |
|---|---|
| `E_ADMM` (start) | 0.143766 |
| `E_gauge`, fixed ADMM scales | 0.143762 |
| `E_gauge`, full `Q_NQ` export | 0.143762 |
| planes visited / moves accepted | 541 / 13 |

## Coordinate descent — `screen_k32`

Rank blocks 0, 496 planes, strategy `screen`. Calibration data only.

| sweep | accepted | E oracle | E real bf16 | cum vs start (oracle) | cum vs start (real) |
|---|---|---|---|---|---|
| 1 | 20 | 0.12508627 | 0.12509023 | -0.0155% | **-0.0180%** |

| | held-out block error |
|---|---|
| `E_ADMM` (start) | 0.143766 |
| `E_gauge`, fixed ADMM scales | 0.143754 |
| `E_gauge`, full `Q_NQ` export | 0.143754 |
| planes visited / moves accepted | 582 / 20 |

## Coordinate descent — `screen_k64`

Rank blocks 0, 496 planes, strategy `screen`. Calibration data only.

| sweep | accepted | E oracle | E real bf16 | cum vs start (oracle) | cum vs start (real) |
|---|---|---|---|---|---|
| 1 | 33 | 0.12507720 | 0.12508226 | -0.0227% | **-0.0244%** |

| accepts | E oracle | E real bf16 | cum ΔE oracle | cum ΔE real | eps_Delta (cum) | sign Δ U | sign Δ V | reversals |
|---|---|---|---|---|---|---|---|---|
| 25 | 0.12507845 | 0.12508357 | -2.721e-05 | -2.917e-05 | 1.96e-06 | 0.001% | 0.015% | 321 |

| | held-out block error |
|---|---|
| `E_ADMM` (start) | 0.143766 |
| `E_gauge`, fixed ADMM scales | 0.143746 |
| `E_gauge`, full `Q_NQ` export | 0.143746 |
| planes visited / moves accepted | 654 / 33 |

## Through the identical common Step 3

| arm | E_ADMM | E_gauge pre-export | E_gauge | E_final | Δ vs 0a | Δ vs 0b_200 | sign Δ U | sign Δ V |
|---|---|---|---|---|---|---|---|---|
| **0a baseline** | 0.143766 | — | 0.143766 | **0.115388** | — | — | 0.00% | 0.00% |
| 0a duplicate (replay floor) | 0.143766 | — | 0.143766 | **0.115165** | -0.19% | — | 0.00% | 0.00% |
| **4_cyclic** | 0.143766 | 0.143728 | 0.143728 | **0.115354** | -0.03% | -2.46% | 0.002% | 0.022% |
| **4_cyclic_b0123** | 0.143766 | 0.143605 | 0.143606 | **0.115393** | +0.00% | -2.43% | 0.007% | 0.093% |
| **4_cyclic_fixedscale** | 0.143766 | 0.143728 | 0.143728 | **0.115342** | -0.04% | -2.47% | 0.002% | 0.022% |
| **4_greedy** | 0.143766 | 0.143730 | 0.143730 | **0.115356** | -0.03% | -2.46% | 0.002% | 0.022% |

Gate: a cell passes only if it beats **both** 0a and its matched 0b by more than δ = 0.0031. 0a threshold = 0.115030.
