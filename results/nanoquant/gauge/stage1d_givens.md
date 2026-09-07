# Stage 1D — exact discrete gauge search on `mlp.down_proj`, block 0

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
exhaustive — 10240 breakpoints, 9489 distinct binary models per plane, half the
naive search. And a naive column-wise flip count reports ~10240 "flips" for a
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
| distinct binary models per plane | 9489 |
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

Two strategies 4.4x apart in move count converge to the same binary solution to
five decimals *and to nearly the same sign pattern* (61 vs 57 net U flips, 2947 vs
2927 net V flips). After 378 accumulated rotations the tracked `R` has
orthogonality error **3e-15** and reproduces the scored factors to **6e-17**, so
the exact-equivalence premise held throughout.

Net sign change over the whole layer: **0.002% of U, 0.022% of V**. Step 3 flips
0.98% and 2.55%. The discrete gauge search moves ~100x fewer signs than Step 3.

## Result 3 — the three numbers

    E_ADMM  0.143766  ->  E_gauge  0.143728  ->  E_final  0.115354

| arm | E_final | Δ vs 0a | verdict |
|---|---|---|---|
| 0a baseline | 0.115388 | — | — |
| **0a duplicate** (identical state, re-run) | **0.115165** | **−0.193%** | replay floor |
| 4_cyclic | 0.115354 | −0.030% | **FAIL** |
| 4_cyclic, fixed ADMM scales | 0.115342 | −0.041% | **FAIL** |
| gate threshold | 0.115030 | −0.31% | |

The gauge's improvement is **0.15x** the difference between running the identical
baseline twice. Re-running 0a moves `E_final` 6.5x further than exhaustively
searching every reachable one-plane binary model does.

## What kind of failure this is

This is **not** the interesting failure. A "large pre-Step-3 gain destroyed by
Step 3" result would have implicated Step 3's path dependence and pointed
straight at compensation-aware rounding. That is not what happened: the
pre-Step-3 gain was itself microscopic (−0.026% held-out) and Step 3 neither
destroyed nor amplified it. The gauge does not contain the capacity in the first
place.

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

## Through the identical common Step 3

| arm | E_ADMM | E_gauge pre-export | E_gauge | E_final | Δ vs 0a | Δ vs 0b_200 | sign Δ U | sign Δ V |
|---|---|---|---|---|---|---|---|---|
| **0a baseline** | 0.143766 | — | 0.143766 | **0.115388** | — | — | 0.00% | 0.00% |
| 0a duplicate (replay floor) | 0.143766 | — | 0.143766 | **0.115165** | -0.19% | — | 0.00% | 0.00% |
| **4_cyclic** | 0.143766 | 0.143728 | 0.143728 | **0.115354** | -0.03% | -2.46% | 0.002% | 0.022% |
| **4_cyclic_fixedscale** | 0.143766 | 0.143728 | 0.143728 | **0.115342** | -0.04% | -2.47% | 0.002% | 0.022% |
| **4_greedy** | 0.143766 | 0.143730 | 0.143730 | **0.115356** | -0.03% | -2.46% | 0.002% | 0.022% |

Gate: a cell passes only if it beats **both** 0a and its matched 0b by more than δ = 0.0031. 0a threshold = 0.115030.
