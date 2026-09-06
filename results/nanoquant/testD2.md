# Test D2 — mirror descent, corrected (block 0)

**Verdict: no cell clears the floor. Best is `H2_conf` at lr = 3e-3,
`post` = 0.122876 against the joint STE control's 0.114852 — +6.99%, which is
22x the 0.31% determinism floor. Not within floor: clearly worse.**

Both prescribed fixes worked mechanically. Neither closed the gap, and the
sweep produced a cleaner explanation of why than either variant was designed to
test.

## Setup

Same banked post-ADMM block-0 state as Test D (`pre` = 0.143661403, cache
round-trip bit-identical), joint Step 3, 1024 optimizer steps, scales and bias on
`optimi.AdamW` at 1e-5 with the released cosine schedule. Only the binary update
differs. The Θ optimizer is `optimi.AdamW`, weight decay 0, with the same cosine
schedule as the control's (T_max = 1024, `eta_min` = 1e-4·lr) so that the two
differ in update rule and not in schedule.

**Sign assert passes in all 16 cells.** Both inits preserve `sign(proxy)`
exactly, and the block error of `sign(Θ₀)` reproduces the cached `pre` to the
last digit — 0.143661403 in every cell.

`c` for the `conf` init, solved per factor by bisection so that
mean|tanh(Θ₀)| = 0.9, lands between **1.75 and 2.90** across the 14 factors.

## The 16 cells

| variant | init | lr | `post` | vs control | sign flips |
|---|---|---|---|---|---|
| — | — | *STE control* | **0.114852** | — | **1.03215%** |
| R2 | conf | 1e-4 | 0.137863 | +20.04% | 0.00556% |
| R2 | conf | 3e-4 | 0.137249 | +19.50% | 0.01316% |
| R2 | conf | 1e-3 | 0.134471 | +17.08% | 0.04425% |
| R2 | conf | 3e-3 | 0.127191 | +10.74% | 0.18117% |
| R2 | corner | 1e-4 | 0.138009 | +20.16% | 0.00000% |
| R2 | corner | 3e-4 | 0.138001 | +20.16% | 0.00000% |
| R2 | corner | 1e-3 | 0.138013 | +20.17% | 0.00000% |
| R2 | corner | 3e-3 | 0.138059 | +20.21% | 0.00000% |
| H2 | conf | 1e-4 | 0.137181 | +19.44% | 0.00876% |
| H2 | conf | 3e-4 | 0.135900 | +18.33% | 0.02497% |
| H2 | conf | 1e-3 | 0.131113 | +14.16% | 0.09739% |
| **H2** | **conf** | **3e-3** | **0.122876** | **+6.99%** | 0.33798% |
| H2 | corner | 1e-4 | 0.137853 | +20.03% | 0.00000% |
| H2 | corner | 3e-4 | 0.137854 | +20.03% | 0.00000% |
| H2 | corner | 1e-3 | 0.137853 | +20.03% | 0.00000% |
| H2 | corner | 3e-3 | 0.137852 | +20.03% | 0.00000% |

All start from `pre` = 0.143661. **No cell diverged** — Test D's four blow-ups
are gone, and `post` now moves monotonically with lr instead of cliff-edging from
0.01% to 50% flips between two adjacent step sizes. Fix 1 did what it was for.

## What the sweep actually shows

### 1. The corner init cannot flip a sign at any swept lr — for a geometric reason

`atanh(0.95) = 1.8318`, so the corner init puts **every** |Θ₀| at 1.8318 exactly.
Adam's per-coordinate step is bounded by roughly the learning rate, so over 1024
steps the reachable displacement is at most ~`lr · 1024`: 0.1 at lr = 1e-4, and
3.07 at lr = 3e-3. Only the largest swept lr can cross 1.8318 at all, and only if
the gradient sign is coherent across essentially every step. Measured flip rate:
**exactly 0.00000% in all eight corner cells.**

Discarding the confidence ordering does not merely lose information — it removes
the low-|Θ| tail that is the only place a bounded optimizer can flip anything.

### 2. That accident is a useful control: sign flips are ~80% of Step 3

The eight zero-flip cells are Step 3 with the ADMM signs **frozen**, i.e. pure
scale tuning. They agree to five digits regardless of variant and lr
(H2: 0.137852-0.137854; R2: 0.138001-0.138059). So:

| | block error | share of the gain |
|---|---|---|
| post-ADMM `pre` | 0.143661 | — |
| scales only (signs frozen) | **0.137853** | 20.2% |
| STE control (scales + 1.03% flips) | **0.114852** | **79.8%** |

**About four fifths of what Step 3 buys comes from flipping roughly one percent
of the binary signs**, not from the scales.

### 3. Every cell lies on one post-vs-flip-rate curve — the mirror map is not the variable

Pooling all ten cells that flipped anything, across both D2 variants, both D2
inits, and Test D's best hard cell:

| cell | flips % | `post` |
|---|---|---|
| `R2_conf_1e-4` | 0.00556 | 0.137863 |
| `H2_conf_1e-4` | 0.00876 | 0.137181 |
| `D:H_3e-2` | 0.01072 | 0.137533 |
| `R2_conf_3e-4` | 0.01316 | 0.137249 |
| `H2_conf_3e-4` | 0.02497 | 0.135900 |
| `R2_conf_1e-3` | 0.04425 | 0.134471 |
| `H2_conf_1e-3` | 0.09739 | 0.131113 |
| `R2_conf_3e-3` | 0.18117 | 0.127191 |
| `H2_conf_3e-3` | 0.33798 | 0.122876 |
| `ctrl_ste` | 1.03215 | 0.114852 |

A single log-linear fit, `post = -0.00983·log10(flip) + 0.09886`, gives
**R² = 0.923** over a 200x range in flip rate, with a maximum residual of 0.0035
(3.1% of the control). Relaxed versus hard, `tanh` versus `sign`, mirror versus
straight-through, Adam versus SGD — none of it separates the cells once you
condition on how many signs moved. **The binary update rule is not the active
variable in Step 3. The sign-flip budget is.**

The control does sit **below** the trend: the fit predicts 0.11839 at 1.032%
flips against the measured 0.114852, so STE is ~3% better than the pooled curve
at its own flip rate. Two readings are consistent with the data and this sweep
cannot separate them — STE's flips may be better *chosen*, or its scale
co-adaptation may be better matched. Note the control is a 3x extrapolation
beyond the mirror cells' maximum flip rate, so that 3% is the least secure number
here.

### 4. Why the mirror cells run out of flips

The `conf` init reaches mean|tanh(Θ₀)| = 0.9, so most coordinates sit at
|Θ₀| ≈ 1.5-2.9 (the solved `c` range), and only the low-|Θ| tail is reachable
within `lr · 1024`. Flip rate scales smoothly with lr — 0.0056 → 0.0132 → 0.0442
→ 0.181% for R2, 0.0088 → 0.0250 → 0.0974 → 0.338% for H2 — roughly 3.5x per 3x
of lr, but it starts three decades below STE's 1.03%. Linear extrapolation on
that trend puts the lr matching STE's flip rate near **1.7e-2**, outside the
prescribed sweep. Whether the trend continues or destabilises there is untested.

STE has no such ceiling because it operates on the raw latent, whose magnitude
distribution the ADMM already set near zero for uncertain coordinates; the
`atanh` reparameterisation stretches exactly those coordinates away from the
decision boundary.

### Per-layer detail, best cell

Cumulative `post` with layers 1..k tuned and the rest held at their ADMM signs:

| layer | `H2_conf_3e-3` | `H2_corner_3e-3` (frozen signs) |
|---|---|---|
| `q_proj` | 0.139781 | 0.137852 |
| `v_proj` | 0.139417 | 0.137852 |
| `o_proj` | 0.138541 | 0.137852 |
| `k_proj` | 0.138587 | 0.137852 |
| `gate_proj` | 0.135284 | 0.137852 |
| `up_proj` | 0.131126 | 0.137852 |
| `down_proj` | **0.122876** | 0.137852 |

The gain is concentrated in the three MLP layers, matching where STE puts its
flips.

| layer | H2 flips V/U % | STE flips V/U % | H2 `J` post | STE `J` post | `J` pre |
|---|---|---|---|---|---|
| `q_proj` | 0.0655/0.1160 | 0.3101/0.4783 | 8.052e-3 | 9.275e-3 | 7.735e-3 |
| `v_proj` | 0.1888/0.2223 | 0.7678/0.4247 | 2.609e-1 | 2.557e-1 | 2.637e-1 |
| `o_proj` | 0.1645/0.2276 | 0.5364/0.6666 | 8.366e-2 | 8.653e-2 | 8.212e-2 |
| `k_proj` | 0.0900/0.1053 | 0.1567/0.0860 | 2.034e-2 | 1.923e-2 | 1.913e-2 |
| `gate_proj` | 0.0217/0.7057 | 0.0154/**3.4830** | 1.957e-1 | 2.211e-1 | 1.914e-1 |
| `up_proj` | 0.0435/0.8250 | 0.0339/**3.9446** | 2.501e-1 | 2.767e-1 | 2.508e-1 |
| `down_proj` | **1.1839**/0.7718 | **2.5589**/0.9877 | 1.630e-1 | 1.691e-1 | 1.662e-1 |

H2's flip distribution is roughly a 4-5x scaled-down copy of STE's, except in
`down_proj` V where it is only 2.2x down — the one place the mirror step keeps up.
As in Tests A, B and D, `J` rises where block error falls fastest (`gate`, `up`).

## Verdict

Best cell `H2_conf_3e-3` is **+6.99%** worse than the control, 22x the 0.31%
floor. Not within floor. Test D2 does not promote.

The corrected experiment is nonetheless more informative than the original. Fix 1
(Adam in the dual space) removed the divergence and the flip-rate cliff
completely. Fix 2 (corner-aware init) removed variant R's catastrophic hardening
gap — `R2_conf_3e-3` reaches 0.127191 where Test D's `R_1e-2` hardened to 1.593.
What remains is not a defect of mirror descent: all ten flipping cells and the
STE control fall on one log-linear `post`-versus-flip-rate curve at R² = 0.923.
Within the prescribed lr sweep the mirror parameterisation simply cannot buy
enough sign flips, because `atanh` pushes the uncertain coordinates that STE
flips cheaply away from the decision boundary.

If this line is worth one more cell, the question it now poses is narrow and
testable: **run `H2_conf` at lr ∈ {1e-2, 3e-2} and see whether the curve
continues to STE's flip rate or breaks.** That distinguishes "the mirror map is
equivalent to STE at matched flip budget" from "STE's flips are better chosen".

## Artifacts

* `artifacts/testD2/runs/*.json` — 16 cells, per-layer `J`, flips, cumulative curves, solved `c`
* `nqx/testD2.py`, `testD2.sh`
