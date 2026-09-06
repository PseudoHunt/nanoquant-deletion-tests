# Test B — full input covariance in the ADMM (Change 2)

**Verdict: not promoted. Test B improves `pre` on both blocks (−2.0%, −8.9%) and
does not convert it: `post` is +3.4% on block 0 and +0.13% on block 8 (inside the
noise floor).**

Along the way it produced a result that matters more than the verdict: applied
exactly as specified, **the change makes the ADMM diverge on every transposed
layer**, for a reason that has nothing to do with the hypothesis.

## What was implemented

In the `A`-update, `(X_A^T X_A + stab) A_ls^T = X_A^T W_norm^T + rho(Z-U)` becomes
`(X_A^T H~ X_A + stab) A_ls^T = X_A^T H~ W_norm^T + rho(Z-U)`, with
`H~ = D_in^-1 H_in D_in^-1`. The `B`-update and every `rank1_approx` sign
projection are untouched, so the projection stays in the original basis.
`H~ W_norm^T` is precomputed once per layer; the per-iteration cost is one
`d_in x d_in` by `d_in x r` product, about 4 ms, so ADMM time per cell rises from
33 s to roughly 50 s.

### Transposed layers

Upstream transposes any layer with `d_out < d_in` (`k_proj`, `v_proj`,
`down_proj`) and solves the flipped problem. There, the factor carrying the
input-covariance weighting is the inner **`B`**-update, not the inner
`A`-update — the brief's "leave the `B`-update untouched" is written for the
direct branch. Taken literally it would leave 3 of 7 layer types, including the
largest, completely unchanged. Both branches therefore apply the identical
operator to the update of the **output-side** factor, which is the same
mathematical operation in both cases (the objective `tr(M^T H M)` puts `H` on the
summed index in both).

### The normalisation that the change requires

Applied as written, the H-weighted ADMM explodes on transposed layers:

| layer | | 60 it | 150 it | 400 it |
|---|---|---|---|---|
| `q_proj` (direct) | stock `J` | 9.01e-3 | 7.76e-3 | 7.18e-3 |
| | H-weighted `J` | 1.05e-2 | 7.40e-3 | **5.99e-3** |
| `v_proj` (transposed) | stock `J` | 2.96e-1 | 2.77e-1 | 2.62e-1 |
| | H-weighted `J` | 2.97e+2 | 3.39e+9 | **9.17e+31** |
| `down_proj` (transposed) | stock `J` | 2.25e-1 | 2.09e-1 | — |
| | H-weighted `J` | 2.77e+2 | 8.08e+14 | — |

The cause is in upstream's ADMM bookkeeping. `_admm_solve_step` puts
`rho * mean(diag(X^T X)) + reg` on the LHS against `rho * (Z - U)` on the RHS.
That is self-consistent only because their `X` has unit-norm columns, so
`mean(diag(X^T X)) = 1` exactly. Weighting by `H~` moves that mean and silently
rescales the consensus penalty relative to the data term; the transposed branch,
where the *unweighted* update is the badly conditioned one (`X_A` is 512 x 384
for `v_proj`), loses the consensus pull and diverges.

The fix is a pure rescaling, `H~ <- H~ / mean(diag H~)`. The least-squares
solution `(X^T H X)^-1 X^T H Y` is invariant to the scale of `H`, so no estimate
changes; only the `rho` arithmetic returns to its calibration. With it:

| layer | stock `J` @400 | H~ normalised `J` @400 |
|---|---|---|
| `v_proj` | 2.624e-1 | **2.221e-1** (−15.4%, flat from 150 it) |
| `q_proj` | 7.184e-3 | **7.005e-3** (−2.5%) |

It is also a no-op at `H_in = D_in^2` (mean diagonal exactly 1.0), so the
bit-exact identity unit test is unaffected.

**This is worth stating plainly: reporting "Change 2 diverges" would have been
true and completely misleading.** The divergence is an interaction with an
undocumented invariant of the released solver, not a property of using the full
covariance.

## Unit test — reduction to upstream at `H_in = D_in^2`

Required: reproduce the original iterates to 1e-6. Achieved **bit-exactly**
(rel = 0.000e+00) on all five layer shapes that occur in Llama-3.2-1B —
2048x2048, 2048x512, 512x2048, 8192x2048, 2048x8192 — for `W_final`, `A`, `B`,
`scale_pre` and `scale_post`.

Two details were needed and are easy to miss:

* `H @ X` must run with **TF32 off**. TF32 rounds its *inputs* to a 10-bit
  mantissa, so a 1e-7 difference in `H~` becomes ~1e-4 in the product, and the
  `sign()` projection amplifies that to O(1) within 40 iterations.
* `H~` must be formed by **division**, `H_in / (d (x) d)`, not by multiplying with
  a rounded reciprocal. Division gives exactly the identity; the reciprocal form
  gives identity-to-1e-7, which the chaotic iteration then blows up.
* `H_in = D_in^2` must be built as `diag(norm_i * norm_i)`, not `diag(i_norm)` —
  these differ by float rounding, with the same consequence.

A guard test confirms the change is not a silent no-op: with a genuinely
non-diagonal `H`, the result differs from stock by rel 0.91 and lowers the
objective it targets.

## Results (seed 0)

| arm | block | `pre` | `post` | PPL (1 block quantised) | KL-to-FP |
|---|---|---|---|---|---|
| baseline | 0 | 1.41595e-1 | **1.23275e-1** | 11.1307 | 0.13951 |
| **B** | 0 | **1.38752e-1** | **1.27651e-1** | 11.2520 | 0.15162 |
| baseline | 8 | 1.40450e-2 | **1.14722e-2** | 10.3604 | 0.06404 |
| **B** | 8 | **1.27958e-2** | **1.14865e-2** | 10.3770 | 0.06548 |

Paired against baseline at the same seed:

| | `d_pre` | `d_post` | verdict |
|---|---|---|---|
| block 0 | −2.787e-3 (**−1.97%**) | +4.248e-3 (**+3.44%**) | lose |
| block 8 | −1.249e-3 (**−8.89%**) | +1.436e-5 (**+0.13%**) | inside the 0.25% noise floor |

Determinism floor: 0.31% (block 0), 0.25% (block 8). Block 8's `post` difference
is inside it; block 0's is more than ten times it, in the wrong direction. The
gate requires beating baseline on both blocks, so seeds 1-2 were not spent.

### Per-layer `post`

Unlike Test A, Test B is not uniformly worse — it wins on three of seven layers
in block 0 and three in block 8, all of them MLP:

| layer | blk0 base | blk0 B | blk8 base | blk8 B |
|---|---|---|---|---|
| `q_proj` | **5.802e-4** | 7.774e-4 | **2.833e-4** | 3.409e-4 |
| `v_proj` | **1.1216e-2** | 1.2301e-2 | **1.3210e-3** | 1.3893e-3 |
| `o_proj` | **1.8830e-2** | 1.9014e-2 | **2.2511e-3** | 2.3336e-3 |
| `k_proj` | **1.4716e-2** | 1.6061e-2 | **1.7904e-3** | 1.9442e-3 |
| `gate_proj` | **5.0662e-2** | 5.0502e-2 | **3.2876e-3** | 3.3424e-3 |
| `up_proj` | 7.4951e-2 | **7.3870e-2** | 7.7349e-3 | **7.3317e-3** |
| `down_proj` (= block) | **1.2327e-1** | 1.2765e-1 | **1.1472e-2** | 1.1487e-2 |

`up_proj` improves by 1.4% (block 0) and 5.2% (block 8), and `gate_proj` is level.
The loss is concentrated in attention and in `down_proj`, which *is* the block
error. So the full covariance helps precisely where the preconditioner's diagonal
approximation is worst — the wide MLP projections — and hurts on the layers whose
inputs are closer to isotropic.

## Gate outcome

Same as Test A: Step 3 erases the `pre` gap. Test B buys a larger `pre`
improvement than A (−8.9% on block 8) and still converts none of it. Change 2 is
not load-bearing at 1 bpw on this model. Its one durable contribution is
negative and methodological — the `rho` scaling invariant documented above.
