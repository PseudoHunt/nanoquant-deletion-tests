# Test A — sequential binarisation with refit and rank-wise compensation (Change 1)

**Verdict: not promoted. Test A improves `pre` on both blocks and loses on `post`
on both blocks, by 4.0% and 4.5% — 13-18x the measured determinism floor.**

## What was implemented

After `factorize_admm_nanoquant` returns, the joint `sign()` of both factors is
replaced by:

1. **`V_b = sign(B_z)`**, keeping NanoQuant's column scale `s2 = scale_pre` and
   the per-rank magnitude of `B`'s rank-1 field
   (`s_mid_B[k] = mean_j |B[k,j]| / mean(.)`), forming
   `V~ = diag(s_mid_B) V_b diag(s2)` — exactly the product their kernel computes
   as `(x * scale_pre) @ V^T` then `* scale_mid`.
   Unit-tested: `V~` reproduces their real `B` factor to **1.5e-3**, which is
   bf16 round-off — `|B|` is exactly rank-1 in exact arithmetic, so nothing is
   lost by writing it in kernel form.
2. **Closed-form refit** `A* = W H_in V~^T (V~ H_in V~^T + dI)^-1`, one `r x r`
   Cholesky solve, `d = 0.01 * mean(diag(V~ H V~^T))`.
   Unit-tested: `A*` strictly lowers `tr((W - A V~) H_in (.)^T)` versus `A_z`
   (2.744e4 -> 2.124e4, a 22.6% reduction, on the synthetic case; 10x on real
   layers).
3. **Column-wise binarisation with GPTQ-style compensation** against
   `H_r = V~ H_in V~^T + dI`, columns ordered by `diag(H_r)` descending,
   `u_k = s1 * s3[k] * sign(a_k)`, error propagated through the standard
   Cholesky-of-inverse form in blocks of 128. `s1` = mean-abs of `A*` rows,
   `s3` = per-column mean-abs normalised to mean 1, both frozen for the loop.
   Unit-tested: beats plain `sign()` at the same scales (1.2% on the synthetic
   case).
4. **Change 4 (ALS scale refit): deferred, and the mechanism below says it would
   hurt.** See the verdict discussion.

Step 3 then runs unchanged.

### Choices that were not pinned by the brief

* **`s3[k]` is the mean-abs column norm normalised to mean 1**, not the raw L2
  column norm. With a raw L2 norm `s1_i * s3_k` overshoots `|A*[i,k]|` by
  `sqrt(d_out)`, so some normalisation is required for the expression to be a
  magnitude estimate at all; after normalising, L2 and mean-abs agree to a few
  percent.
* **`scale_mid` is now populated** (it was `None` on NanoQuant's path, though
  `NanoQuantLinear` and both CUDA kernels already support it). This costs
  `16r` bits per layer: **+0.0020 bpw, so 0.9877 against the baseline's 0.9857**.
  Test A is therefore very slightly *more* expensive as well as worse.
* **Latents for Step 3** are `|A*|` rescaled to the mean magnitude of their own
  `A_latent`, carrying our sign decisions. Magnitude matters here because STE
  sign flips depend on how far a latent sits from zero relative to the learning
  rate; matching the baseline's global magnitude keeps the flip dynamics
  comparable rather than confounding the comparison.

## Harness

Single-block harness: blocks 0 and 8 of Llama-3.2-1B, everything else FP16.
FP inputs and FP outputs for 128 calibration + 32 held-out sequences cached once
(wikitext2 *validation* for the held-out split, disjoint from the calibration
source and the PPL source). `gamma = 0.2` per Appendix C. Seed 0.

## Results (seed 0)

| arm | block | `pre` | `post` | PPL (1 block quantised) | KL-to-FP |
|---|---|---|---|---|---|
| baseline | 0 | 1.41595e-1 | **1.23275e-1** | 11.1307 | 0.13951 |
| **A** | 0 | **1.39482e-1** | **1.28324e-1** | 11.2545 | 0.15119 |
| baseline | 8 | 1.40450e-2 | **1.14722e-2** | 10.3604 | 0.06404 |
| **A** | 8 | **1.36143e-2** | **1.19923e-2** | 10.3903 | 0.06713 |

Paired against baseline at the same seed:

| | `d_pre` | `d_post` | verdict |
|---|---|---|---|
| block 0 | −2.057e-3 (**−1.45%**) | +4.921e-3 (**+3.99%**) | lose |
| block 8 | −4.307e-4 (**−3.07%**) | +5.202e-4 (**+4.53%**) | lose |

**Determinism floor** (identical config, identical seed, fresh process):
**0.31%** on block 0 and **0.25%** on block 8. The `post` regression is more than
an order of magnitude larger, so seeds 1-2 were not spent.

### Per-layer `post` block error

Test A is worse on **every one of the seven layers in both blocks**, so this is
not an MLP-specific effect:

| layer | blk0 base | blk0 A | blk8 base | blk8 A |
|---|---|---|---|---|
| `q_proj` | 5.802e-4 | 9.286e-4 | 2.833e-4 | 3.479e-4 |
| `v_proj` | 1.1216e-2 | 1.2108e-2 | 1.3210e-3 | 1.4874e-3 |
| `o_proj` | 1.8830e-2 | 2.0845e-2 | 2.2511e-3 | 2.2961e-3 |
| `k_proj` | 1.4716e-2 | 1.5892e-2 | 1.7904e-3 | 1.9212e-3 |
| `gate_proj` | 5.0662e-2 | 5.3543e-2 | 3.2876e-3 | 3.5742e-3 |
| `up_proj` | 7.4951e-2 | 7.8747e-2 | 7.7349e-3 | 8.1026e-3 |
| `down_proj` (= block) | 1.2327e-1 | 1.2832e-1 | 1.1472e-2 | 1.1992e-2 |

## Why it loses — the mechanism

The H-weighted layer objective `J = tr((W-W^)H(W-W^)^T)/tr(W H W^T)`, logged at
both measurement points, gives a clean answer.

| block 0 layer | base `J` pre -> post | A `J` pre -> post |
|---|---|---|
| `v_proj` | 0.2640 -> **0.2339** (down) | **0.2235** -> 0.2301 (up) |
| `up_proj` | 0.2646 -> 0.2692 (up) | **0.2350** -> 0.2814 (up) |
| `down_proj` | 0.1961 -> **0.1843** (down) | **0.1857** -> 0.1894 (up) |
| `gate_proj` | 0.1933 -> 0.2027 (up) | **0.1754** -> 0.2162 (up) |

Change 1 does what it was designed to do: at `pre` it lowers `J` in six of seven
layers (`v_proj` 0.2235 against 0.2640). But Step 3 does **not** descend `J` — it
descends *block output* error, a different objective. From the baseline's
starting point the two happen to agree often enough that `J` falls; from Test A's
starting point, which is already near a local optimum of `J`, Step 3 can only
climb `J`, and it arrives at a worse block error than it would have reached from
the sloppier ADMM point.

So the better initialisation is worse **because** it is better on the wrong
objective. The refit spends the factorisation's freedom on layer-wise weight
fidelity, which Step 3 was going to trade away anyway — and having spent it,
Step 3 has less room left to trade.

This also disposes of **Change 4**: an ALS refit of `(s1, s2, s3)` minimises the
same layer objective still further, i.e. pushes harder in exactly the direction
that just lost. It was deferred under the agreed rule (only run if A works) and
the mechanism predicts it would make A worse, not better.

## Gate outcome

The brief's stop rule — *"if Step 3 erases the `pre` gap, those changes are not
load-bearing"* — is satisfied in a stronger form than anticipated. Step 3 does
not merely erase Test A's `pre` gap; it **inverts** it. Change 1 is not
load-bearing, and at these ranks it is actively harmful.
