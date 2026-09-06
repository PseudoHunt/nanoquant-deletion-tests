# Test C — SBD hybrid falsification (Change 7)

**Status: stopped at the gate. The three arms cannot be built at equal total bits.**

Model: Llama-3.2-1B (`unsloth/Llama-3.2-1B`, ungated mirror — see `repro.md`).
Config: NanoQuant `bits=1.0`, their rank accounting. No training, no ADMM: this
section is pure spectral analysis of the FP weights, so it is seed-independent.

## Method

Per linear layer, SVD of the FP weight `W` (`d_out x d_in`), then a
Gavish–Donoho-style fit of the Marchenko–Pastur noise scale from the median
singular value:

* `beta = min(d_out,d_in)/max(d_out,d_in)`
* `mu_beta` = median of the MP singular-value law at `beta`, obtained by
  numerically inverting the MP CDF (midpoint quadrature; the `1/lambda`
  singularity at `beta=1` is avoided by integrating on cell midpoints)
* `sigma_hat = s_median / (sqrt(max(d_out,d_in)) * mu_beta)`
* `edge = sigma_hat * (sqrt(d_out) + sqrt(d_in))`
* `k_RMT = #{ s_i > 1.05 * edge }`

**Estimator validation.** On synthetic iid Gaussian matrices at
`beta in {1, 0.25, 0.0625}` the estimator recovers `sigma = 1` to within 0.4%,
the empirical bulk edge matches `1+sqrt(beta)` to <1.5%, and `k_RMT = 0` on pure
noise in every case. The machinery is correct; the numbers below are a property
of the weights, not of the fit.

## `k_RMT` per layer (16 blocks, min / median / max across blocks)

| layer | shape | transposed by NQ | rank budget `r` | `k_RMT` min/med/max | `16*k_RMT` (med) | cost vs `r` | blocks where `16*k_RMT < r` | energy in top-`k_RMT` |
|---|---|---|---|---|---|---|---|---|
| `self_attn.q_proj` | 2048x2048 | no | 992 | 322 / 405 / 452 | 6480 | 6.54x | 0/16 | 0.81 |
| `self_attn.k_proj` | 512x2048 | yes | 384 | 89 / 118 / 155 | 1888 | 4.94x | 0/16 | 0.65 |
| `self_attn.v_proj` | 512x2048 | yes | 384 | 1 / 26 / 93 | 416 | 1.10x | 7/16 | 0.14 |
| `self_attn.o_proj` | 2048x2048 | no | 992 | 155 / 285 / 465 | 4560 | 4.60x | 0/16 | 0.54 |
| `mlp.gate_proj` | 8192x2048 | no | 1600 | 84 / 173 / 263 | 2768 | 1.74x | 3/16 | 0.27 |
| `mlp.up_proj` | 8192x2048 | no | 1600 | 17 / 69 / 194 | 1104 | 0.69x | 12/16 | 0.10 |
| `mlp.down_proj` | 2048x8192 | yes | 1600 | 45 / 93 / 149 | 1488 | 0.93x | 10/16 | 0.15 |

Across all 112 factorised layers: `k_RMT` quartiles = **1 / 77 / 126 / 263 / 465**
(min/Q1/median/Q3/max), mean 168.3.

## Why the test stops

The task's stop rule was written for the opposite failure: *"if `k_RMT` comes out
at 0–2 for most layers, Test C's residual is too small to matter."* That rule
does **not** fire — only **1 of 112** layers has `k_RMT <= 2`. What fires instead
is the reverse infeasibility.

At the stated equal-bits accounting (an FP16 rank-`k` residual costs
`16k(d_out+d_in)` bits, i.e. `16k` binary ranks):

| quantity | bpw |
|---|---|
| NanoQuant binary factors (`r*(d_in+d_out)`) | 0.9741 |
| + `scale_pre`/`scale_post` | **0.9857** (their ~1.0 bpw point) |
| FP16 residual at `k = k_RMT`, **alone** | **1.7609** |

The RMT spike subspace in FP16 costs **1.81x the entire NanoQuant bit budget**
before a single binary rank is spent. Consequently:

* `spike-fp` requires binary rank `r - 16*k_RMT`, which is **negative in 80 of
  112 layers** (all 16 `q_proj`, all 16 `k_proj`, all 16 `o_proj`, and most
  `gate_proj`). Median `16*k_RMT / r` = **2.08x**.
* Only **10 of 112** layers leave even half the budget for the binary factors.
* `err-fp` has exactly the same cost and is equally unconstructible.

There is no equal-total-bits three-arm comparison to run at ~1.0 bpw. Any
constructible variant would have to cut `k` far below `k_RMT` — the budget
affords a median of **k = 16** at 25% of the bits, versus `k_RMT` = 126 — at
which point the split is no longer at the bulk edge and the arm is no longer the
RMT-motivated hybrid the test was designed to falsify.

## What this means

The premise behind the SBD hybrid is that a trained weight matrix is a *small*
spike subspace plus an MP noise bulk, so that protecting the spikes in FP16 is
cheap. On Llama-3.2-1B that premise is false in both halves:

1. **The spike count is not small.** Hundreds of singular values sit above the
   bulk edge (median 126, up to 465).
2. **The MP null model does not fit.** For `q_proj` the "spikes" carry a median
   **81%** of the Frobenius energy — a spectrum that is 80% signal above the
   noise edge is not spike-plus-bulk, it is heavy-tailed. This is consistent
   with the heavy-tailed self-regularisation literature on trained networks; the
   MP edge is not a meaningful cut for these matrices.

So the honest reading is not "the spike subspace is too small to matter" but
"there is no cheap spike subspace to protect at all". At sub-1-bit budgets an
FP16 low-rank residual of *any* RMT-justified size dominates the cost of the
thing it is supposed to help. `spike-fp` cannot beat `more-rank` at equal bits
because `spike-fp` cannot be built at equal bits.

**Verdict rule outcome:** `spike-fp` does not beat both others beyond seed
spread, because it is not constructible. Test C does not count in favour of the
hybrid.

## Artifacts

* `artifacts/rmt_llama32_1b.json` — full per-layer record (spectra head,
  `sigma_hat`, `edge`, `k_RMT`, energy fractions)
* `nqx/rmt.py` — estimator + validation
