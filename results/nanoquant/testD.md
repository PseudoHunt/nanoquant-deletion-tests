# Test D — mirror descent replacing STE in Step 3 (block 0)

**Verdict: no mirror cell clears the floor. The best of the eight, `H` at
η = 3e-2, is +19.75% worse than the STE control on `post` block error — 64x the
0.31% determinism floor. Four of eight cells diverge outright.**

## Setup, and two asserts that do not hold

Block 0 of Llama-3.2-1B, γ = 0.2, seed 0, existing harness (cached FP inputs and
targets, 32 held-out sequences from wikitext2 validation). Step 1
(`tune_nonfact`) and Step 2 (ADMM) were run for all seven layers in the released
order, nothing finalised, and the block banked with every `NanoQuantLinear`
holding live latent proxies. **Cache round-trip is bit-identical**
(0.143661403 in memory, 0.143661403 reloaded), so the state is clean.

Both asserted numbers in the brief are structural mismatches, not cache errors:

| assert | expected | measured | why |
|---|---|---|---|
| post-ADMM `pre` block error | 0.1416 | **0.143661** | — |
| STE control `post` | 0.1233 | **0.114852** | — |

The released pipeline does **not** produce a state in which all seven layers hold
live latents. It interleaves: for each layer in turn, Step 1 → ADMM → Step 3 →
`finalize()`, so by the time `down_proj` is factorised the first six are already
refined and hardened, their latents deleted. The banked 0.1416 / 0.1233 pair
comes from that trajectory. A snapshot with all seven proxies alive — the state
the brief asks for, and the only one from which every variant can start from an
identical point — is a strictly earlier and slightly worse state (0.143661,
+1.5%), and a single joint Step 3 pass over it is a different schedule from
seven sequential ones.

That restructuring was required by the brief's own constraints: a shared starting
state for all nine cells is incompatible with interleaving, and the step budget
(8 epochs x 128 samples, batch 1 = 1024 optimizer steps) was held fixed as
instructed. **All nine cells below are internally paired** — same banked state,
same 1024 steps, same loss, same scale/bias optimizer — so the mirror-vs-STE
comparison is exact. Only the comparison to the banked 0.1233 is not.

### Incidental result worth following up

The STE control on this schedule reaches **0.114852**, against **0.123275** for
the released interleaved schedule — **6.8% better block error using 1024 Step-3
steps instead of 7 x 1024**, and from a worse starting point (0.143661 vs
0.141595). One seed, one block, not a controlled comparison of schedules, so it
is an observation and not a claim. But deferring all of Step 3 to a single joint
pass over the fully-quantised block appears to be both cheaper and better than
refining layer-by-layer with the downstream layers still FP.

## The nine cells

`post` measured after hard `sign()` was written into `V`/`U`, `_binarized` set,
latents deleted and the relaxed forward disabled. No number here is taken on a
relaxed forward.

| cell | variant | η | `post` block err | vs STE control | sign flips | Step 3 |
|---|---|---|---|---|---|---|
| `ctrl_ste` | STE + AdamW (released) | 1e-5 | **0.114852** | — | **1.0321%** | 25.1 s |
| `R_3e-3` | relaxed, β 1→10 | 3e-3 | 1.72192 | +1399% | 0.0134% | 22.8 s |
| `R_1e-2` | relaxed, β 1→10 | 1e-2 | 1.59317 | +1287% | 0.0272% | 22.8 s |
| `R_3e-2` | relaxed, β 1→10 | 3e-2 | 6.75e+12 | diverged | 50.2902% | 22.6 s |
| `R_1e-1` | relaxed, β 1→10 | 1e-1 | 4.99e+14 | diverged | 49.9500% | 22.6 s |
| `H_3e-3` | hard, β = 1 | 3e-3 | 0.137761 | +19.95% | 0.0036% | 25.2 s |
| `H_1e-2` | hard, β = 1 | 1e-2 | 0.137764 | +19.95% | 0.0051% | 25.1 s |
| **`H_3e-2`** | hard, β = 1 | 3e-2 | **0.137533** | **+19.75%** | 0.0107% | 25.1 s |
| `H_1e-1` | hard, β = 1 | 1e-1 | 7.71e+13 | diverged | 49.9549% | 24.9 s |

All cells start from `pre` = 0.143661. Determinism floor: **0.31%**.

### Sign flips from the ADMM init, per layer (V / U, %)

| layer | `ctrl_ste` | `H_3e-2` | `R_1e-2` |
|---|---|---|---|
| `q_proj` | 0.3101 / 0.4783 | 0.0027 / 0.0056 | 0.0032 / 0.0134 |
| `v_proj` | 0.7678 / 0.4247 | 0.0184 / 0.0270 | 0.0376 / 0.0814 |
| `o_proj` | 0.5364 / 0.6666 | 0.0157 / 0.0091 | 0.0270 / 0.0079 |
| `k_proj` | 0.1567 / 0.0860 | 0.0118 / 0.0239 | 0.0205 / 0.1592 |
| `gate_proj` | 0.0154 / **3.4830** | 0.0002 / 0.0130 | 0.0003 / 0.0117 |
| `up_proj` | 0.0339 / **3.9446** | 0.0002 / 0.0143 | 0.0002 / 0.0134 |
| `down_proj` | **2.5589** / 0.9877 | 0.0078 / 0.0006 | 0.0054 / 0.0001 |

### Per-layer `J` (H-weighted layer objective) and relative Frobenius after hardening

| layer | `J` pre | `J` post, STE | `J` post, `H_3e-2` | fro post, STE | fro post, `H_3e-2` |
|---|---|---|---|---|---|
| `q_proj` | 7.735e-3 | 9.275e-3 | 7.849e-3 | 0.1941 | 0.1962 |
| `v_proj` | 2.637e-1 | 2.557e-1 | 2.633e-1 | 0.3226 | 0.3196 |
| `o_proj` | 8.212e-2 | 8.653e-2 | 8.380e-2 | 0.2656 | 0.2623 |
| `k_proj` | 1.913e-2 | 1.923e-2 | 2.144e-2 | 0.2043 | 0.2057 |
| `gate_proj` | 1.914e-1 | 2.211e-1 | 1.938e-1 | 0.3627 | 0.3342 |
| `up_proj` | 2.508e-1 | 2.767e-1 | 2.505e-1 | 0.3802 | 0.3443 |
| `down_proj` | 1.662e-1 | 1.691e-1 | 1.678e-1 | 0.3863 | 0.3504 |

STE raises `J` in five of seven layers while cutting block error by 20%; the same
trade seen in Tests A and B. `H_3e-2` leaves `J` almost unchanged because it
leaves almost everything unchanged — and its *Frobenius* error is lower than
STE's in the three MLP layers while its block error is 20% worse, which is the
same point again from the other side.

## Why the mirror step fails — measured, not inferred

### Variant H: no η produces a usable flip rate

The mirror update is plain SGD on Θ with the raw `dL/dU`. Measured on one batch
at the banked state, `|dL/dU|` within a single layer spans three orders of
magnitude:

| layer (V factor) | median | p99 | max | max/median |
|---|---|---|---|---|
| `q_proj` | 4.60e-4 | 7.63e-3 | 6.40e-2 | 139x |
| `o_proj` | 5.53e-4 | 6.90e-3 | 1.52e-1 | **275x** |
| `down_proj` | 3.82e-5 | 3.00e-4 | 2.00e-2 | **525x** |

At η = 3e-2 the per-step relative displacement `η|g|/|Θ|` has median 5e-6 to
2.5e-4 — over 1024 noisy steps that moves essentially nothing, and the measured
flip rate is 0.0107% against STE's 1.0321%, **100x fewer**. Raising η by 3.3x to
1e-1 does not scale the flip rate up smoothly; it goes to **49.95%**, i.e. the
signs are randomised and the block explodes to 7.7e13. The transition is a cliff,
not a gradient: once enough signs flip the block output degrades, the gradients
grow, and it runs away.

That is exactly what AdamW's second-moment normalisation prevents. It gives every
latent a step of order `lr` regardless of its gradient magnitude, which is what
produces a controlled ~1% flip rate concentrated where it matters (`gate_proj` U
3.48%, `up_proj` U 3.94%, `down_proj` V 2.56%). A scalar step size on the raw
gradient cannot reproduce that, because the gradient's dynamic range within one
layer is larger than the window between "moves nothing" and "moves everything".

### Variant R: the prescribed initialisation does not start at the ADMM point

`Θ = atanh(clamp(proxy/max|proxy|, ±0.99))` gives `tanh(1·Θ) = proxy/max|proxy|`
exactly — the **normalised real-valued proxy, not the binary**. Measured mean
`|tanh(βΘ)|` at the banked state:

| layer | β = 1 (V) | β = 10 (V) | β = 1 (U) | β = 10 (U) |
|---|---|---|---|---|
| `q_proj` | 0.3098 | 0.9350 | 0.1819 | 0.8512 |
| `gate_proj` | 0.4002 | 0.9917 | 0.1441 | 0.8019 |
| `down_proj` | 0.2351 | 0.9079 | 0.1575 | 0.8820 |

So variant R begins training a model whose factors have magnitude 0.14–0.41
instead of 1. Its first-epoch loss is **1.69e-4 against the control's 1.98e-5**,
8.5x worse, and it spends its whole budget climbing back to 2.60e-5 — still 1.56x
above the control's final 1.67e-5. It never reaches ±1 either: at the final
β = 10 the mean is 0.80–0.99 with a long lower tail. The hard `sign()` at the end
is therefore a jump the training never saw, and it takes `post` to 1.59 — more
than **ten times worse than the un-tuned starting point of 0.1437**.

The β anneal 1→10 over 8 epochs is not enough to close that gap; and it cannot
be fixed by raising η, because η ≥ 3e-2 diverges during training (`R_3e-2` block
loss reaches 1.15e+11 by epoch 4).

## Verdict

Stop rule: the best mirror cell must beat the STE control by more than 0.31%.
Best mirror cell is `H_3e-2` at **+19.75%** — worse, by 64x the floor. It also
loses to the banked interleaved baseline (0.123275) by 11.6%. **No cell clears
the floor; reporting the full table and stopping.** Block 8 and extra seeds not
run, as instructed.

The negative result is specific rather than generic. Mirror descent is not being
beaten on its optimisation geometry; it is beaten because (a) a scalar step size
on the raw gradient cannot span the gradient's within-layer dynamic range, which
is what Adam is doing for STE, and (b) the prescribed `atanh` initialisation puts
the relaxed forward at the real-valued proxy rather than at the binary point, so
variant R optimises a model that its own hardening then discards. A mirror
variant with per-parameter step normalisation, or an initialisation that makes
`tanh(βΘ)` start near `sign(proxy)`, would be a different and untested proposition.

## Artifacts

* `artifacts/testD/block0_postadmm.pt` — banked post-ADMM block-0 state
* `artifacts/testD/runs/*.json` — the nine cells, per-layer `J` and flip fractions
* `nqx/testD.py`, `nqx/testD_diag.py`, `testD.sh`
