# NanoQuant deletion tests — summary

Model **Llama-3.2-1B** (Qwen2.5 is unsupported in the released code despite the
README — `model_type: qwen2` is in neither gate). NanoQuant at **bits = 1.0**,
their own rank accounting, 128 wikitext2 calibration samples, seed 0.
Repo `SamsungLabs/NanoQuant` @ `a9e0a43`, unmodified; every change is patched in
at run time from `nqx/`.

## Main table

`pre` = held-out block-output error after ADMM, before Step 3.
`post` = the same after Step 3. Both on a 32-sequence held-out split drawn from
wikitext2 **validation**, disjoint from calibration (train) and PPL (test).
Arms were run in the single-block harness (blocks 0 and 8, everything else FP16);
full-model wiki2/c4 was only spent on the baseline, because **no arm qualified
for promotion**.

| arm | blk | `pre` block err | `post` block err | wiki2 | c4 | n seeds |
|---|---|---|---|---|---|---|
| **FP16 reference** | — | — | — | **9.7499** | **14.018** | — |
| **baseline (stock NanoQuant, full model)** | all | — | — | **27.481** | **89.394** | 1 |
| paper Table 2, `L3-1` @ 1.00 bpw | all | — | — | *25.59* | *n/r* | — |
| baseline (harness) | 0 | 1.41595e-1 | **1.23275e-1** | 11.131 † | — | 3 |
| baseline (harness) | 8 | 1.40450e-2 | **1.14722e-2** | 10.360 † | — | 1 |
| determinism duplicate | 0 | 1.41555e-1 | 1.23014e-1 | 11.137 † | — | 1 |
| determinism duplicate | 8 | 1.40547e-2 | 1.15009e-2 | 10.354 † | — | 1 |
| **A** (Change 1) | 0 | 1.39482e-1 | 1.28324e-1 *(+3.99%)* | 11.255 † | — | 1 |
| **A** (Change 1) | 8 | 1.36143e-2 | 1.19923e-2 *(+4.53%)* | 10.390 † | — | 1 |
| **B** (Change 2) | 0 | 1.38752e-1 | 1.27651e-1 *(+3.44%)* | 11.252 † | — | 1 |
| **B** (Change 2) | 8 | 1.27958e-2 | 1.14865e-2 *(+0.13%)* | 10.377 † | — | 1 |
| **A+B** | 0 | 1.40246e-1 | 1.31925e-1 *(+6.91%)* | 11.486 † | — | 1 |
| **A+B** | 8 | 1.29534e-2 | 1.18644e-2 *(+3.42%)* | 10.389 † | — | 1 |
| **C** (`spike-fp` / `err-fp` / `more-rank`) | — | — | — | — | — | **0 — not constructible** |
| **D** control: STE, joint Step 3 | 0 | 1.43661e-1 | **1.14852e-1** | — | — | 1 |
| **D** best mirror (`hard`, η=3e-2) | 0 | 1.43661e-1 | 1.37533e-1 *(+19.75%)* | — | — | 1 |

† PPL with **only that one block** quantised, the harness's secondary metric.
Percentages are paired against the baseline at the same seed.

**Determinism floor: 0.31% (block 0), 0.25% (block 8)** — the pipeline is not
bit-deterministic, so this is the noise on any single cell. Every arm's `post`
regression on block 0 is more than ten times it, so seeds 1-2 were not spent
(gate: seeds 1-2 only for arms that beat baseline on both blocks at seed 0).

## Verdicts

| test | outcome |
|---|---|
| **Step 0 reproduction** | wiki2 **27.48** at 0.9857 bpw against the paper's **25.59** — a **+7.4%** gap. BF16 matches the paper to 0.01 (9.7499 vs 9.74), so model and eval protocol are right. Two config knobs in the released defaults contradict Appendix C; a corrected run is in flight. |
| **Test A** (Change 1) | **Fails.** `pre` −1.45% / −3.07%, `post` **+3.99% / +4.53%**. Worse on all 7 layers in both blocks. Also costs +0.0020 bpw (a `scale_mid` it has to populate). Change 4 deferred and predicted harmful. |
| **Test B** (Change 2) | **Fails.** `pre` −1.97% / **−8.89%**, `post` **+3.44%** / +0.13%. Wins on `up_proj` (−1.4% / −5.2%) and loses on attention and `down_proj`. |
| **A+B** | **Fails, worst of the three.** `post` +6.91% / +3.42%, despite a smaller `pre` gain than B alone. |
| **Test C** (Change 7) | **Not constructible.** Stopped at the gate. |
| **Test D** (mirror descent for Step 3) | **Fails.** Best of 8 mirror cells is **+19.75%** worse than the STE control, 64x the floor; 4 of 8 diverge. Measured cause: raw `dL/dU` spans 139-525x within a layer, so a scalar step size either moves nothing (0.01% sign flips vs STE's 1.03%) or randomises signs (50%); and `atanh(proxy/max)` starts the relaxed forward at the real-valued proxy, not the binary point. |

## The single result that matters

Both changes improve the pre-STE point, both by margins far outside the noise
floor, and **neither converts any of it**. The per-layer H-weighted objective
`J`, logged at both measurement points, says why:

* From the **baseline's** starting point, Step 3 *descends* `J` in the layers
  that dominate the block error (`v_proj` 0.2640 -> 0.2339, `down_proj`
  0.1961 -> 0.1843).
* From **Test A's** starting point, Step 3 *climbs* `J` (0.2235 -> 0.2301,
  0.1857 -> 0.1894) and arrives at a worse block error.

Step 3 optimises block output, not layer-wise weight reconstruction; from the
stock ADMM point the two objectives happen to agree often enough to help, but a
starting point already near a local optimum of the layer objective leaves Step 3
nothing to trade. The ordering across arms confirms it — A+B optimises the layer
objective hardest at init and is the worst at `post`.

So the answer to the framing question is negative in a specific and useful way:
**at 1 bpw on this model, initialisation quality measured on layer-wise
reconstruction is not the binding constraint. Step 3 is.** It removes 14% of the
block error, and it does so from wherever it starts.

## Incidental result from Test D, worth a follow-up

Deferring all of Step 3 to a **single joint pass** over the fully-quantised block
reached **0.114852** against the released interleaved schedule's **0.123275** —
**6.8% better block error using 1024 Step-3 steps instead of 7 x 1024**, and from
a worse starting point. One seed, one block, and the two schedules were not
otherwise controlled, so this is an observation rather than a claim. It is the
only thing measured across all four tests that moved `post` in the right
direction.

## Test C

`k_RMT` is **large**, not small: median 126, max 465, only 1 of 112 layers at
<= 2. The stated stop rule (residual too small to matter) does not fire; the
opposite infeasibility does. At the equal-bits accounting the brief specifies,
an FP16 residual at `k = k_RMT` costs **1.7609 bpw against a total budget of
0.9857 bpw — 1.81x the entire budget** — so `spike-fp` needs *negative* binary
rank in **80 of 112 layers**. There is no equal-bits three-arm comparison to run.
The spectra are heavy-tailed rather than MP-bulk-plus-spikes (the "spikes" carry
a median 81% of `q_proj`'s Frobenius energy), so the bulk edge is neither a
budget-feasible nor a well-defined split. See `testC.md`, including the
designed-but-not-run comparison at the RMT-implied ~2.7 bpw budget.

## Files

| file | contents |
|---|---|
| `repro.md` | Step 0: model selection, forced deviations, per-layer ADMM error, timings, paper comparison, byte-exact gate |
| `testA.md` | Change 1: implementation, unit tests, per-layer results, mechanism |
| `testB.md` | Change 2: the divergence and its fix, bit-exact reduction test, per-layer results |
| `testC.md` | RMT analysis, infeasibility, heavy-tail interpretation |
| `testD.md` | Mirror descent vs STE in Step 3: 9-cell table, sign-flip fractions, mechanism |
| `NOTES.md` | The `requires_grad` harness bug, ADMM chaos, config drift |
| `nqx/` | All instrumentation and variants; nothing in the upstream repo was edited |
