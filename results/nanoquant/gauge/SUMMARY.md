# Gauge-NQ — summary

**Question.** NanoQuant's continuous low-rank factorisation `W_cont = U V^T` is
non-identifiable under orthogonal transformations of its latent rank space:
`(U R)(V R)^T = U V^T` for any `R^T R = I`. Binary projection breaks that
symmetry — `sign(U R) sign(V R)^T` generally differs from `sign(U) sign(V)^T`.
Does choosing `R` by geometry, or by downstream function error, expose a better
binary solution at identical rank, bpw, ADMM, Step-3 optimiser, data and deployed
representation?

**Answer: no, and for a specific reason.** The gauge freedom is real and very
large, and NanoQuant's ADMM already sits at the bottom of it.

Scope: block 0 of Llama-3.2-1B, `mlp.down_proj`, NanoQuant at bits = 1.0
(0.9857 bpw), gamma = 0.2, 128 wikitext2 calibration sequences at seed 0, 32
held-out sequences from wikitext2 validation. Stage 1 did not pass its gate, so
Stages 2-4 (full MLP, attention, block 8, Transformer function gauges, seeds 1-2,
full model) were not entered — the same promotion rule the rest of this
repository uses. Detail and every table: `stage1_down.md`.

## Headline table

| arm | E_gauge (pre-Step-3, held-out) | E_final (held-out) | Δ vs 0a |
|---|---|---|---|
| post-ADMM start `E_ADMM` | 0.143766 | — | — |
| **0a** baseline, `R = I` | 0.143766 | **0.115388** | — |
| 0a duplicate (replay floor) | 0.143766 | 0.115165 | −0.19% |
| 0b_100 extra STE x100 | 0.133687 *(−7.0%)* | 0.116673 | **+1.11%** |
| 0b_200 extra STE x200 | 0.132425 *(−7.9%)* | 0.118267 | **+2.50%** |
| 2 ITQ, `R0 = I` (refreshed / frozen) | 0.143791 *(+0.02%)* | 0.115075 / 0.115109 | −0.27% / −0.24% |
| 2 ITQ, `R0` random | 0.341970 | 0.141594 | +22.7% |
| 3 functional gauge, lr 3e-3 @100 / @200 | 0.230205 / 0.289858 | 0.130715 / 0.140818 | +13.3% / +22.0% |
| 3 functional gauge, best swept lr (3e-5 @200) | 0.145652 | 0.115172 | −0.19% |
| Arm 1, 16 Haar gauges (no Step 3) | 0.358 – 0.367, median 0.361 | — | — |

Gate: δ = 0.0031. A gauge passes only if it beats **both** 0a and its matched 0b
by more than δ. **Every cell fails the 0a leg.** The cells that are numerically
below 0a (−0.19% to −0.27%) are inside the 0.19% same-state replay floor measured
here and below the 0.31% floor this repository established.

## The eight questions

**1. Does binary error vary materially across the latent gauge class?**
**Yes, enormously.** Sixteen Haar SO(32) block-rotations leave `U R (V R)^T`
unchanged to a relative 8.5e-7 and move the held-out block error from 0.14377 to
0.358–0.367 — a factor of **2.5** — flipping ~50% of both U and V signs. The
gauge is not a nuisance parameter.

**2. Does Joint-ITQ-inspired geometric alignment improve final NanoQuant error?**
**No.** From `R = I` it is a fixed point: 20 rounds move `E_gauge` by +0.018%.
Structurally so — with no `scale_mid`, NanoQuant's magnitude field is rank-1, so
the Procrustes target `M ⊙ sign(X0 R)` agrees with `X0 R` in sign by
construction. From a *random* init the alignment behaves normally, descending
0.3614 → 0.3420 over 20 rounds, and still lands 2.4x above the identity
(`E_final` +22.7%). The geometric criterion converges to a basin unrelated to the
one the ADMM found.

**3. Does functional gauge optimisation improve final error?**
**No, at any step size.** At the pre-registered lr = 3e-3 it diverges outright
(calibration `L_func` 0.1251 → 0.2722; `E_final` +22.0%). Swept over
{3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5} — selected on the **calibration** objective
only, never held-out — *every* learning rate raises the calibration objective
above the identity's 0.125113 from the first probe; smaller rates simply move
less. The best point reached anywhere in the sweep is 0.125104, a 0.007%
improvement.

A full-batch line search settles why. The functional gradient at `R = I`,
accumulated over all 128 calibration sequences and walked along its own unit
steepest-descent ray, gives a curve flat to four decimals out to a parameter
displacement of t = 0.1 and monotonically rising after; the best point on the ray
is **0.011% below** the identity, twenty times under the replay floor. Random
tangent directions have the same flat-then-rising shape — and at every t where
anything moves, the *gradient* direction is **worse** than a random one
(t = 1: 0.1313 vs 0.1255; t = 3: 0.1754 vs 0.1326). The STE gradient through the
hard sign is not merely uninformative about the gauge, it is anti-correlated with
the true objective at any usable step size.

**4. Does functional gauge beat the step/data-matched extra-STE control?**
On that leg only, and vacuously: the extra-STE controls end up *worse* than
doing nothing, so beating them is not evidence of anything. No cell passes the
0a leg, so no gauge contribution can be claimed.

**5. Does the gain appear before common Step 3, or does gauge provide a better
basin?** **Neither — there is no gain at either point.** The only intervention
that improves the pre-Step-3 point is the extra-STE control, by 7.0% / 7.9%, and
it gives all of that back and more during Step 3 (+1.11% / +2.50% final). Every
gauge that moves at all makes the pre-Step-3 point worse.

**6. How many U/V signs change?**
Random Haar gauge ~50.1% / 50.1%. ITQ from `R = I` 0.00% / 0.00%; from random
50.0% / 50.0%. Functional gauge at lr 3e-3 7.62% / 11.16% (@100) and 21.44% / 23.74%
(@200); at lr 3e-5, 0.00% / 0.31%. For scale, the common Step 3 itself flips
0.98% of `down_proj`'s U signs and 2.55% of its V signs from the baseline start.

**7. Does the effect replicate on block 8?** **Not run.** Stage 1 did not pass its
gate; block-8 caches were built but not spent.

**8. Does the best latent gauge improve the full model at equal bpw?**
**Not run**, same reason. No configuration qualified for promotion.

## What this adds to the repository's existing result

`SUMMARY.md` for the deletion tests concluded that at 1 bpw *initialisation
quality measured on layer-wise reconstruction is not the binding constraint —
Step 3 is*. Gauge-NQ sharpens that in two ways.

First, the **0b control is a stronger version of Tests A and B**. A and B changed
the initialisation *rule*; 0b changes nothing but runs NanoQuant's own update on
its own objective for 100 or 200 extra steps. It improves the starting point by
7.9% and finishes **2.5% worse**, and the harm grows monotonically with the number
of extra steps. So it is not that better initialisations happen not to help — it
is that pushing the starting point down *spends* something Step 3 needs. The
sign-flip columns say what: from the extra-STE starts, Step 3 flips **more**
`down_proj` signs (U 1.29% / 1.59% vs 0.98%) and still arrives higher. The easy
flips have already been taken and Step 3 is left a worse-conditioned remainder.
This is consistent with Test D2's decomposition, where four fifths of Step 3's
gain came from flipping ~1% of the signs.

Second, it **closes off the gauge direction specifically**. This was the strongest
remaining "the initialisation is leaving something on the table" hypothesis,
because the symmetry is exact and the effect on the binary solution is provably
large. It is now measured: the effect is large (2.5x), and the ADMM point is at
the bottom of it. Every route tested — geometric alignment, function-error
descent at six step sizes, a full-batch line search, and 16 random probes —
either cannot move or moves uphill.

## Caveats

* One block, one seed, one layer, block size b = 32 (as specified). The gate was
  not passed, so nothing was promoted.
* The post-ADMM state was re-derived on a fresh VM. The ADMM is chaotic
  (NOTES.md#2), so it is an *independent draw*, not a replay: `E_ADMM` lands
  +0.073% and Arm 0a's `E_final` +0.47% from the previously cached D2 values.
  Every arm branches from the same reproduced state, so all within-table
  comparisons are controlled.
* The brief excluded a dense rank x rank rotation, so the gauge class probed here
  is block-diagonal SO(32). Arm 1 makes a dense *random* rotation a strictly
  worse bet, but a dense *optimised* one is not ruled out by anything measured.
* Arm 0b uses NanoQuant's own importance-weighted Step-3 loss (it is by
  definition the ordinary NanoQuant update) while Arm 3 uses the unweighted
  relative functional loss the brief specifies. The controls are step- and
  data-matched, not loss-matched, and not compute matched.

## Files

| file | contents |
|---|---|
| `stage1_down.md` | Stage 0 gates, all arms, the gate table, Arm 1/2/3 traces, per-layer post table, J pre/post, narrative |
| `repro_gate.md` | fresh-VM environment, the section 0.3 baseline reproduction, the one forced harness change |
| `stage0.json` | machine-readable Stage 0 assertions |
| `post_admm_meta.json` | the reproduced post-ADMM state: ranks, shapes, dtypes, absence of `scale_mid`, checksums |
| `runs/*.json` | every arm, every checkpoint, full traces |
| `linesearch.json` | full-batch gradient line search at `R = I` |
| `nqx/gauge/` | `core.py` (gauge map, Cayley, Q_NQ), `state.py` (Snapshots A/B), `harness.py` (materialisation, common Step 3, metrics), `arms.py`, `stage0.py`, `linesearch.py`, `report.py` |
