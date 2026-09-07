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

Started from a random rotation instead, the alignment does what ITQ is supposed
to do — it descends, hard — but it descends *back toward* the identity's
neighbourhood without reaching it (see the random-init rows). That is the same
message as Arm 1 from the other side.

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

The line search settles what that means. Accumulating the functional gradient
over the whole calibration set at `R = I` and walking the *true* calibration loss
along the unit steepest-descent ray gives a curve that rises monotonically out of
`t = 0`; random tangent directions rise the same way. So this is not an
over-stepping artefact and not a sign error in the STE path (Stage 0 already
proved the gradient is finite, non-zero, and reaches only the Cayley parameters):
**`R = I` sits at a local minimum of the binary functional error within the gauge
class**, and the steepest-STE-descent direction points out of it, not along it.

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
