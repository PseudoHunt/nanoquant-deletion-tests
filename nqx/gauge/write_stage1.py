"""Assemble results/nanoquant/gauge/stage1_down.md: fixed methods header +
generated tables (report.py) + a hand-written narrative held in NARRATIVE.md."""
import json
import os

from . import report

OUT = "/home/work/exp/results/nanoquant/gauge"
GC = "/home/work/exp/artifacts/gauge"
NARR = os.path.join(os.path.dirname(__file__), "narrative_stage1.md")

HEADER = """# Gauge-NQ Stage 1 — latent gauge on `mlp.down_proj`, block 0

Does an *exactly equivalent* latent basis, chosen either geometrically or by
downstream function error, improve NanoQuant's final ~1-bit solution at identical
rank, bpw, ADMM, final optimiser, data and deployed representation?

Everything below is one block (block 0) of **Llama-3.2-1B** at NanoQuant
**bits = 1.0** (measured 0.9857 bpw), gamma = **0.2** (Appendix C), 128 wikitext2
calibration sequences at seed 0, 32 held-out sequences from wikitext2
**validation**. `NanoQuant @ a9e0a43`, unmodified; every change is patched in at
run time from `nqx/gauge/`.

## What the gauge actually is here

`admm_nq.factorize_admm_nanoquant` returns **no `scale_mid`**, and
`NanoQuantLinear._setup_path` only creates one when the factor results carry it.
The reproduced post-ADMM state therefore has **no rank-wise middle scale (no
s3)**, so brief section 2.1 applies unchanged and section 2.2's
fold -> rotate -> unfold is not instantiated. Nothing manufactures an s3.

    stored U_latent : [out,  rank]        math U : [out, rank]  ==  stored U_latent
    stored V_latent : [rank, in ]         math V : [in,  rank]  ==  stored V_latent.T

    deployed:  W_eff = diag(scale_post) . sign(U_latent) . sign(V_latent) . diag(scale_pre)

    gauge:     U_R = U R,  V_R = V R   =>   U_R V_R^T = U R R^T V^T = U V^T

Block-orthogonal R along the rank axis, block size b = 32 (rank 1600 -> 50
blocks, 992 -> 31, 384 -> 12).

## Q_NQ, the single deterministic export operator

NanoQuant's own export rule is *mean magnitude over the rank axis*
(`scale_pre = B_final.abs().mean(dim=0)`, `scale_post = A_final.abs().mean(dim=1)`).
Those two vectors come from `A_final`/`B_final` — the rank-1-projected `A_z`/`B_z`
— and are not recoverable from the stored latents alone, so Q_NQ applies
NanoQuant's rule to the rotated latents *relative to the base latents*:

    scale_post(R) = scale_post_0 * rowmean|U R| / rowmean|U|
    scale_pre (R) = scale_pre_0  * colmean|V_R| / colmean|V|

This is section 4.2's requirement (export statistics recomputed from the current
rotated factors, then detached) while satisfying the section 4.3 identity gate
exactly: at R = I the ratio is 1.0 and the cached ADMM export is reproduced
bit-for-bit.

**There is no SVID on the export path.** `admm_nq.svid` / `rank1_approx` are used
only *inside* the ADMM outer loop; the post-ADMM -> deployed conversion draws no
random numbers at all. Q_NQ is therefore already a deterministic function of
(U_R, V_R), proved by bit-comparing repeated calls rather than by RNG replay.
Snapshot A is still banked so the claim is auditable.

One numerical trap had to be fixed to make the identity gate hold: fp32
`mean(dim=...)` on CUDA is **not** bit-reproducible across two tensors holding
identical values at different addresses — the reduction splits differently and a
handful of coordinates come out ~1e-7 apart, which is enough to move the export
ratio off 1.0 by an ulp. The export statistic is accumulated in fp64, which puts
the layout-dependent discrepancy at ~1e-16 so the ratio rounds to exactly 1.0f.

## Stage 0 — every assertion passed

| check | result |
|---|---|
| no `scale_mid` in the reproduced state | PASS (section 2.1 path) |
| `\\|U_R V_R - U V\\|_F / \\|U V\\|_F` for random block-orthogonal R, all 7 layers | PASS, max **8.5e-7** |
| `R = I` gives bit-identical U and V | PASS, all 7 layers |
| Cayley orthogonality `max_i \\|R_i^T R_i - I\\|_F` | PASS, **3.7e-6** at random A, **0.0** at A = 0 |
| Cayley at A = 0 is exactly I | PASS, max deviation 0.0 |
| `Q_NQ(U, V)` reproduces the cached export bit-for-bit (signs *and* scales) | PASS, all 7 layers |
| Q_NQ deterministic (repeat calls bit-identical) | PASS |
| functional loss reaches the Cayley parameters, finite and non-zero | PASS |
| frozen base `U`, `V` have `grad is None` | PASS |
| no export scale/magnitude tensor is an optimiser parameter | PASS |
| one Adam step moves R, and R stays orthogonal | PASS |
| held-out guard rejects the held-out split, accepts calibration | PASS |

Machine-readable: `stage0.json`.

## Reference points

| | value |
|---|---|
| released interleaved schedule, block 0 `pre` / `post` (section 0.3 gate) | 0.1415907 / 0.1232319 |
| reproduced D2 post-ADMM held-out block error `E_ADMM` | **0.1437658** |
| previously cached D2 `pre` / joint-Step-3 `post` | 0.1436614 / 0.1148516 |
| **Arm 0a `E_final`** (this VM's baseline) | **0.1153880** |
| **Arm 0a duplicate `E_final`** (replay floor) | **0.1151653**, i.e. **0.19%** |
| identity-gauge full-calibration functional loss | 0.125113 |

The ADMM was re-derived from scratch on this VM. NOTES.md#2 records that the
ADMM is chaotic — a relative 1e-7 perturbation of the input weight diverges to a
relative 0.87 by iteration 160, and the default runs 400 — so the reproduced
post-ADMM state is an *independent draw* from the same procedure, not a replay.
`E_ADMM` lands +0.073% from the cached value and Arm 0a's `E_final` +0.47% from
the cached 0.1148516. The 0.19% same-state replay floor measured here is inside
the 0.31% floor the repo established, so **δ = 0.0031 is kept as the gate**, and
every arm below branches from the *same* reproduced state, so the comparison is
internally controlled.

## Results

"""


def main():
    os.makedirs(OUT, exist_ok=True)
    body = report.main()
    narr = open(NARR).read() if os.path.exists(NARR) else ""
    doc = HEADER + body + "\n" + narr
    tmp = f"{OUT}/stage1_down.md.tmp"
    with open(tmp, "w") as f:
        f.write(doc)
    os.replace(tmp, f"{OUT}/stage1_down.md")
    print(f"wrote {OUT}/stage1_down.md ({len(doc)} bytes)")


if __name__ == "__main__":
    main()
