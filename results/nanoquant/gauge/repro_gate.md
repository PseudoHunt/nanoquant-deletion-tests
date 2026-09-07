# Gauge-NQ bring-up: baseline reproduction gate (brief section 0.3)

Fresh VM, both repositories re-cloned, environment rebuilt from scratch.

## Environment

| | |
|---|---|
| GPU | 1x NVIDIA A30 24 GB, driver 580.126.20, CUDA 13.0 |
| CPU | 16 vCPU, 377 GB RAM |
| torch | 2.11.0+cu130 |
| transformers | 4.57.1 (the forced deviation recorded in `repro.md`; the 4.51.3 pin is unsatisfiable) |
| datasets | 5.0.1 |
| `gemlite` / `optimum` / `zeus-ml` | not installed, as in `repro.md` |
| NanoQuant | `SamsungLabs/NanoQuant` @ `a9e0a43`, unmodified |
| model | `unsloth/Llama-3.2-1B`, snapshot `9535bd9b1d1dea6acafbdc4813b728796aeb28da` — byte-identical to the snapshot the earlier results were produced from (found already in the persistent volume) |
| gamma | 0.2 (Appendix C), not the repo default 0.4 |
| calibration | wikitext2, 128 x 2048, seed 0 |
| held-out | 32 x 2048 from wikitext2 **validation** |

## Result — released interleaved schedule, block 0

| | required | measured | abs. difference |
|---|---|---|---|
| `pre`  (held-out block error after ADMM) | 0.1416 | **0.1415907** | 9.3e-6 |
| `post` (after Step 3) | 0.1233 | **0.1232319** | 6.8e-5 |

Tolerance was 1e-3 absolute. Both pass by more than an order of magnitude.
Against the previously published cells (`1.41595e-1` / `1.23275e-1`) the
differences are 9.4e-6 and 4.3e-5, i.e. well inside the recorded 0.31%
determinism floor for block 0.

Artifact: `artifacts/sbh_runs/gauge_base_b0.json`.

## One forced change to the harness

`nqx/sbh.py::do_cache` carried `@torch.no_grad()` over the whole cache stage.
That is the NOTES.md#1 fix, but the decorator is wider than the bug: NanoQuant's
calibration pass collects `o_norm` from **backward** hooks and calls
`loss.backward()`, which raises `RuntimeError: element 0 of tensors does not
require grad` under a global no-grad. The committed caches must therefore have
been produced before the decorator was added, and the cache stage as committed
could not be re-run.

The guard is now scoped to the activation-capture section
(`_capture_fp_states`), which is the part NOTES.md#1 is actually about; the
calibration pass runs in grad mode exactly as upstream. Explicit
`assert not t.requires_grad` was added at save, and the saved caches were
re-checked on load:

```
in_b0  (160, 2048, 2048) bf16 requires_grad=False
out_b0 (160, 2048, 2048) bf16 requires_grad=False
in_b8  (160, 2048, 2048) bf16 requires_grad=False
out_b8 (160, 2048, 2048) bf16 requires_grad=False
```

The reproduced `pre`/`post` above confirm the rebuilt caches are equivalent to
the ones the published baseline used.
