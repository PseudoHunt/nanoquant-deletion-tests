# Step 0 — reproduction of the stock NanoQuant pipeline

## Model selection

The brief asks for Qwen2.5-0.5B if `nanoquant` supports it. **It does not.** The
README advertises "Qwen (Qwen-2.5, Qwen-3)", but the code gates on
`config.model_type`, and both gates list only `qwen3`:

* `utils/utils.py::get_layers_to_factorize` — `["llama", "mistral", "mixtral", "mobilellm", "qwen3"]` or `gemma*`, else `raise ValueError`
* `utils/utils.py::get_decoder_layers` — same list

`Qwen/Qwen2.5-0.5B` has `model_type: "qwen2"`, so it raises
`ValueError: Unsupported model type: qwen2` before any work happens. (`Rnj-1`,
also advertised, has no gate either.) Falling back as instructed to
**Llama-3.2-1B** (`model_type: "llama"`, fully supported).

`meta-llama/Llama-3.2-1B` is gated (HTTP 401 without a token), so this uses the
ungated mirror **`unsloth/Llama-3.2-1B`**. Two independent checks say the weights
are the ones the paper used: the config matches Llama-3.2-1B exactly (16 layers,
hidden 2048, intermediate 8192, 32 heads / 8 KV heads, vocab 128256), and the
measured BF16 wikitext2 perplexity is **9.7499** against the paper's **9.74**
for `L3-1` in Table 2.

## Environment

| | |
|---|---|
| GPU | 1x NVIDIA A30 24 GB, driver 580.126.20, CUDA 13.0 |
| torch | 2.11.0+cu130 |
| transformers | **4.57.1** — see deviation below |
| repo | `SamsungLabs/NanoQuant` @ `a9e0a43`, Apache-2.0, unmodified |

**Deviation (forced):** `pyproject.toml` pins `transformers==4.51.3`, but the
repo's own `load_utils.load_model` calls
`AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16, ...)`. The
`dtype=` alias only exists from transformers 4.56; on the pinned 4.51.3 the
pipeline dies immediately with
`TypeError: LlamaForCausalLM.__init__() got an unexpected keyword argument 'dtype'`.
The pin is unsatisfiable with the code as published, so this uses 4.57.1, the
nearest 4.x that runs it. `gemlite`, `optimum` and `zeus-ml` are not installed;
none is reachable on this path (`gemlite` is lazily imported, the other two are
kernel-benchmark only).

## Configuration

Stock defaults, matching the README's own example, via `python -m nanoquant.main`:

```
--bits 1.0 --seed 0 --num_calib_samples 128 --calib_dataset wikitext2
--nonfact_epochs 8 --fact_epochs 8 --admm_outer_iters 400
--ppl_task "wikitext2,c4" --zeroshot_task ""
```

`calib_strategy=online`, `calib_shrinkage=0.4`, `admm_inner_iters=5`,
`admm_reg=3e-2`, `admm_penalty_scheduler=linear`, `tune_model=True`
(`model_kd_epochs=8`, not exposed on the CLI).

Ranks from their own `calculate_ranks` bit accounting
(`rank = bits*d_in*d_out/(d_in+d_out) - 16`, floored to a multiple of 32):

| layer | shape | rank |
|---|---|---|
| `q_proj`, `o_proj` | 2048x2048 | 992 |
| `k_proj`, `v_proj` | 512x2048 | 384 |
| `gate_proj`, `up_proj` | 8192x2048 | 1600 |
| `down_proj` | 2048x8192 | 1600 |

Measured storage over the 112 factorised layers: binary factors 0.9741 bpw,
plus `scale_pre`/`scale_post` **0.9857 bpw total**. (Embeddings, `lm_head` and
norms stay FP16, as in the paper.)

## Results (seed 0)

### Per-layer ADMM reconstruction error

Normalised error `||W_final - W||^2 / ||W||^2` as the repo prints it, mean +- sd
over the 16 blocks:

| layer | recon error | ADMM time |
|---|---|---|
| `self_attn.q_proj` | 0.2158 +- 0.0273 | 2.45 s |
| `self_attn.v_proj` | 0.3340 +- 0.0274 | 1.19 s |
| `self_attn.o_proj` | 0.2717 +- 0.0202 | 2.62 s |
| `self_attn.k_proj` | 0.2598 +- 0.0290 | 1.27 s |
| `mlp.gate_proj` | 0.3042 +- 0.0121 | 8.64 s |
| `mlp.up_proj` | 0.3407 +- 0.0121 | 8.35 s |
| `mlp.down_proj` | 0.3320 +- 0.0064 | 8.19 s |
| **all 112 layers** | **0.2940** | 4.67 s |

### Per-block wikitext2 PPL (their own after-each-block probe)

| block | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| PPL | 11.18 | 12.08 | 12.45 | 13.27 | 13.86 | 14.28 | 14.50 | 15.24 |

| block | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|
| PPL | 16.25 | 17.99 | 19.52 | 21.11 | 22.24 | 23.81 | 25.86 | 32.29 |

The final 32.29 is before the global KD stage, which recovers it to 27.48.

### End-to-end perplexity vs the paper

| | wiki2 | c4 |
|---|---|---|
| BF16, measured | 9.7499 | 14.018 |
| BF16, paper Table 2 (`L3-1`) | 9.74 | - |
| **NanoQuant 1.0 bpw, measured (seed 0)** | **27.48** | **89.39** |
| NanoQuant 1.00 bpw, paper Table 2 (`L3-1`) | **25.59** | - |

**Reproduction gap: +1.89 PPL, +7.4% relative.** The BF16 number matches the
paper to 0.01, so the model, tokenizer and PPL protocol are right; the gap is in
the quantisation run itself. Plausible contributors, none verified: the paper's
seed and calibration draw are unstated; `model_kd_epochs` is not CLI-exposed and
defaults to 8; transformers 4.57.1 vs whatever the authors ran. The paper
reports no c4 perplexity for this model, so 89.39 has no reference.

### Timing (single A30, seed 0)

| stage | time |
|---|---|
| calibration (`collect_stats`, 128 seqs) | 190 s |
| ADMM over all 112 layers | 523 s |
| Step 1, non-factorised tuning | 1287 s |
| Step 3, factorised refinement | 1741 s |
| global KD + final wiki2/c4 eval | 706 s |
| other (data prep, per-block PPL probes, transfers) | ~243 s |
| **total** | **4690 s (78 min)** |

## Byte-exact re-run gate

**Status: not yet run at full-model scale** (queued behind the single-block
harness). One result is already in hand and changes how the gate should be read.

The stock ADMM is **chaotically unstable**, because `rank1_approx` applies a
`sign()` projection every outer iteration. Perturbing the input weight matrix by
a relative **1e-7** and re-running upstream against itself:

| outer iters | 1 | 2 | 5 | 10 | 20 | 40 | 80 | 160 |
|---|---|---|---|---|---|---|---|---|
| rel. difference in `W_final` | 2.7e-6 | 1.1e-6 | 1.3e-6 | 2.0e-6 | 8.3e-7 | **1.5e-1** | **6.3e-1** | **8.7e-1** |

At the 400 iterations the default config uses, a last-bit difference anywhere in
the pipeline produces a completely different factorisation. So the byte-exact
gate tests only whether every CUDA kernel on the path is bit-deterministic; it
is not a stability property of the method, and a failure would say nothing about
whether a later edit is clean. The gate is still worth running as a
determinism check, and is reported when it completes.

## Artifacts

* `logs/repro_seed0_run1.log` — full timestamped run log
* `artifacts/step0_seed0_run1.json` — parsed per-layer errors, timings, PPLs
* `artifacts/fp_baseline.json` — BF16 reference PPLs
* `parse_step0.py` — log parser
