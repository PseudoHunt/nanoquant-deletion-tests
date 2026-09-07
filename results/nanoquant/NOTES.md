# Notes: things that bit, and would have silently biased results

## 1. `requires_grad=True` leaking into cached activations (harness bug, mine)

The single-block harness caches each block's FP input to disk. My first cache
stage called `cache_inputs_and_kwargs` **outside** `torch.no_grad()`. Inside that
function the captured tensor is filled by `cache['inputs'][i] = inp.cpu()`; under
grad mode that in-place index assignment makes the *destination* require grad, so
`in_b0.pt` was serialised with `requires_grad=True`.

Every subsequent training step therefore also back-propagated into a
160x2048x2048 input tensor and accumulated a gradient buffer for it. Cost:
**361 ms/step instead of 15.7 ms/step, a 23x slowdown**, with numerically
identical results. Upstream avoids this only because `compress_block_recon`
carries an `@torch.no_grad()` decorator; nothing else in the path would have
caught it.

Worth recording for two reasons. It never changed a number, so no correctness
check would have flagged it -- but every timing claim built on that harness would
have been wrong by more than an order of magnitude. And it produced a convincing
false result on the way: benchmarking "real cached activations" against
`torch.randn` showed real data 20x slower and reproduced under reversed ordering,
which looks exactly like a numerics finding. It was an artifact of the random
tensors not requiring grad.

Fix: `@torch.no_grad()` on the cache stage, plus `.detach().requires_grad_(False)`
on load. Guard for anyone reusing this: assert `not t.requires_grad` on every
cached activation tensor.

## 2. The ADMM is chaotic, so "byte-exact" is a determinism test, not a stability test

`rank1_approx` applies `sign()` every outer iteration, so the iteration map is
discontinuous. Perturbing the input weight by a relative **1e-7** and re-running
upstream against itself diverges to a relative difference of **0.15 by iteration
40** and **0.87 by iteration 160**; the default config runs 400. Consequences:

* A byte-exact re-run gate tests only whether every CUDA kernel on the path is
  bit-deterministic. It cannot certify that a later edit is "clean".
* Any equivalence check against upstream has to be made at the level of the
  update rule, not the trajectory. Test B's unit test is built that way: the
  solve step at `H = I` is bit-exact, and the full loop is bit-exact at
  `H_in = D_in^2` for all five real Llama-3.2-1B layer shapes.
* Getting that bit-exactness required two fixes that are easy to miss: computing
  `H @ X` with TF32 **off** (TF32 rounds its inputs, so a 1e-7 difference becomes
  1e-4), and forming `H~ = H_in / (d (x) d)` by division rather than by
  multiplying with a rounded reciprocal (so `H~` is *exactly* the identity rather
  than identity to 1e-7).

## 3. Config drift between the repo defaults and the paper's Appendix C

See `repro.md`. Two knobs in the shipped defaults do not match the paper:
`calib_shrinkage` (gamma) is 0.4 where Appendix C says 0.2 for Llama/Qwen, and
`model_kd_lr` is 1e-5 where Appendix C says 1e-6.

## 4. `mean(dim=...)` on CUDA is not bit-reproducible across two identical tensors

Gauge-NQ needs `Q_NQ(U, V)` to reproduce the cached ADMM export **bit-for-bit**
at `R = I`, because that is what makes Arm 0a the baseline rather than an
approximation of it. The export statistic is NanoQuant's own mean magnitude over
the rank axis, applied relatively:
`scale_post(R) = scale_post_0 * rowmean|U R| / rowmean|U|`, whose ratio ought to
be exactly 1.0 when `U R` is bit-identical to `U`.

It was not. `torch.equal(U_R, U0)` passed on all seven layers and the ratio still
missed 1.0 in 4 of 8192 coordinates on `down_proj` and 14 of 2048 on `o_proj`,
by ~1e-7 relative. The two tensors hold identical values at different addresses,
and the fp32 reduction splits differently for different alignments, so the
partial sums are accumulated in a different order. Nothing about the input
differs; only where it lives.

Consequences worth generalising:

* `torch.equal(a, b)` does **not** imply `f(a) == f(b)` for a reduction `f` on
  CUDA. Bit-exactness of inputs is not bit-exactness of outputs.
* It is shape-independent in the confusing way: `q_proj` (2048x2048) was clean
  and `o_proj` (2048x2048) was not, so it looks data-dependent and sends you
  hunting for a numerical bug in the values.

Fix: accumulate the statistic in fp64 (`x.abs().sum(dim=..., dtype=torch.float64) / n`).
The layout-dependent discrepancy drops to ~1e-16, so the ratio rounds to exactly
1.0 in fp32, while a genuine rotation's ratio is unaffected. This is the same
family as the TF32 trap in item 2: the arithmetic is "the same" right up until
you need an identity to hold exactly.

## 5. The `@torch.no_grad()` fix in item 1 was scoped too wide to re-run

`nqx/sbh.py::do_cache` as committed carries `@torch.no_grad()` on the whole cache
stage. That is item 1's fix, but the decorator covers more than the bug: NanoQuant's
calibration pass collects `o_norm` from **backward** hooks and calls
`loss.backward()` inside `_run_calibration_loop`, which raises
`RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`
under a global no-grad.

So the committed caches must have been produced *before* the decorator was added,
and the cache stage as committed could not be re-run — which only shows up on a
fresh VM, when it is the first thing you need. The guard is now scoped to the
activation-capture section (`_capture_fp_states`), which is the part item 1 is
about; the calibration pass runs in grad mode exactly as upstream. Rebuilt caches
reproduce block 0's `pre`/`post` to 9.3e-6 / 6.8e-5.

Guard for anyone reusing this: a fix that makes the numbers right is not
necessarily a fix that leaves the pipeline runnable, and the difference is
invisible until someone starts from nothing.

## 6. Comparing single runs against a single baseline draw produced a sign error

Every `Δ vs 0a` in the gauge work was measured against **one** run of the
baseline. Two facts, discovered late, make that invalid.

**Step 3 is the only stochastic stage, and it is not small.** Two runs of the
identical `arm4_b0123` state agreed to all 17 digits on `E_gauge_pre_export` and
`E_gauge` and differed on `E_final`:

```
                 run 1                run 2
E_gauge_pre      0.14360547733625048  0.14360547733625048   IDENTICAL
E_gauge          0.14360616334438345  0.14360616334438345   IDENTICAL
E_final          0.11539898669453942  0.11539345723449609   differs by 5.5e-06
```

So everything through ADMM-state load, gauge materialisation and `Q_NQ` export is
bit-reproducible across processes; the variance is entirely Step 3's. The
mechanism is item 2 one stage later: Step 3 flips ~1% of the binary signs, so a
last-bit difference can flip one sign the other way early and the trajectory
diverges.

**The distribution is wide and I sampled its tail.** Re-running the *identical*
`0a` state eight times:

```
0a_r5 0.115119726   0a_dup 0.115165320   0a_r6 0.115187942   0a_r1 0.115197283
0a_r3 0.115231123   0a_r2 0.115242372    0a_r4 0.115322623   0a    0.115388046

mean 0.115231804   sd 8.69e-05 (0.0754%)   spread 0.2329%
```

`0a` — the value every published `Δ vs 0a` was measured against — is **rank 8 of
8, the worst draw**, 1.8 sd above the mean. Differences of that size were being
reported as if they were effects. Re-scored against the distribution:

| arm | vs `0a` (as published) | vs baseline mean | z | |
|---|---|---|---|---|
| discrete gauge, 1 block | −0.030% | **+0.106%** | +1.41 | within noise |
| discrete gauge, 4 blocks | +0.010% | **+0.145%** | +1.92 | within noise |
| `0b_100` extra STE | +1.11% | +1.251% | +16.6 | **significant** |
| `0b_200` extra STE | +2.50% | +2.634% | +34.9 | **significant** |

The gauge arms flip sign: not marginally better than baseline, marginally worse.
The extra-STE regression is untouched and in fact far outside noise — that result
never depended on the floor.

Two guards for anyone reusing this harness:

* Any `E_final` claim smaller than **2 sd = 0.15%** carries no information from a
  single run. The repo's stated 0.31% gate is ~4 sd and remains conservative; the
  error was in the *point estimates*, not the gate.
* A comparison needs replicates of **both** sides. Replicating only the baseline
  tells you the floor but still leaves the treatment arm as one draw.
