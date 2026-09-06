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
