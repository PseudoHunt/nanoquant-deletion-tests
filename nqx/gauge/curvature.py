"""Kronecker-factored curvature and block-residual gradient, per linear layer.

For a linear layer with input activations X [N, in] and pre-nonlinearity output
z = X W^T [N, out], and the TRUE block objective

    L = ||B_q - B_fp||_F^2 / ||B_fp||_F^2

evaluated at the *current quantised* block (so the other six projections keep
their NanoQuant errors, and any cross-layer error cancellation is present), this
accumulates three quantities in one backward pass over the calibration set:

    G_W    = dL/dW            = sum_n g_n x_n^T        [out, in]
    H_in   = (1/N) X^T X                               [in, in]
    H_out  = (1/N) sum_n g_n g_n^T                     [out, out]

with g_n = dL/dz_n.  `N` is the number of **tokens** (128 sequences x 2048), and
that convention is identical for every layer.

Two things to be explicit about, because they are easy to get silently wrong:

* `G_W` is the *true* gradient of the normalised block error, so the factor of 2
  that appears when the objective is written as ||r||^2 is already inside it.  A
  first-order prediction is therefore `<G_W, dW>`, with coefficient 1.
* `H_out` is the **empirical Fisher** (outer products of gradients), not the
  Gauss-Newton factor J^T J.  It has the right eigenstructure but its overall
  scale carries an extra factor of order ||residual||^2.  That is precisely why
  the curvature term needs a free coefficient lambda, and why lambda has to be
  swept over decades rather than over {0.25 ... 4} alone.

Nothing here trains anything: the backward pass exists only to build a scorer.
"""
import time

import torch

from nanoquant.modules.hub import NanoQuantConfigDataclass  # noqa: F401  import-order guard

from . import core as C
from . import harness as H

NAMES = H.NAMES


class LayerCurvature:
    def __init__(self, name, G_W, H_in, H_out, n_tokens, in_f, out_f):
        self.name, self.G_W, self.H_in, self.H_out = name, G_W, H_in, H_out
        self.n_tokens, self.in_features, self.out_features = n_tokens, in_f, out_f

    def to(self, dev):
        for k in ["G_W", "H_in", "H_out"]:
            setattr(self, k, getattr(self, k).to(dev))
        return self


def collect(st, layers=None, device="cuda", n_seq=None, dtype=torch.float32,
            h_out_samples=32768):
    """Accumulate G_W, H_in (all tokens) and H_out (a token subsample).

    H_out is [out, out]; for gate/up_proj that is 8192^2, and accumulating
    `Gz^T Gz` over all 262144 tokens costs ~7e13 flops -- an hour in fp32 and
    far worse in fp64.  It is a covariance used inside a trace (never inverted),
    so it is estimated from a strided token subsample of `h_out_samples` tokens
    (>= 4x the largest output dimension).  `G_W` and `H_in` use every token.
    Accumulation is fp32; the final scoring is fp64.  Both choices are validated
    in `surrogate_diagnostic` against denser / higher-precision references.
    """
    """One forward+backward pass per calibration sequence, accumulating the
    Kronecker factors and the block-residual gradient for every layer."""
    blk, kwargs = st["blk"], st["kwargs"]
    sub = H.nq_sub(blk)
    layers = layers or [n for n in NAMES if n != "mlp.down_proj"]
    cal_in, cal_out = st["cal_in"], st["cal_out"]
    C.GUARD.assert_not_heldout(cal_in, cal_out, where="curvature.collect")
    n = n_seq or cal_in.shape[0]
    tok_per_seq = cal_in.shape[1]
    stride = max(1, (n * tok_per_seq) // max(h_out_samples, 1))

    acc = {}
    for name in layers:
        m = sub[name]
        acc[name] = {
            "G_W": torch.zeros(m.out_features, m.in_features, device=device, dtype=dtype),
            "H_in": torch.zeros(m.in_features, m.in_features, device=device, dtype=dtype),
            "H_out": torch.zeros(m.out_features, m.out_features, device=device, dtype=dtype),
        }

    cap = {}
    hooks = []
    for name in layers:
        def fwd(mod, inp, out, nm=name):
            cap[nm] = (inp[0], out)
            out.retain_grad()
        hooks.append(sub[name].register_forward_hook(fwd))

    # every parameter frozen: we want grads w.r.t. activations only
    for p in blk.parameters():
        p.requires_grad_(False)

    n_tok = 0
    t0 = time.time()
    denom = 0.0
    with torch.no_grad():
        for j in range(n):
            denom += float(cal_out[j:j + 1].double().square().sum())

    for j in range(n):
        x = cal_in[j:j + 1].to(device)
        t = cal_out[j:j + 1].to(device)
        with torch.enable_grad():
            xg = x.clone().requires_grad_(True)   # gives the graph something to hang on
            y = blk(xg, **kwargs)[0]
            L = (y.double() - t.double()).square().sum() / denom
            L.backward()
        for name in layers:
            xin, zout = cap[name]
            gz = zout.grad
            if gz is None:
                raise RuntimeError(f"no grad captured at {name}")
            X = xin.detach().to(dtype).flatten(0, -2)     # [T, in]
            Gz = gz.detach().to(dtype).flatten(0, -2)     # [T, out]
            with C.no_tf32():
                acc[name]["G_W"].add_(Gz.T @ X)
                acc[name]["H_in"].add_(X.T @ X)
                Gs = Gz[::stride]
                acc[name]["H_out"].add_(Gs.T @ Gs)
                acc[name]["n_hout"] = acc[name].get("n_hout", 0) + Gs.shape[0]
            zout.grad = None
        n_tok += cal_in.shape[1]
        blk.zero_grad(set_to_none=True)
        cap.clear()

    for h in hooks:
        h.remove()
    print(f"[curv] {len(layers)} layers, {n} sequences / {n_tok} tokens "
          f"in {time.time()-t0:.0f}s", flush=True)

    out = {}
    for name in layers:
        m = sub[name]
        nh = acc[name]["n_hout"]
        lc = LayerCurvature(
            name, acc[name]["G_W"].double(),
            (acc[name]["H_in"] / n_tok).double(), (acc[name]["H_out"] / nh).double(),
            n_tok, m.in_features, m.out_features)
        lc.n_hout = nh
        lc.h_out_stride = stride
        out[name] = lc
    return out
