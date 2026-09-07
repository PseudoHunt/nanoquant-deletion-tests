"""Per-layer quadratic oracles for the six projections whose output does NOT
reach the block output affinely.

Only `mlp.down_proj` is affine in the block output: its result enters the final
residual add and nothing follows it.  `o_proj` is **not** -- its output lands in
`h = x + attn(...)`, and `h` then passes through `post_attention_layernorm` and
the SwiGLU MLP before reaching the block output.  So for q/k/v/o/gate/up we fall
back to a layer-local surrogate:

    E_layer(W) = sum_o  o_norm[o] . (w_o - wfp_o)^T H (w_o - wfp_o),   H = X^T X

i.e. the *full* input Gram (not NanoQuant's diagonal `i_norm`) with NanoQuant's
own `o_norm` output weighting.  Expanded, that is the same shape the block oracle
already has,

    E = const - 2<G, W> + tr(D_w W H W^T),   G = D_on W_fp H,   D_w = diag(o_norm)

so the identical Givens algebra scores it, with the per-output weight carried
through `Oracle.out_weight`.

**This surrogate is exactly the objective this repository has already shown can
mislead.**  Tests A, B and A+B all improved layer-wise reconstruction and all
ended with *worse* post-Step-3 block error.  It is used here only to rank
candidate sign patterns cheaply; every layer is then gated on the true
calibration block error and reverted if that got worse.
"""
import time

import torch

from nanoquant.modules.hub import NanoQuantConfigDataclass  # noqa: F401  import-order guard

from . import core as C
from . import harness as H
from . import quad as Q
from . import state as S

NAMES = H.NAMES


@torch.no_grad()
def build_all(st, device="cuda", n_seq=None):
    """One pass over the calibration set, capturing every layer's input Gram."""
    blk, kwargs = st["blk"], st["kwargs"]
    sub = H.nq_sub(blk)
    cal_in, cal_out = st["cal_in"], st["cal_out"]
    C.GUARD.assert_not_heldout(cal_in, cal_out, where="layer_oracle.build_all")
    n = n_seq or cal_in.shape[0]

    cap, hooks = {}, []
    for name in NAMES:
        hooks.append(sub[name].register_forward_hook(
            lambda mod, i, o, n=name: cap.__setitem__(n, i[0].detach())))

    Hm = {n_: torch.zeros(sub[n_].in_features, sub[n_].in_features,
                          device=device, dtype=torch.float64) for n_ in NAMES}
    t0 = time.time()
    for j in range(n):
        blk(cal_in[j:j + 1].to(device), **kwargs)
        for name in NAMES:
            z = cap[name].double().flatten(0, -2)
            with C.no_tf32():
                Hm[name].add_(z.T @ z)
    for h in hooks:
        h.remove()
    print(f"[layer_oracle] input Grams for {len(NAMES)} layers over {n} sequences "
          f"in {time.time()-t0:.0f}s", flush=True)

    oracles = {}
    for name in NAMES:
        m = sub[name]
        Hh = 0.5 * (Hm[name] + Hm[name].T)
        W_fp = st["W_refs"][name].to(device).double()
        on = m.o_norm.to(device).double() if hasattr(m, "o_norm") \
            else torch.ones(m.out_features, device=device, dtype=torch.float64)
        with C.no_tf32():
            G = (on.unsqueeze(1) * W_fp) @ Hh
            # constant so E is reported on the same relative scale as the block
            # oracle: E = (T_sq - 2<G,W> + tr(D_w W H W^T)) / Y_sq  with
            # T_sq = tr(D_on W_fp H W_fp^T) and Y_sq the same quantity, so
            # E(W_fp) = 0 and E is a relative reconstruction error.
            T_sq = float((on.unsqueeze(1) * W_fp * (W_fp @ Hh)).sum())
        blob = {"H": Hh.cpu(), "G": G.cpu(), "T_sq": T_sq, "Y_sq": T_sq,
                "out_weight": on.cpu(),
                "so0": m.scale_post.data.detach().float().view(-1).cpu(),
                "sp0": m.scale_pre.data.detach().float().view(-1).cpu(),
                "in_features": m.in_features, "out_features": m.out_features,
                "layer": name, "kind": "layer_surrogate"}
        oracles[name] = Q.Oracle(blob)
        del Hh, W_fp, G
        torch.cuda.empty_cache()
    del Hm
    torch.cuda.empty_cache()
    return oracles
