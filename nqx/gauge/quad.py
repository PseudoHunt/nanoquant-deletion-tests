"""Exact forward-pass-free oracle for the calibration block error as a function
of `mlp.down_proj`'s deployed weight.

Why this is exact rather than a surrogate
-----------------------------------------
In a Llama decoder layer, `down_proj`'s output enters the block output through a
plain residual add and nothing else:

    h   = x + attn(ln1(x))
    out = h + down_proj( silu(gate(ln2(h))) * up(ln2(h)) )

so with every other projection frozen, the block output is **affine** in
`down_proj`'s weight.  Writing `Z` for down_proj's input activations over the
calibration set and `T` for the target its output must hit
(`T = Y_fp - (out - down_proj_out)`), the calibration block error is exactly

    E(W) = ( ||T||^2 - 2 <G, W> + tr(W H W^T) ) / ||Y_fp||^2,
    G = T^T Z   [out, in],   H = Z^T Z   [in, in]

so any candidate weight can be scored with no forward pass at all.  This is what
makes discrete search over sign-pattern breakpoints affordable: the alternative
is a full block forward over 128x2048 tokens per candidate.

`H` here is the raw uncentred second moment of the *current block's* down_proj
input.  It is **not** NanoQuant's `i_norm` / `H_in` statistic (clipped at the
99.9th percentile and shrunk toward its diagonal); that one is the ADMM's
objective, this one is the block error.  They are different matrices and are not
interchangeable.

The oracle is fp32 while the deployed forward is bf16 with a different
association order (`(x s_pre) B_V^T B_U^T s_post` rather than `x W^T`), so it is
used for *search* only.  Every reported number still comes from the real forward.
`verify()` measures the gap.
"""
import json
import os
import time

import torch

from nanoquant.modules.hub import NanoQuantConfigDataclass  # noqa: F401  import-order guard

import instrument as I

from . import core as C
from . import harness as H
from . import state as S

LAYER = "mlp.down_proj"


class Oracle:
    def __init__(self, blob):
        for k, v in blob.items():
            setattr(self, k, v)

    def to(self, dev):
        for k in ["H", "G", "so0", "sp0", "out_weight"]:
            if getattr(self, k, None) is not None:
                setattr(self, k, getattr(self, k).to(dev))
        return self

    def out_w(self, device=None):
        """Per-output weight in the quadratic term.  Ones for the exact
        block-output oracle (down_proj); `o_norm` for the layer surrogates."""
        w = getattr(self, "out_weight", None)
        if w is None:
            w = torch.ones(self.out_features, dtype=torch.float64)
            self.out_weight = w
        return w.to(device) if device is not None else w

    def error(self, W):
        """E(W) for a deployed weight W [out, in].  fp64: E is a ~6x cancellation
        of terms of order ||T||^2, so fp32 here would put ~1e-5 of noise on a
        curve whose whole signal is ~5e-5."""
        Wd = W.double()
        w = self.out_w(Wd.device)
        with C.no_tf32():
            quad = (w.unsqueeze(1) * Wd * (Wd @ self.H)).sum()
            lin = (self.G * Wd).sum()
        return float((self.T_sq - 2 * lin + quad) / self.Y_sq)


@torch.no_grad()
def build(seed=0, device="cuda"):
    st = H.load_post_admm(seed=seed)
    blk, kwargs = st["blk"], st["kwargs"]
    sub = H.nq_sub(blk)
    m = sub[LAYER]
    cal_in, cal_out = st["cal_in"], st["cal_out"]
    C.GUARD.assert_not_heldout(cal_in, cal_out, where="quad.build")

    cap = {}
    h1 = m.register_forward_hook(lambda mod, i, o: cap.__setitem__("z", i[0].detach()))
    h2 = m.register_forward_hook(lambda mod, i, o: cap.__setitem__("d", o.detach()))

    in_f, out_f = m.in_features, m.out_features
    # fp64 accumulation.  H and G are the objective itself, not just intermediate
    # algebra: an fp32 H carries ~1e-6 relative error, and since the quadratic
    # term is O(1e4) that is ~1e-2 absolute -- larger than a median improving
    # move (delta_f ~ 0.3).  It would not show up in any self-consistency check,
    # because the same wrong H is used on both sides; it would simply mean the
    # search is optimising a slightly wrong objective.
    Hm = torch.zeros(in_f, in_f, device=device, dtype=torch.float64)
    Gm = torch.zeros(out_f, in_f, device=device, dtype=torch.float64)
    T_sq = torch.zeros((), device=device, dtype=torch.float64)
    Y_sq = torch.zeros((), device=device, dtype=torch.float64)

    t0 = time.time()
    for j in range(cal_in.shape[0]):
        x = cal_in[j:j + 1].to(device)
        yfp = cal_out[j:j + 1].to(device).float()
        y = blk(x, **kwargs)[0].float()
        z = cap["z"].double().flatten(0, -2)         # [T, in]
        d = cap["d"].double().flatten(0, -2)         # [T, out]
        t = (yfp.double().flatten(0, -2) - y.double().flatten(0, -2)) + d
        with C.no_tf32():
            Hm.add_(z.T @ z)
            Gm.add_(t.T @ z)
        T_sq += t.square().sum()
        Y_sq += yfp.double().square().sum()
        del x, yfp, y, z, d, t
    h1.remove(); h2.remove()
    Hm = 0.5 * (Hm + Hm.T)
    print(f"[quad] H, G accumulated over {cal_in.shape[0]} sequences in {time.time()-t0:.0f}s",
          flush=True)

    base = C.LayerBase(LAYER, m).to(device)
    blob = {"H": Hm.cpu(), "G": Gm.cpu(), "T_sq": float(T_sq), "Y_sq": float(Y_sq),
            "so0": base.so0.cpu(), "sp0": base.sp0.cpu(),
            "in_features": in_f, "out_features": out_f, "layer": LAYER, "seed": seed}
    S.atomic_save(blob, f"{H.GCACHE}/quad_oracle_fp64.pt")
    return Oracle(blob), st, base


@torch.no_grad()
def verify(oracle, st, base, device="cuda"):
    """Oracle vs the real bf16 forward, at the current point and at random gauges."""
    blk, kwargs = st["blk"], st["kwargs"]
    sub = H.nq_sub(blk)
    m = sub[LAYER]
    oracle.to(device)
    rec = {"cases": []}

    def real_calib_err():
        return H.func_loss_over(blk, st["cal_in"].to(device), st["cal_out"].to(device), kwargs)

    g = torch.Generator(device=device); g.manual_seed(999)
    for tag, Rs in [("identity", C.identity_Rs(base.rank, 32, device)),
                    ("haar_a", [C.haar_so(s, g) for s in C.block_sizes(base.rank, 32)]),
                    ("haar_b", [C.haar_so(s, g) for s in C.block_sizes(base.rank, 32)])]:
        H.materialize_gauge(m, base, Rs)
        real = real_calib_err()
        W = I.effective_weight(m).float()
        quad = oracle.error(W)
        rec["cases"].append({"case": tag, "real_forward": real, "oracle": quad,
                             "rel_gap": abs(quad - real) / real})
        print(f"[quad] {tag:9s} real = {real:.8f}  oracle = {quad:.8f}  "
              f"rel gap = {abs(quad-real)/real:.2e}", flush=True)
    H.materialize_gauge(m, base, C.identity_Rs(base.rank, 32, device))
    rec["max_rel_gap"] = max(c["rel_gap"] for c in rec["cases"])
    S.atomic_json(rec, f"{H.GCACHE}/quad_verify.json")
    return rec


if __name__ == "__main__":
    os.makedirs(H.GCACHE, exist_ok=True)
    o, st, base = build()
    r = verify(o, st, base)
    print(f"[quad] max relative gap oracle-vs-forward: {r['max_rel_gap']:.2e}")
