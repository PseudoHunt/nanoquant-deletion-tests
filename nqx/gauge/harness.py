"""Gauge-NQ harness: post-ADMM state, materialisation, the common Step 3, metrics.

Everything an arm needs, so that the only thing that differs between arms is the
state that the *identical* common Step 3 starts from.
"""
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanoquant.modules.hub import NanoQuantConfigDataclass  # noqa: F401  import-order guard

import instrument as I
import testD as D
from nanoquant.core.compress_block import fused_weighted_mse
from nanoquant.modules.linear import NanoQuantLinear
from nanoquant.optimi import AdamW
from nanoquant.utils.utils import cleanup_memory, set_seed

from . import core as C
from . import state as S

CACHE = D.CACHE                      # /home/work/exp/artifacts/sbh
DCACHE = D.DCACHE                    # /home/work/exp/artifacts/testD
GCACHE = "/home/work/exp/artifacts/gauge"
NAMES = D.NAMES
PAPER_GAMMA = D.PAPER_GAMMA

# The calibration batch sequence used by the gauge search (Arm 3) and, by
# construction, by the step/data-matched extra-STE control (Arm 0b).
GAUGE_DATA_SEED = 20260907


def gauge_batch_order(n_steps, n_samples=128, seed=GAUGE_DATA_SEED):
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    out = []
    while len(out) < n_steps:
        out.extend(torch.randperm(n_samples, generator=g).tolist())
    return out[:n_steps]


# ---------------------------------------------------------------------------
def load_post_admm(seed=0, device="cuda"):
    """The reproduced D2 post-ADMM state plus everything measurement needs."""
    model, qd, kwargs, cal_in, cal_out, ho_in, ho_out = D.load_harness(seed)
    C.GUARD.register(ho_in, ho_out)
    blob = torch.load(f"{DCACHE}/block0_postadmm.pt", weights_only=False)
    blk = blob["block"].to(device)
    return dict(model=model, qd=qd, kwargs=kwargs, cal_in=cal_in, cal_out=cal_out,
                ho_in=ho_in, ho_out=ho_out, blk=blk, W_refs=blob["W_refs"],
                ranks=blob["ranks"], admm_pre=blob["pre"])


def nq_sub(blk):
    return {n: m for n, m in blk.named_modules() if isinstance(m, NanoQuantLinear)}


def fresh_block(device="cuda"):
    """A fresh copy of the post-ADMM block, so arms never contaminate each other."""
    blob = torch.load(f"{DCACHE}/block0_postadmm.pt", weights_only=False)
    return blob["block"].to(device)


# ---------------------------------------------------------------------------
# Materialisation (section 6)
# ---------------------------------------------------------------------------
@torch.no_grad()
def materialize_gauge(module, base, Rs, check_identity=False):
    """Start from the frozen ADMM U0/V0, apply the gauge, run Q_NQ once, and
    replace *both* the continuous latent proxies and the export state.

    Returns a dict of diagnostics.  The continuous proxies written back are the
    gauged ones, so the common Step 3 starts from a state consistent with the
    selected gauge (never original U,V paired with gauged signs).
    """
    U_R, V_R = C.gauge(base, Rs)
    # NanoQuant stores latents in bf16; do the export from the bf16 values so the
    # module's own binary_ste reproduces Q_NQ's signs exactly.
    U_Rb = U_R.to(torch.bfloat16)
    V_Rb = V_R.to(torch.bfloat16)
    B_U, B_V, sp, so = C.q_nq(U_Rb.float(), V_Rb.float(), base)
    C.assert_qnq_deterministic(U_Rb.float(), V_Rb.float(), base)

    if check_identity:
        assert torch.equal(U_Rb, module.U_latent.data), "identity gauge moved U_latent"
        assert torch.equal(V_Rb, module.V_latent.data), "identity gauge moved V_latent"

    dev, dt = module.U_latent.device, module.U_latent.dtype
    sign_dU = (C.pos_sign(base.U0) != B_U).float().mean().item()
    sign_dV = (C.pos_sign(base.V0) != B_V).float().mean().item()
    # fp32-vs-bf16 sign agreement of the rotated factors (should be exactly 0)
    B_U32, B_V32, _, _ = C.q_nq(U_R, V_R, base)
    fp_bf_dU = (B_U32 != B_U).float().mean().item()
    fp_bf_dV = (B_V32 != B_V).float().mean().item()

    module.U_latent.data.copy_(U_Rb.to(dev, dt))
    module.V_latent.data.copy_(V_Rb.to(dev, dt))
    module.scale_pre.data.copy_(sp.view(1, -1).to(dev, dt))
    module.scale_post.data.copy_(so.view(1, -1).to(dev, dt))

    # the module's own forward must now reproduce Q_NQ's deployed weight
    W_mod = I.effective_weight(module)
    W_qnq = C.effective_W(B_U.to(dev), B_V.to(dev), sp.to(dev), so.to(dev))
    rel = (W_mod - W_qnq).norm().item() / max(W_qnq.norm().item(), 1e-30)
    assert rel < 1e-6, f"materialised module disagrees with Q_NQ, rel = {rel:.3e}"
    del U_R, V_R, U_Rb, V_Rb, B_U, B_V, B_U32, B_V32, W_mod, W_qnq
    return {"sign_delta_U": sign_dU, "sign_delta_V": sign_dV,
            "fp32_vs_bf16_sign_delta_U": fp_bf_dU, "fp32_vs_bf16_sign_delta_V": fp_bf_dV,
            "module_vs_qnq_rel": rel}


@torch.no_grad()
def apply_gauge_signs_only(module, base, Rs):
    """Gauged binary signs with the *original* (R = I) export statistics.

    Used only to measure `E_gauge_pre_export`: how much of a gauge's gain is
    present before the export statistics are re-extracted.  Never used as a
    starting state for Step 3.
    """
    U_R, V_R = C.gauge(base, Rs)
    dev, dt = module.U_latent.device, module.U_latent.dtype
    module.U_latent.data.copy_(U_R.to(torch.bfloat16).to(dev, dt))
    module.V_latent.data.copy_(V_R.to(torch.bfloat16).to(dev, dt))
    module.scale_pre.data.copy_(base.sp0.view(1, -1).to(dev, dt))
    module.scale_post.data.copy_(base.so0.view(1, -1).to(dev, dt))


# ---------------------------------------------------------------------------
# Differentiable gauged forward (Arm 3)
# ---------------------------------------------------------------------------
class GaugedForward:
    """Replaces one NanoQuantLinear's forward with the differentiable Q_NQ path.

    U0/V0 are frozen base tensors.  Scales and magnitudes are recomputed from the
    current rotated factors and detached; gradients reach the Cayley parameters
    only through the hard-sign STE (brief section 4.2).
    """

    def __init__(self, module, base, cayley):
        self.m = module
        self.base = base
        self.cayley = cayley
        self._orig = module.forward
        self.last_pre_export = None
        module.forward = self._forward

    def factors(self):
        Rs = self.cayley.Rs()
        with C.no_tf32():
            U_R = C.apply_R_U(self.base.U0, Rs)
            V_R = C.apply_R_V(self.base.V0, Rs)
        return U_R, V_R

    def _forward(self, x):
        U_R, V_R = self.factors()
        B_U, B_V, sp, so = C.q_nq(U_R, V_R, self.base, ste=True)
        dt = self.m.dtype
        y = F.linear(x * sp.to(dt), B_V.to(dt))
        y = F.linear(y, B_U.to(dt))
        y = y * so.to(dt)
        if self.m.bias is not None:
            y = y + self.m.bias
        return y

    def forward_fixed_scales(self, x):
        """Same, but with the base (R = I) export statistics: the pre-export point."""
        U_R, V_R = self.factors()
        B_U, B_V, _, _ = C.q_nq(U_R, V_R, self.base, ste=True)
        dt = self.m.dtype
        y = F.linear(x * self.base.sp0.to(dt), B_V.to(dt))
        y = F.linear(y, B_U.to(dt))
        y = y * self.base.so0.to(dt)
        if self.m.bias is not None:
            y = y + self.m.bias
        return y

    def remove(self):
        self.m.forward = self._orig


class _FixedScaleSwap:
    """Context manager swapping a GaugedForward onto its pre-export variant."""

    def __init__(self, gf):
        self.gf = gf

    def __enter__(self):
        self.gf.m.forward = self.gf.forward_fixed_scales

    def __exit__(self, *e):
        self.gf.m.forward = self.gf._forward
        return False


# ---------------------------------------------------------------------------
# Functional objective (section 12)
# ---------------------------------------------------------------------------
def func_loss(blk, x, t, kwargs, eps=1e-12):
    """||Block_FP(X) - Block_gauged(X)||_F^2 / (||Block_FP(X)||_F^2 + eps)."""
    y = blk(x, **kwargs)[0]
    num = (y.float() - t.float()).square().sum()
    den = t.float().square().sum() + eps
    return num / den


@torch.no_grad()
def func_loss_over(blk, X, T, kwargs, chunk=1):
    tot_n, tot_d = 0.0, 0.0
    for j in range(0, X.shape[0], chunk):
        x = X[j:j + chunk].to("cuda", non_blocking=True)
        t = T[j:j + chunk].to("cuda", non_blocking=True)
        y = blk(x, **kwargs)[0]
        tot_n += (y.float() - t.float()).square().sum().item()
        tot_d += t.float().square().sum().item()
    return tot_n / max(tot_d, 1e-30)


# ---------------------------------------------------------------------------
# The common Step 3 (section 7)
# ---------------------------------------------------------------------------
def common_step3(blk, qd, cal_in_d, cal_out_d, imp, kwargs, snapB, tag=""):
    """Restore Snapshot B, build a fresh AdamW, run the identical 1024-step
    Step 3 over the whole block, and assert the batch order matched Arm 0a's.

    Delegates to `nqx/testD.py::step3(mode="ste")`, which is byte-for-byte the
    routine that produced the reference `ctrl_ste` post = 0.114852.
    """
    C.GUARD.assert_not_heldout(cal_in_d, cal_out_d, where=f"common_step3[{tag}]")
    S.restore_snapshot_B(snapB)
    t0 = time.time()
    with S.PermRecorder(qd["num_calib_samples"]) as rec:
        flips = D.step3(blk, qd, cal_in_d, cal_out_d, imp, kwargs,
                        mode="ste", eta=0.0, beta_end=1.0)
    rec.assert_matches(snapB, where=f"common_step3[{tag}]")
    return {"flips": flips, "flip_mean": sum(flips.values()) / len(flips),
            "seconds": time.time() - t0}


# ---------------------------------------------------------------------------
# Metrics (section 8)
# ---------------------------------------------------------------------------
@torch.no_grad()
def J_table(blk, W_refs, names=NAMES, key="pre"):
    sub = nq_sub(blk)
    out = {}
    for n in names:
        H = I.prep_hin(f"model.layers.0.{n}", f"{CACHE}/hin", shrinkage=PAPER_GAMMA, device="cuda")
        j, f = I.recon_objectives(W_refs[n].cuda(), sub[n], H)
        out[n] = {f"J_{key}": j, f"fro_{key}": f}
        del H
    cleanup_memory()
    return out


@torch.no_grad()
def snapshot_signs(blk, names=NAMES):
    sub = nq_sub(blk)
    out = {}
    for n in names:
        m = sub[n]
        for attr in ["U_latent", "V_latent"]:
            src = getattr(m, attr, None)
            if src is None:
                src = getattr(m, attr.replace("_latent", ""))
            out[f"{n}.{attr}"] = C.pos_sign(src.data.float()).to(torch.int8).cpu()
    return out


@torch.no_grad()
def binary_of(blk, names=NAMES):
    """Current hard binary factors, as bf16, for the per-layer post table."""
    sub = nq_sub(blk)
    out = {}
    for n in names:
        m = sub[n]
        for attr in ["U_latent", "V_latent"]:
            src = getattr(m, attr, None)
            if src is None:
                src = getattr(m, attr.replace("_latent", ""))
            out[f"{n}.{attr}"] = C.pos_sign(src.data.float()).to(torch.bfloat16).cpu()
    return out


@torch.no_grad()
def layer_post_curve(blk, init_bin, tuned_bin, ho_in, ho_out, kwargs, names=NAMES):
    """Cumulative per-layer diagnostic: layers 1..k at their tuned signs, the rest
    back at their pre-Step-3 signs."""
    sub = nq_sub(blk)
    curve = {}
    for k, n in enumerate(names):
        for kk, nn_ in enumerate(names):
            m = sub[nn_]
            src = tuned_bin if kk <= k else init_bin
            m.V.data.copy_(src[f"{nn_}.V_latent"].to("cuda"))
            m.U.data.copy_(src[f"{nn_}.U_latent"].to("cuda"))
        curve[n] = I.block_err(blk, ho_in, ho_out, kwargs)
    for nn_ in names:
        m = sub[nn_]
        m.V.data.copy_(tuned_bin[f"{nn_}.V_latent"].to("cuda"))
        m.U.data.copy_(tuned_bin[f"{nn_}.U_latent"].to("cuda"))
    return curve


def sign_delta(a, b):
    return {k: (a[k] != b[k]).float().mean().item() for k in a}


def save_run(rec, name):
    os.makedirs(f"{GCACHE}/runs", exist_ok=True)
    S.atomic_json(rec, f"{GCACHE}/runs/{name}.json")
