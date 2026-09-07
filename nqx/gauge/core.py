"""Gauge-NQ core: gauge map, block-Cayley rotations, and the deterministic
NanoQuant export operator Q_NQ.

Parameterisation actually present in the reproduced D2 state
--------------------------------------------------------------------------
`nanoquant.core.admm_nq.factorize_admm_nanoquant` returns no `scale_mid`, and
`NanoQuantLinear._setup_path` only creates one when `factor_results.scale_mid`
exists.  The reproduced post-ADMM block therefore has **no rank-wise middle
scale (no s3)**, so the simple gauge map of brief section 2.1 applies and
section 2.2's fold -> rotate -> unfold is not instantiated.  Nothing here
manufactures an s3.

Storage convention (verified against `linear.py::_setup_path`):

    stored U_latent : [out,  rank]     (= A_latent.mT)
    stored V_latent : [rank, in ]      (= B_latent)

Mathematical convention used by the brief:

    U : [out,  rank]   ==  stored U_latent
    V : [in,   rank]   ==  stored V_latent.T

Deployed representation (verified against `linear.py::_compute_forward` with
`scale_mid is None`):

    W_eff = diag(scale_post) . sign(U_latent) . sign(V_latent) . diag(scale_pre)

Export statistics.  NanoQuant's own rule is *mean magnitude* over the rank axis
(`admm_nq.py`: `scale_pre = B_final.abs().mean(dim=0)`,
`scale_post = A_final.abs().mean(dim=1)`).  Those two vectors are extracted from
`A_final`/`B_final`, which are the rank-1-projected `A_z`/`B_z` and are *not*
recoverable from the stored latents alone.  Q_NQ therefore applies NanoQuant's
mean-magnitude rule to the rotated latents *relative to the base latents*:

    scale_post(R) = scale_post_0 * rowmean|U R| / rowmean|U|
    scale_pre (R) = scale_pre_0  * colmean|V_R| / colmean|V|

which (a) is exactly NanoQuant's rule up to a fixed per-coordinate constant that
absorbs the A_z-vs-A_latent discrepancy, and (b) reproduces the cached ADMM
export *bit-for-bit* at R = I, because the ratio is exactly 1.0 there.  This is
what brief section 4.2 asks for (export statistics recomputed from the current
rotated factors, then detached) while satisfying the section 4.3 identity gate.

Determinism.  There is no SVID / `power_iteration` call anywhere on the export
path -- `rank1_approx` is used only *inside* the ADMM outer loop.  Q_NQ is
therefore a deterministic function of (U_R, V_R) with no RNG dependence at all;
`assert_qnq_deterministic` proves it by bit-comparing repeated calls.  Snapshot A
is still captured (see `state.py`) so the claim is auditable.
"""
import contextlib

import torch
import torch.nn as nn


def rank_absmean(x, dim):
    """NanoQuant's mean-magnitude export statistic, accumulated in fp64.

    fp32 `mean(dim=...)` on CUDA is *not* bit-reproducible across two tensors
    that hold identical values at different addresses: the reduction is split
    differently and a handful of coordinates come out ~1e-7 apart.  That is
    enough to break the section 4.3 identity gate, because the export ratio
    `m(R)/m(I)` then misses 1.0 by an ulp.  Accumulating in fp64 puts the
    layout-dependent discrepancy at ~1e-16, so the ratio rounds to exactly 1.0
    in fp32 while a genuine rotation's ratio is unaffected.
    """
    return x.abs().sum(dim=dim, dtype=torch.float64) / x.shape[dim]

# ---------------------------------------------------------------------------
# TF32.  `nanoquant.core.admm_nq` switches TF32 *on* at import time, and the
# reproduced baseline ran with it on, so the ambient setting is never changed
# for anything on the NanoQuant path (ADMM, Step 3, block forwards).  It is
# switched off only around the fp32 gauge algebra, where a TF32-rounded matmul
# would destroy the R = I exactness gate (NOTES.md#2 records the same trap).
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def no_tf32():
    prev_mm = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_mm
        torch.backends.cudnn.allow_tf32 = prev_cudnn


def block_sizes(rank, b):
    """Rank split into blocks of size `b`, with a smaller final block if `b`
    does not divide `rank` (rank 992 with b = 64, for example)."""
    sizes = [b] * (rank // b)
    if rank % b:
        sizes.append(rank % b)
    assert sum(sizes) == rank
    return sizes


def pos_sign(x):
    """NanoQuant's `binary_ste` sign: sign(0) and sign(-0.0) are both +1."""
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


# ---------------------------------------------------------------------------
# 1. Frozen base state
# ---------------------------------------------------------------------------
class LayerBase:
    """The immutable post-ADMM factors of one NanoQuantLinear, in fp32.

    U0/V0 are exact upcasts of the module's bf16 latents, so a bf16 round-trip
    of an unrotated factor is the identity.
    """

    def __init__(self, name, module):
        assert not hasattr(module, "scale_mid"), (
            f"{name}: the reproduced state carries a scale_mid; section 2.2 "
            "would apply and this build does not implement it")
        assert module.do_train and not module._binarized, f"{name}: not a live post-ADMM module"
        self.name = name
        self.U0 = module.U_latent.data.detach().float().clone()      # [out, rank]
        self.V0 = module.V_latent.data.detach().float().clone()      # [rank, in]
        self.sp0 = module.scale_pre.data.detach().float().clone().view(-1)    # [in]
        self.so0 = module.scale_post.data.detach().float().clone().view(-1)   # [out]
        self.out, self.rank = self.U0.shape
        assert self.V0.shape[0] == self.rank
        self.inn = self.V0.shape[1]
        # base export statistics (NanoQuant's mean-magnitude rule, over rank)
        self.mU0 = rank_absmean(self.U0, 1)                           # [out], fp64
        self.mV0 = rank_absmean(self.V0, 0)                           # [in],  fp64
        assert (self.mU0 > 0).all(), f"{name}: zero row in U0, export ratio undefined"
        assert (self.mV0 > 0).all(), f"{name}: zero column in V0, export ratio undefined"
        self.U0.requires_grad_(False)
        self.V0.requires_grad_(False)

    def to(self, device):
        for a in ["U0", "V0", "sp0", "so0", "mU0", "mV0"]:
            setattr(self, a, getattr(self, a).to(device))
        return self


# ---------------------------------------------------------------------------
# 2. Gauge map (section 2.1: no s3, so U_R = U R and V_R = V R)
# ---------------------------------------------------------------------------
def apply_R_U(U0, Rs):
    """U0 [out, rank] -> U0 R, with Rs a list of [b_i, b_i] orthogonal blocks."""
    out = U0.shape[0]
    parts, off = [], 0
    for R in Rs:
        b = R.shape[0]
        parts.append(U0[:, off:off + b] @ R)
        off += b
    assert off == U0.shape[1]
    return torch.cat(parts, dim=1)


def apply_R_V(V0, Rs):
    """V0 [rank, in] -> stored form of (V R), i.e. R^T V0 blockwise.

    math V = V0.T, math V_R = V R, stored V_R = (V0.T R).T = R.T V0.
    """
    parts, off = [], 0
    for R in Rs:
        b = R.shape[0]
        parts.append(R.transpose(0, 1) @ V0[off:off + b, :])
        off += b
    assert off == V0.shape[0]
    return torch.cat(parts, dim=0)


def gauge(base, Rs):
    """(U_R, V_R) in stored convention, fp32, TF32 disabled."""
    with no_tf32():
        return apply_R_U(base.U0, Rs), apply_R_V(base.V0, Rs)


def identity_Rs(rank, b, device, dtype=torch.float32):
    return [torch.eye(s, device=device, dtype=dtype) for s in block_sizes(rank, b)]


# ---------------------------------------------------------------------------
# 3. Q_NQ -- the one deterministic NanoQuant export operator
# ---------------------------------------------------------------------------
def q_nq(U_R, V_R, base, ste=False):
    """NanoQuant's conversion of a latent factor pair to the deployed state.

    Returns (B_U, B_V, scale_pre, scale_post).  Scales are always detached
    (brief 4.2).  With `ste=True` the binary factors carry a straight-through
    gradient to (U_R, V_R) -- this is NanoQuant's own `binary_ste`.
    """
    with no_tf32():
        sU, sV = pos_sign(U_R), pos_sign(V_R)
        if ste:
            B_U = (sU - U_R).detach() + U_R
            B_V = (sV - V_R).detach() + V_R
        else:
            B_U, B_V = sU, sV
        mU = rank_absmean(U_R.detach(), 1)
        mV = rank_absmean(V_R.detach(), 0)
        scale_post = (base.so0 * (mU / base.mU0).float()).detach()
        scale_pre = (base.sp0 * (mV / base.mV0).float()).detach()
    return B_U, B_V, scale_pre, scale_post


def effective_W(B_U, B_V, scale_pre, scale_post):
    """diag(scale_post) B_U B_V diag(scale_pre)."""
    with no_tf32():
        return (B_U @ (B_V * scale_pre.unsqueeze(0))) * scale_post.unsqueeze(1)


def assert_qnq_deterministic(U_R, V_R, base):
    a = q_nq(U_R, V_R, base)
    b = q_nq(U_R, V_R, base)
    for x, y, nm in zip(a, b, ["B_U", "B_V", "scale_pre", "scale_post"]):
        assert torch.equal(x, y), f"Q_NQ is not deterministic in {nm}"


# ---------------------------------------------------------------------------
# 4. Block-Cayley rotation (section 3.1)
# ---------------------------------------------------------------------------
class BlockCayley(nn.Module):
    """R_i = (I - A_i)(I + A_i)^{-1} for skew A_i, one per rank block.

    A_i is stored as a free matrix `a_i`; the skew part is `a - a^T`, so
    a = 0 gives A = 0 and R = I.  Solved with torch.linalg.solve; no inverse is
    ever materialised.
    """

    def __init__(self, rank, b=32, device="cuda", dtype=torch.float32):
        super().__init__()
        self.sizes = block_sizes(rank, b)
        self.rank = rank
        self.b = b
        # one parameter per distinct block size keeps the batched solve fast
        self.groups = nn.ParameterList()
        self._group_sizes = []
        for s in sorted(set(self.sizes)):
            n = self.sizes.count(s)
            self.groups.append(nn.Parameter(torch.zeros(n, s, s, device=device, dtype=dtype)))
            self._group_sizes.append(s)

    def Rs(self):
        """List of [b_i, b_i] orthogonal matrices in rank-block order."""
        built = {}
        for p, s in zip(self.groups, self._group_sizes):
            A = p - p.transpose(-1, -2)
            I = torch.eye(s, device=p.device, dtype=p.dtype).expand_as(A)
            # R (I + A) = (I - A)  =>  (I + A)^T R^T = (I - A)^T
            Rt = torch.linalg.solve((I + A).transpose(-1, -2), (I - A).transpose(-1, -2))
            built[s] = list(Rt.transpose(-1, -2))
        order = {s: 0 for s in built}
        out = []
        for s in self.sizes:
            out.append(built[s][order[s]])
            order[s] += 1
        return out

    @torch.no_grad()
    def orthogonality_error(self):
        m = 0.0
        for R in self.Rs():
            I = torch.eye(R.shape[0], device=R.device, dtype=R.dtype)
            m = max(m, (R.transpose(0, 1) @ R - I).norm().item())
        return m


def haar_so(n, generator, device="cuda", dtype=torch.float32):
    """Haar-random SO(n): Gaussian -> QR -> sign-correct the R diagonal ->
    flip one column if det < 0."""
    A = torch.randn(n, n, generator=generator, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    d = torch.diagonal(R)
    Q = Q * torch.sign(torch.where(d == 0, torch.ones_like(d), d)).unsqueeze(0)
    if torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


# ---------------------------------------------------------------------------
# 5. Held-out guard (section 8)
# ---------------------------------------------------------------------------
class HeldoutGuard:
    """Records the exact byte range of the held-out split and refuses to let any
    tensor overlapping it be referenced by an optimiser loop.

    Byte ranges rather than storages: the calibration and held-out splits are
    disjoint slices `x[:128]` / `x[128:]` of one cached tensor and therefore
    share a storage, so a storage-level check would reject the calibration data
    as well.
    """

    def __init__(self):
        self.ranges = []

    @staticmethod
    def _span(t):
        assert t.is_contiguous(), "held-out guard needs contiguous tensors"
        start = t.data_ptr()
        return start, start + t.numel() * t.element_size()

    def register(self, *tensors):
        for t in tensors:
            self.ranges.append(self._span(t))
        return self

    def assert_not_heldout(self, *tensors, where=""):
        for t in tensors:
            lo, hi = self._span(t)
            for a, b in self.ranges:
                assert hi <= a or lo >= b, \
                    f"held-out tensor reached an optimiser path at {where}"


GUARD = HeldoutGuard()
