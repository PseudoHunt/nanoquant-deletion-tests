"""Stage 0 -- exactness and plumbing (brief sections 2, 3.1, 4, 5).

Nothing in Stage 1 runs until every assertion here passes.
"""
import json
import sys
import time

import torch

from nanoquant.modules.hub import NanoQuantConfigDataclass  # noqa: F401  import-order guard

import instrument as I
import testD as D
from nanoquant.core.compress_block import fused_weighted_mse
from nanoquant.optimi import AdamW

from . import core as C
from . import harness as H
from . import state as S

REPORT = {}


def _ok(name, cond, **extra):
    REPORT[name] = {"pass": bool(cond), **extra}
    status = "PASS" if cond else "FAIL"
    print(f"[stage0] {status:4s}  {name}  {extra}", flush=True)
    assert cond, f"stage 0 assertion failed: {name}  {extra}"


# ---------------------------------------------------------------------------
def test_no_s3(blk, ranks):
    """Section 1.2 / 2.2: does the reproduced state carry a rank-wise middle scale?"""
    desc = S.describe_state(blk, H.NAMES, ranks)
    any_mid = any(e["has_scale_mid"] for e in desc.values())
    REPORT["state_description"] = desc
    _ok("no_scale_mid_in_reproduced_state", not any_mid,
        note=("section 2.1 applies (U_R = U R, V_R = V R); section 2.2's "
              "fold -> rotate -> unfold is not instantiated and no s3 is manufactured"))
    for n, e in desc.items():
        assert e["U_latent_shape"] == [e["U_latent_shape"][0], e["rank"]], n
        assert e["V_latent_shape"][0] == e["rank"], n
    _ok("shape_convention_U_out_rank__V_rank_in", True,
        example={n: {"U": e["U_latent_shape"], "V": e["V_latent_shape"], "rank": e["rank"]}
                 for n, e in list(desc.items())[:2]})
    return desc


# ---------------------------------------------------------------------------
def test_gauge_exactness(bases, b=32, seed=1234):
    """Section 2.1: ||U_R V_R - U V||_F / ||U V||_F < 1e-6 in fp32, and
    R = I reproduces U, V exactly."""
    g = torch.Generator(device="cuda"); g.manual_seed(seed)
    rels, ident = {}, {}
    for name, base in bases.items():
        Rs = [C.haar_so(s, g) for s in C.block_sizes(base.rank, b)]
        U_R, V_R = C.gauge(base, Rs)
        with C.no_tf32():
            W0 = base.U0 @ base.V0
            WR = U_R @ V_R
            rel = (WR - W0).norm().item() / max(W0.norm().item(), 1e-30)
        rels[name] = rel
        Is = C.identity_Rs(base.rank, b, base.U0.device)
        Ui, Vi = C.gauge(base, Is)
        ident[name] = bool(torch.equal(Ui, base.U0) and torch.equal(Vi, base.V0))
        del U_R, V_R, W0, WR, Ui, Vi
        torch.cuda.empty_cache()
    _ok("gauge_product_invariance_lt_1e-6", max(rels.values()) < 1e-6, rel=rels)
    _ok("R_eq_I_gives_bit_identical_U_and_V", all(ident.values()), per_layer=ident)


# ---------------------------------------------------------------------------
def test_cayley(ranks_by_layer, b=32):
    """Section 3.1: Cayley blocks are orthogonal to 1e-5 and start at the identity."""
    orth, ident = {}, {}
    for name, rank in ranks_by_layer.items():
        cay = C.BlockCayley(rank, b=b, device="cuda")
        orth[f"{name}_at_zero"] = cay.orthogonality_error()
        Rs = cay.Rs()
        ident[name] = max((R - torch.eye(R.shape[0], device=R.device)).norm().item() for R in Rs)
        with torch.no_grad():
            for p in cay.groups:
                p.normal_(0, 0.3)
        orth[f"{name}_at_random"] = cay.orthogonality_error()
    _ok("cayley_orthogonality_lt_1e-5", max(orth.values()) < 1e-5, max_err=orth)
    _ok("cayley_zero_param_is_identity", max(ident.values()) < 1e-7, max_dev=ident)


# ---------------------------------------------------------------------------
def test_identity_export(blk, bases, b=32):
    """Section 4.3: Q_NQ(U, V) reproduces the cached post-ADMM export bit-for-bit."""
    sub = H.nq_sub(blk)
    res = {}
    for name, base in bases.items():
        m = sub[name]
        Is = C.identity_Rs(base.rank, b, base.U0.device)
        U_R, V_R = C.gauge(base, Is)
        B_U, B_V, sp, so = C.q_nq(U_R, V_R, base)
        C.assert_qnq_deterministic(U_R, V_R, base)
        # cached export: the module's own binary_ste and its own scale tensors
        cB_U = m.binary_ste(m.U_latent.data.float())
        cB_V = m.binary_ste(m.V_latent.data.float())
        res[name] = {
            "signs_U_identical": bool(torch.equal(B_U, cB_U)),
            "signs_V_identical": bool(torch.equal(B_V, cB_V)),
            "scale_pre_identical": bool(torch.equal(sp, m.scale_pre.data.float().view(-1))),
            "scale_post_identical": bool(torch.equal(so, m.scale_post.data.float().view(-1))),
        }
        del U_R, V_R, B_U, B_V, cB_U, cB_V
        torch.cuda.empty_cache()
    allok = all(all(v.values()) for v in res.values())
    _ok("identity_export_bit_for_bit", allok, per_layer=res)
    _ok("q_nq_deterministic_no_rng", True,
        note="repeated Q_NQ calls are bit-identical; the export path contains no SVID/RNG")


# ---------------------------------------------------------------------------
def test_gradient_path(blk, bases, cal_in, cal_out, kwargs, layer="mlp.down_proj", b=32):
    """Section 5: the functional loss must actually reach the Cayley parameters."""
    sub = H.nq_sub(blk)
    m = sub[layer]
    base = bases[layer]
    cay = C.BlockCayley(base.rank, b=b, device="cuda")
    gf = H.GaugedForward(m, base, cay)
    try:
        for _, mm in D.nq_modules(blk):
            for p in mm.parameters():
                p.requires_grad_(False)
        base.U0.requires_grad_(False)
        base.V0.requires_grad_(False)

        opt = torch.optim.Adam(cay.parameters(), lr=3e-3)
        R0 = [R.detach().clone() for R in cay.Rs()]

        x = cal_in[0:1].to("cuda")
        t = cal_out[0:1].to("cuda")
        C.GUARD.assert_not_heldout(x, t, where="stage0_gradient_path")
        loss = H.func_loss(blk, x, t, kwargs)
        loss.backward()

        grads_ok = any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max() > 0
                       for p in cay.parameters())
        gmax = max(float(p.grad.abs().max()) for p in cay.parameters() if p.grad is not None)
        _ok("cayley_receives_finite_nonzero_grad", grads_ok, loss=float(loss), max_abs_grad=gmax)
        _ok("frozen_base_U_has_no_grad", base.U0.grad is None)
        _ok("frozen_base_V_has_no_grad", base.V0.grad is None)

        opt_ids = {id(p) for grp in opt.param_groups for p in grp["params"]}
        cay_ids = {id(p) for p in cay.parameters()}
        module_ids = {id(p) for p in m.parameters()}
        _ok("optimizer_holds_only_cayley_params", opt_ids == cay_ids, n_params=len(opt_ids))
        _ok("no_export_scale_or_magnitude_is_an_optimizer_param",
            len(opt_ids & module_ids) == 0)

        opt.step()
        R1 = cay.Rs()
        moved = max((a - b_).norm().item() for a, b_ in zip(R1, R0))
        _ok("one_adam_step_moves_R", moved > 0, delta_R_fro=moved)
        _ok("R_still_orthogonal_after_step", cay.orthogonality_error() < 1e-5,
            orth_err=cay.orthogonality_error())
    finally:
        gf.remove()
        for p in cay.parameters():
            p.grad = None


# ---------------------------------------------------------------------------
def test_heldout_guard(ho_in, cal_in):
    C.GUARD.register(ho_in)
    caught = False
    try:
        C.GUARD.assert_not_heldout(ho_in, where="stage0_selftest")
    except AssertionError:
        caught = True
    _ok("heldout_guard_rejects_heldout_tensor", caught)
    C.GUARD.assert_not_heldout(cal_in, where="stage0_selftest")
    _ok("heldout_guard_accepts_calibration_tensor", True)


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    st = H.load_post_admm(seed=0)
    blk, ranks = st["blk"], st["ranks"]
    sub = H.nq_sub(blk)
    bases = {n: C.LayerBase(n, sub[n]).to("cuda") for n in H.NAMES}

    REPORT["admm_pre_cached"] = st["admm_pre"]
    REPORT["admm_pre_reloaded"] = I.block_err(blk, st["ho_in"], st["ho_out"], st["kwargs"])
    print(f"[stage0] post-ADMM held-out block error = {REPORT['admm_pre_reloaded']:.9f}", flush=True)

    desc = test_no_s3(blk, ranks)
    test_heldout_guard(st["ho_in"], st["cal_in"])
    test_gauge_exactness(bases)
    test_cayley({n: bases[n].rank for n in H.NAMES})
    test_identity_export(blk, bases)
    test_gradient_path(blk, bases, st["cal_in"], st["cal_out"], st["kwargs"])

    REPORT["seconds"] = time.time() - t0
    REPORT["ranks"] = {n: bases[n].rank for n in H.NAMES}
    REPORT["block_sizes_b32"] = {n: len(C.block_sizes(bases[n].rank, 32)) for n in H.NAMES}
    S.atomic_json(REPORT, f"{H.GCACHE}/stage0.json")
    print(f"[stage0] ALL PASS in {REPORT['seconds']:.0f}s")


if __name__ == "__main__":
    import os
    os.makedirs(H.GCACHE, exist_ok=True)
    main()
