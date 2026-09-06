"""Variant factorisation paths for Tests A and B.

Test A (Change 1): sequential binarisation -- keep NanoQuant's binary input
factor, refit the outer factor in closed form under H_in, then binarise it
column-by-column with GPTQ-style error compensation. Optional Change 4 refits
the three scale vectors by alternating least squares under H_in.

Test B (Change 2): the ADMM A-update solves against the full input covariance
expressed in NanoQuant's preconditioned coordinates, instead of against
X_A^T X_A. B-update and the rank1_approx sign projections are untouched.
"""
import argparse
import time

import torch
import torch.nn.functional as F

from nanoquant.core.admm_nq import RHO_SCHEDULER_REGISTRY, rank1_approx
from nanoquant.modules.linear import NanoQuantLinear
from nanoquant.utils.utils import cleanup_memory, find_layers, set_seed

import instrument as I


class no_tf32:
    """The H-multiplications are extra matmuls that upstream does not perform, so
    they run at full fp32: TF32 would inject ~1e-4 noise into an iteration that is
    chaotic at the 1e-7 level, making the H=I identity check meaningless."""
    def __enter__(self):
        self.prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    def __exit__(self, *a):
        torch.backends.cuda.matmul.allow_tf32 = self.prev


def _sign(x):
    s = x.sign()
    s[s == 0] = 1
    return s


# ---------------------------------------------------------------------------
# Test B: ADMM with full input covariance in the A-update
# ---------------------------------------------------------------------------
@torch.no_grad()
def _admm_solve_step_H(X, Y, Z, U, rho, reg, HY=None, H=None, eps=1e-12):
    """Solve (X^T H X + stab*I) F = X^T H Y + rho(Z-U).

    With H = I this is bit-identical in exact arithmetic to upstream
    _admm_solve_step, and reproduces it to <1e-6 numerically (unit-tested)."""
    orig_dtype = X.dtype
    X, Y, Z, U = (t.to(torch.float32) for t in (X, Y, Z, U))

    if H is None:
        HX = X
    else:
        with no_tf32():
            HX = H @ X
    system_matrix = X.mT @ HX
    system_matrix = 0.5 * (system_matrix + system_matrix.mT)
    diag_mean = system_matrix.diagonal().mean().abs()
    stabilizer = torch.clamp(rho * diag_mean + reg, min=eps)
    system_matrix.diagonal().add_(stabilizer)

    rhs = (X.mT @ (Y if HY is None else HY)) + rho * (Z - U)

    L, info = torch.linalg.cholesky_ex(system_matrix, upper=False)
    Factor = torch.cholesky_solve(rhs, L, upper=False) if info.item() == 0 \
        else torch.linalg.solve(system_matrix, rhs)
    return Factor.to(orig_dtype)


@torch.no_grad()
def factorize_admm_hin(W, i_norm, o_norm, mid_rank, H_in=None, outer_iters=400, inner_iters=5,
                       reg=3e-2, is_transpose=False, eps=1e-12, rho_scheduler='linear',
                       print_admm_steps=False, _h_side='A'):
    """Upstream factorize_admm_nanoquant with only the outer-factor update changed (Change 2).

    H_in is the covariance of the layer input (d_in x d_in). Upstream transposes
    any layer with d_out < d_in (k/v/down_proj) and solves the flipped problem;
    there the factor that carries the input-covariance weighting is the inner
    *B*-update, not the inner A-update. Both branches apply the identical
    operator H~ = D_in^-1 H_in D_in^-1 to the update of the output-side factor
    and leave the input-side factor and every rank1_approx projection stock.
    """
    if is_transpose:
        results = factorize_admm_hin(W.mT, o_norm, i_norm, mid_rank, H_in, outer_iters, inner_iters, reg,
                                     False, eps, rho_scheduler, print_admm_steps, _h_side='B')
        return {
            "W_final": results["W_final"].mT,
            "A": results["B"], "B": results["A"],
            "A_latent": results["B_latent"], "B_latent": results["A_latent"],
            "scale_pre": results["scale_post"], "scale_post": results["scale_pre"],
        }

    device = W.device
    out_features, in_features = W.shape

    norm_i = i_norm.sqrt().clamp(eps)
    norm_o = o_norm.sqrt().clamp(eps).unsqueeze(1)
    W_norm = W * norm_i.unsqueeze(0) * norm_o

    # H~ = D_in^-1 H_in D_in^-1 in NanoQuant's preconditioned coordinates. D_in is
    # the preconditioner on whichever index the covariance indexes: norm_i in the
    # direct branch, norm_o in the transposed branch (where they are the same
    # original i_norm, just swapped by the recursion).
    H_t, HY = None, None
    if H_in is not None:
        if _h_side == 'A' and H_in.shape[0] == in_features:
            d = norm_i.to(torch.float32)
            # divide rather than multiply by a rounded reciprocal: at H_in = D_in^2
            # this makes H~ bit-exactly the identity, so the H path reduces to
            # upstream's arithmetic exactly rather than to within 1e-7
            H_t = H_in.to(torch.float32) / (d.unsqueeze(0) * d.unsqueeze(1))
            H_t = 0.5 * (H_t + H_t.mT)
            with no_tf32():
                HY = H_t @ W_norm.mT.to(torch.float32)
        elif _h_side == 'B' and H_in.shape[0] == out_features:
            d = norm_o.squeeze(1).to(torch.float32)
            H_t = H_in.to(torch.float32) / (d.unsqueeze(0) * d.unsqueeze(1))
            H_t = 0.5 * (H_t + H_t.mT)
            with no_tf32():
                HY = H_t @ W_norm.to(torch.float32)
        else:
            raise ValueError(f"H_in shape {tuple(H_in.shape)} does not match side {_h_side} "
                             f"for W {tuple(W.shape)}")

    A_ls = torch.randn((out_features, mid_rank), device=device, dtype=W.dtype)
    B_ls = torch.randn((mid_rank, in_features), device=device, dtype=W.dtype)

    A_z, B_z = A_ls, B_ls
    if outer_iters > 0:
        A_z = rank1_approx(A_ls, inner_iters, eps)
        B_z = rank1_approx(B_ls, inner_iters, eps)

    A_u = A_ls - A_z
    B_u = B_ls - B_z

    rho_scheduler_func = RHO_SCHEDULER_REGISTRY[rho_scheduler]
    from nanoquant.core.admm_nq import _admm_solve_step

    for itt in range(outer_iters):
        rho = rho_scheduler_func(itt / outer_iters)

        # 1) X-update
        mid_norm_b = B_z.norm(dim=1).clamp(eps)
        X_A = B_z.mT / mid_norm_b
        if _h_side == 'A':
            A_ls = _admm_solve_step_H(X_A, W_norm.mT, A_z.mT, A_u.mT, rho, reg, HY=HY, H=H_t, eps=eps).mT
        else:
            A_ls = _admm_solve_step(X_A, W_norm.mT, A_z.mT, A_u.mT, rho, reg, eps).mT

        mid_norm_a = A_z.norm(dim=0).clamp(eps)
        X_B = A_z / mid_norm_a
        if _h_side == 'B':
            B_ls = _admm_solve_step_H(X_B, W_norm, B_z, B_u, rho, reg, HY=HY, H=H_t, eps=eps)
        else:
            B_ls = _admm_solve_step(X_B, W_norm, B_z, B_u, rho, reg, eps)

        # 2) Z-update -- sign projection stays in the original basis
        A_z = rank1_approx(A_ls + A_u, inner_iters, eps)
        B_z = rank1_approx(B_ls + B_u, inner_iters, eps)

        # 3) U-update
        A_u.add_(A_ls - A_z)
        B_u.add_(B_ls - B_z)

    A_unbalanced = A_z / norm_o
    B_unbalanced = B_z / norm_i
    A_latent_unb = (A_ls + A_u) / norm_o
    B_latent_unb = (B_ls + B_u) / norm_i

    balance_factor = (B_unbalanced.norm().clamp(eps) / A_unbalanced.norm().clamp(eps)).sqrt()
    A_final = A_unbalanced * balance_factor
    B_final = B_unbalanced / balance_factor
    A_latent = A_latent_unb * balance_factor
    B_latent = B_latent_unb / balance_factor

    if outer_iters > 0:
        A_final = A_final * (1.0 / A_z.norm(dim=0).clamp(eps))

    return {
        "W_final": F.linear(A_final, B_final.mT),
        "A": A_final.mT, "B": B_final,
        "A_latent": A_latent.mT, "B_latent": B_latent,
        "scale_pre": B_final.abs().mean(dim=0).view(1, -1),
        "scale_post": A_final.abs().mean(dim=1).view(1, -1),
    }


# ---------------------------------------------------------------------------
# Test A: sequential binarisation with refit + rank-wise compensation (Change 1)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _gptq_binarize_columns(A, H_r, s1, s3, blocksize=128):
    """Binarise A (d_out x r) column by column with GPTQ-style compensation.

    Columns are ordered by diag(H_r) descending; for column k the quantised
    column is u_k = s1 * s3[k] * sign(a_k) and the residual is propagated to the
    remaining columns through the standard Cholesky-of-inverse form.
    s1 and s3 are fixed for the whole loop."""
    d_out, r = A.shape
    perm = torch.argsort(H_r.diagonal(), descending=True)
    invperm = torch.argsort(perm)
    Hp = H_r[perm][:, perm].contiguous()
    Ap = A[:, perm].clone().float()
    s3p = s3[perm]

    L, info = torch.linalg.cholesky_ex(Hp)
    if info.item() != 0:                       # rare: extra jitter then retry
        Hp.diagonal().add_(1e-4 * Hp.diagonal().mean().abs())
        L = torch.linalg.cholesky(Hp)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(0.5 * (Hinv + Hinv.mT), upper=True)

    Ub = torch.zeros_like(Ap)
    for i1 in range(0, r, blocksize):
        i2 = min(i1 + blocksize, r)
        cnt = i2 - i1
        W1 = Ap[:, i1:i2].clone()
        U1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(cnt):
            w = W1[:, i]
            u = _sign(w)
            q = s1 * s3p[i1 + i] * u
            U1[:, i] = u
            err = (w - q) / Hi[i, i]
            W1[:, i:] -= err.unsqueeze(1) * Hi[i, i:].unsqueeze(0)
            E1[:, i] = err
        Ub[:, i1:i2] = U1
        if i2 < r:
            Ap[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Ub[:, invperm]


@torch.no_grad()
def _als_refit_scales(W, H, U_bin, V_b, s1, mid, s2, rounds=3, jitter=1e-4):
    """Change 4: refit (s1, mid, s2) by alternating least squares under H_in,
    with both binary factors held fixed. Each sub-problem is a closed-form
    weighted least squares."""
    eps = 1e-12
    WH = W @ H
    for _ in range(rounds):
        # (a) s1 : rows decouple
        M = U_bin @ (mid.unsqueeze(1) * V_b * s2.unsqueeze(0))
        MH = M @ H
        s1 = ((W * MH).sum(1) / (M * MH).sum(1).clamp(min=eps))
        # (b) mid : r x r solve,  (P^T P) . (G H G^T) s = diag-terms
        P = s1.unsqueeze(1) * U_bin
        G = V_b * s2.unsqueeze(0)
        GH = G @ H
        Msys = (P.mT @ P) * (GH @ G.mT)
        rhs = ((P.mT @ WH) * G).sum(1)
        Msys = 0.5 * (Msys + Msys.mT)
        Msys.diagonal().add_(jitter * Msys.diagonal().mean().abs().clamp(min=eps))
        mid = torch.linalg.solve(Msys, rhs)
        # (c) s2 : d_in x d_in solve,  (H . C^T C) s = diag(C^T W H)
        C = P @ (mid.unsqueeze(1) * V_b)
        Msys2 = H * (C.mT @ C)
        rhs2 = (C * WH).sum(0)
        Msys2 = 0.5 * (Msys2 + Msys2.mT)
        Msys2.diagonal().add_(jitter * Msys2.diagonal().mean().abs().clamp(min=eps))
        s2 = torch.linalg.solve(Msys2, rhs2)
    return s1, mid, s2


@torch.no_grad()
def factorize_testA(layer, name, rank, quant_config, ctx=None, key=None, hin_key=None, opts=None):
    opts = opts or {}
    eps = 1e-12
    set_seed(quant_config['seed'])
    lx = find_layers(layer)[name]
    W_orig = lx.weight.data.clone()
    device = "cuda"
    W = W_orig.to(device)
    is_transpose = W.shape[0] < W.shape[1]

    from nanoquant.core.admm_nq import factorize_admm_nanoquant
    t0 = time.time()
    admm_fn = opts.get("_admm_fn")
    if admm_fn is None:
        fr = factorize_admm_nanoquant(
            W, lx.i_norm.to(device), lx.o_norm.to(device), mid_rank=rank,
            outer_iters=quant_config['admm_outer_iters'], inner_iters=quant_config['admm_inner_iters'],
            is_transpose=is_transpose, rho_scheduler=quant_config['admm_penalty_scheduler'],
            print_admm_steps=quant_config['admm_print_steps'])
    else:
        fr = admm_fn(layer, name, rank, quant_config, ctx, key, hin_key, W, lx, is_transpose)
    admm_time = time.time() - t0

    t1 = time.time()
    H = I.prep_hin(hin_key, ctx.hin_dir, shrinkage=quant_config['calib_shrinkage'],
                   damp=opts.get("hin_damp", 0.01), device=device)

    Wf = W.float()
    A_f = fr["A"].mT.float().contiguous()     # (d_out, r)  their outer factor
    B_f = fr["B"].float().contiguous()        # (r, d_in)   their input factor
    s2 = fr["scale_pre"].float().view(-1)     # (d_in,)  their column scale

    # 1) binary input factor + its rank-1 magnitude field; Vt is exactly what the
    #    kernel sees:  (x * s2) @ V_b^T  then  * s_mid_B
    V_b = _sign(B_f)
    u_mag = B_f.abs().mean(dim=1)
    s_mid_B = u_mag / u_mag.mean().clamp(min=eps)
    Vt = s_mid_B.unsqueeze(1) * V_b * s2.unsqueeze(0)
    rel = (Vt - B_f).norm() / B_f.norm().clamp(min=eps)

    # 2) closed-form refit of the outer factor under H_in
    G = Vt @ H
    Hr = G @ Vt.mT
    Hr = 0.5 * (Hr + Hr.mT)
    delta = opts.get("delta", 0.01) * Hr.diagonal().mean().abs().clamp(min=eps)
    Hr.diagonal().add_(delta)
    rhs = Wf @ G.mT
    Lc, info = torch.linalg.cholesky_ex(Hr)
    A_star = (torch.cholesky_solve(rhs.mT, Lc)).mT if info.item() == 0 else \
        torch.linalg.solve(Hr, rhs.mT).mT

    def _obj(A):
        M = Wf - A @ Vt
        return (M * (M @ H)).sum().item()

    J_admm, J_star = _obj(A_f), _obj(A_star)

    # 3) column-wise binarisation with GPTQ-style compensation
    s1 = A_star.abs().mean(dim=1)
    coln = A_star.abs().mean(dim=0)
    s3 = coln / coln.mean().clamp(min=eps)
    U_bin = _gptq_binarize_columns(A_star, Hr, s1, s3, blocksize=opts.get("blocksize", 128))

    mid = s_mid_B * s3
    J_bin = _obj(s1.unsqueeze(1) * U_bin @ (s3.unsqueeze(1) * Vt))

    # 4) optional Change 4: ALS refit of the three scale vectors
    J_als = None
    if opts.get("als_rounds", 0) > 0:
        s1, mid, s2 = _als_refit_scales(Wf, H, U_bin, V_b, s1, mid, s2,
                                        rounds=int(opts["als_rounds"]))
        J_als = _obj(s1.unsqueeze(1) * U_bin @ (mid.unsqueeze(1) * V_b * s2.unsqueeze(0)))
    change_time = time.time() - t1

    W_final = s1.unsqueeze(1) * (U_bin @ (mid.unsqueeze(1) * V_b * s2.unsqueeze(0)))

    # latents for Step 3: our sign decisions, rescaled to their latent magnitude
    # scale so that STE flip dynamics stay comparable to the baseline
    A_lat_ref, B_lat_ref = fr["A_latent"].float(), fr["B_latent"].float()
    Am = A_star.abs()
    Am = Am / Am.mean().clamp(min=eps) * A_lat_ref.abs().mean()
    Bm = B_f.abs()
    Bm = Bm / Bm.mean().clamp(min=eps) * B_lat_ref.abs().mean()

    out = {
        "W_final": W_final.to(W.dtype),
        "A": U_bin.mT.to(W.dtype), "B": V_b.to(W.dtype),
        "A_latent": (Am * U_bin).mT.to(W.dtype), "B_latent": (Bm * V_b).to(W.dtype),
        "scale_pre": s2.view(1, -1).to(W.dtype),
        "scale_post": s1.view(1, -1).to(W.dtype),
        "scale_mid": mid.view(1, -1).to(W.dtype),
    }

    if ctx is not None:
        ent = ctx.rec["layers"].setdefault(key, {})
        ent.update({"J_admm": J_admm, "J_refit": J_star, "J_binarized": J_bin, "J_als": J_als,
                    "refit_gain": (J_admm - J_star) / max(J_admm, 1e-30),
                    "vtilde_rel_err": rel.item(), "admm_s": admm_time, "change1_s": change_time})

    new_module = lx
    new_module.__class__ = NanoQuantLinear
    new_module.__quant_convert__(do_train=quant_config['tune_fact'], rank=rank,
                                 factor_results=argparse.Namespace(**out))
    if not quant_config['tune_fact'] and getattr(new_module, "bias", None) is not None:
        pass

    err = (W_final - Wf).square().sum().item()
    nrm = Wf.square().sum().item()
    print(f"\t\tADMM weight recon error: raw={err:.4f}, norm={err/max(nrm,1e-30):.4f}, "
          f"per_el={err/W.numel():.4e}, ADMM time={admm_time:.2f}s | "
          f"[A] Vt_relerr={rel.item():.2e} J_admm={J_admm:.4e} J_refit={J_star:.4e} "
          f"J_bin={J_bin:.4e}" + (f" J_als={J_als:.4e}" if J_als is not None else "")
          + f" change1={change_time:.2f}s")
    del W, Wf, H, G, Hr, A_star
    cleanup_memory()
    return new_module, argparse.Namespace(**out)


# ---------------------------------------------------------------------------
# Test B wrapper + dispatcher
# ---------------------------------------------------------------------------
@torch.no_grad()
def factorize_testB(layer, name, rank, quant_config, ctx=None, key=None, hin_key=None, opts=None):
    """Upstream factorize_and_replace with only the ADMM call swapped for the
    H_in-weighted variant (Change 2). Everything downstream is untouched."""
    opts = opts or {}
    set_seed(quant_config['seed'])
    lx = find_layers(layer)[name]
    W_orig = lx.weight.data.clone()
    device = "cuda"
    W = W_orig.to(device)
    is_transpose = W.shape[0] < W.shape[1]

    H = I.prep_hin(hin_key, ctx.hin_dir, shrinkage=quant_config['calib_shrinkage'],
                   damp=opts.get("hin_damp", 0.01), device=device)
    t0 = time.time()
    fr = factorize_admm_hin(
        W, lx.i_norm.to(device), lx.o_norm.to(device), mid_rank=rank, H_in=H,
        outer_iters=quant_config['admm_outer_iters'], inner_iters=quant_config['admm_inner_iters'],
        is_transpose=is_transpose, rho_scheduler=quant_config['admm_penalty_scheduler'],
        print_admm_steps=quant_config['admm_print_steps'])
    admm_time = time.time() - t0

    new_module = lx
    new_module.__class__ = NanoQuantLinear
    new_module.__quant_convert__(do_train=quant_config['tune_fact'], rank=rank,
                                 factor_results=argparse.Namespace(**fr))

    err = (fr["W_final"].float() - W.float()).square().sum().item()
    nrm = W.float().square().sum().item()
    print(f"\t\tADMM weight recon error: raw={err:.4f}, norm={err/max(nrm,1e-30):.4f}, "
          f"per_el={err/W.numel():.4e}, ADMM time={admm_time:.2f}s [B: H_in-weighted]")
    if ctx is not None:
        ctx.rec["layers"].setdefault(key, {}).update({"admm_s": admm_time, "recon_norm": err / max(nrm, 1e-30)})
    del W, H
    cleanup_memory()
    return new_module, argparse.Namespace(**fr)


def _admm_hin_for_A(layer, name, rank, quant_config, ctx, key, hin_key, W, lx, is_transpose):
    """A+B: Change 1 applied on top of the Change 2 ADMM."""
    H = I.prep_hin(hin_key, ctx.hin_dir, shrinkage=quant_config['calib_shrinkage'], device=W.device)
    fr = factorize_admm_hin(
        W, lx.i_norm.to(W.device), lx.o_norm.to(W.device), mid_rank=rank, H_in=H,
        outer_iters=quant_config['admm_outer_iters'], inner_iters=quant_config['admm_inner_iters'],
        is_transpose=is_transpose, rho_scheduler=quant_config['admm_penalty_scheduler'],
        print_admm_steps=quant_config['admm_print_steps'])
    del H
    return fr


def get_factorize_fn(variant, opts):
    from functools import partial
    if variant == "testA":
        return partial(factorize_testA, opts=opts)
    if variant == "testB":
        return partial(factorize_testB, opts=opts)
    if variant == "testAB":
        o = dict(opts)
        o["_admm_fn"] = _admm_hin_for_A
        return partial(factorize_testA, opts=o)
    raise ValueError(f"unknown variant {variant}")
