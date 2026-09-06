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

    HX = X if H is None else (H @ X)
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
            inv = (1.0 / norm_i).to(torch.float32)
            H_t = H_in.to(torch.float32) * inv.unsqueeze(0) * inv.unsqueeze(1)
            H_t = 0.5 * (H_t + H_t.mT)
            HY = H_t @ W_norm.mT.to(torch.float32)
        elif _h_side == 'B' and H_in.shape[0] == out_features:
            inv = (1.0 / norm_o.squeeze(1)).to(torch.float32)
            H_t = H_in.to(torch.float32) * inv.unsqueeze(0) * inv.unsqueeze(1)
            H_t = 0.5 * (H_t + H_t.mT)
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
