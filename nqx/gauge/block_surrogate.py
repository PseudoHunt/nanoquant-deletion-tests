"""Surrogate scores S0..S4 for a discrete gauge move, plus their unit tests.

All scores predict the CHANGE in the true block error caused by replacing one
layer's deployed weight `W_current` (the current NanoQuant state) with
`W_candidate`.  A harmful move has true dE > 0, so a surrogate is *correct on
direction* when it also returns > 0.

    dW = W_candidate - W_current

    S0  ||X W_cand^T - X W_fp^T||^2  -  ||X W_cur^T - X W_fp^T||^2
        the existing layer reconstruction objective, as a delta.  Known to be
        wrong on all six endpoints.

    S1  tr(dW H_in dW^T)                      input-covariance only (GPTQ family)
    S2  N . tr(H_out dW H_in dW^T)            two-sided K-FAC curvature
    S3  <G_W, dW>                             block-residual linear term
    S4  S3 + lambda . S2                      block Taylor + K-FAC curvature

S1 and S2 are positive semi-definite by construction, so they can never predict
that a move *helps*; they can only rank harm.  S3 is the only term that knows
about the error already present in the other six quantised layers and can
therefore see a move destroying an existing cancellation.

Scaling.  `H_out` is an empirical Fisher (outer products of gradients), whose
scale differs from the Gauss-Newton factor by roughly ||residual||^2.  So the
natural magnitudes of S2 and S3 can be many decades apart and lambda must be
swept logarithmically; a sweep over {0.25..4} alone would only ever report
"S3 dominates" or "S2 dominates" without locating the crossover.
"""
import torch

from . import core as C


def _dbl(x):
    return x.double()


def s0_layer_recon_delta(W_cur, W_cand, W_fp, H_in_sum):
    """Layer reconstruction delta, ||X (W - W_fp)^T||^2, as a difference."""
    with C.no_tf32():
        a, b = _dbl(W_cur) - _dbl(W_fp), _dbl(W_cand) - _dbl(W_fp)
        return float((b * (b @ H_in_sum)).sum() - (a * (a @ H_in_sum)).sum())


def s1_input_quadratic(dW, H_in):
    with C.no_tf32():
        d = _dbl(dW)
        return float((d * (d @ H_in)).sum())


def s2_kfac_quadratic(dW, H_in, H_out, n_tokens):
    """N . tr(H_out dW H_in dW^T), with H_in/H_out the per-token means."""
    with C.no_tf32():
        d = _dbl(dW)
        return float(n_tokens * ((H_out @ d) * (d @ H_in)).sum())


def s3_block_linear(dW, G_W):
    with C.no_tf32():
        return float((_dbl(G_W) * _dbl(dW)).sum())


def s4(dW, G_W, H_in, H_out, n_tokens, lam):
    return s3_block_linear(dW, G_W) + lam * s2_kfac_quadratic(dW, H_in, H_out, n_tokens)


# ---------------------------------------------------------------------------
def test_h_in_identity(X, dW, H_in, n_tokens, tol=1e-6):
    """With H_out = I the quadratic must reproduce ||X dW^T||^2 / N.

    Guards the matrix orientation, which is the single easiest thing to get
    silently wrong here.
    """
    with C.no_tf32():
        d = _dbl(dW)
        lhs = float((d * (d @ H_in)).sum())
        rhs = float((_dbl(X) @ d.T).square().sum() / n_tokens)
    rel = abs(lhs - rhs) / max(abs(rhs), 1e-30)
    return {"quadratic": lhs, "direct": rhs, "rel_err": rel, "pass": bool(rel < tol)}


def test_psd(M, name, n_probe=8, seed=0):
    """Cheap PSD check by random probing plus a symmetry check."""
    g = torch.Generator(device=M.device); g.manual_seed(seed)
    vals = []
    for _ in range(n_probe):
        v = torch.randn(M.shape[0], generator=g, device=M.device, dtype=M.dtype)
        vals.append(float(v @ (M @ v)))
    sym = float((M - M.T).norm() / max(float(M.norm()), 1e-30))
    return {"name": name, "min_quadratic_form": min(vals), "n_probe": n_probe,
            "rel_asymmetry": sym, "finite": bool(torch.isfinite(M).all()),
            "pass": bool(min(vals) >= -1e-8 * abs(max(vals, key=abs)) and sym < 1e-10)}
