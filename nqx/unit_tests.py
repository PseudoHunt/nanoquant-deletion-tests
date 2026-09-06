"""Unit tests for the Test A / Test B changes."""
import sys
import torch

from nanoquant.modules.hub import NanoQuantConfigDataclass  # noqa: F401  import-order guard
from nanoquant.core.admm_nq import factorize_admm_nanoquant
from nanoquant.utils.utils import set_seed

import variants as V

DEV = "cuda"
FAIL = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        FAIL.append(name)


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-30)).item()


def test_admm_identity(d_out, d_in, r, iters=60):
    """Test B unit test: at H_in = D_in^2 the H-weighted A-update must reproduce
    upstream's iterates."""
    torch.manual_seed(7)
    W = torch.randn(d_out, d_in, device=DEV, dtype=torch.bfloat16)
    i_norm = torch.rand(d_in, device=DEV) + 0.5
    o_norm = torch.rand(d_out, device=DEV) + 0.5
    is_t = d_out < d_in
    # H_in = D_in^2 exactly, i.e. diag(norm_i * norm_i) with norm_i = sqrt(i_norm)
    # as the code forms it. Building it as diag(i_norm) instead leaves a ~1e-7
    # rounding gap that TF32 in the rhs matmul amplifies to ~1e-4, which the
    # chaotic sign() projection then blows up (see ut2.py).
    norm_i = i_norm.sqrt().clamp(1e-12)
    H = torch.diag(norm_i * norm_i).float()

    set_seed(0)
    ref = factorize_admm_nanoquant(W, i_norm, o_norm, r, outer_iters=iters, inner_iters=5,
                                   is_transpose=is_t, rho_scheduler='linear')
    set_seed(0)
    got = V.factorize_admm_hin(W, i_norm, o_norm, r, H_in=H, outer_iters=iters, inner_iters=5,
                               is_transpose=is_t, rho_scheduler='linear')
    tag = f"{d_out}x{d_in} r={r} transpose={is_t}"
    for k in ["W_final", "A", "B", "scale_pre", "scale_post"]:
        e = rel(got[k].float(), ref[k].float())
        check(f"admm identity [{k}] {tag}", e < 1e-6, f"rel={e:.3e}")


def test_admm_H_changes_result(d_out=96, d_in=128, r=32, iters=60):
    """Guard against a silent no-op: a non-diagonal H must change the solution."""
    torch.manual_seed(3)
    W = torch.randn(d_out, d_in, device=DEV, dtype=torch.bfloat16)
    i_norm = torch.rand(d_in, device=DEV) + 0.5
    o_norm = torch.rand(d_out, device=DEV) + 0.5
    X = torch.randn(4096, d_in, device=DEV)
    H = (X.mT @ X) / X.shape[0]
    H = H * (i_norm / H.diagonal()).sqrt().unsqueeze(0) * (i_norm / H.diagonal()).sqrt().unsqueeze(1)
    set_seed(0)
    ref = factorize_admm_nanoquant(W, i_norm, o_norm, r, outer_iters=iters, is_transpose=False,
                                   rho_scheduler='linear')
    set_seed(0)
    got = V.factorize_admm_hin(W, i_norm, o_norm, r, H_in=H, outer_iters=iters, is_transpose=False,
                               rho_scheduler='linear')
    e = rel(got["W_final"].float(), ref["W_final"].float())
    check("admm with full H differs from stock", e > 1e-3, f"rel={e:.3e}")

    # and it should lower the H-weighted objective it is optimising
    def J(Wh):
        M = W.float() - Wh.float()
        return (M * (M @ H)).sum().item()
    j_ref, j_got = J(ref["W_final"]), J(got["W_final"])
    check("admm with full H lowers tr((W-W^)H(.)^T)", j_got < j_ref, f"stock={j_ref:.5e} H={j_got:.5e}")


def test_refit_lowers_objective(d_out=256, d_in=192, r=64):
    """Test A step 2 unit test: A* strictly lowers tr((W - A V~) H (.)^T) vs A_z."""
    torch.manual_seed(11)
    W = torch.randn(d_out, d_in, device=DEV, dtype=torch.bfloat16)
    i_norm = torch.rand(d_in, device=DEV) + 0.5
    o_norm = torch.rand(d_out, device=DEV) + 0.5
    set_seed(0)
    fr = factorize_admm_nanoquant(W, i_norm, o_norm, r, outer_iters=120, is_transpose=False,
                                  rho_scheduler='linear')
    X = torch.randn(8192, d_in, device=DEV)
    H = (X.mT @ X) / X.shape[0]
    H.diagonal().add_(0.01 * H.diagonal().mean())

    Wf = W.float()
    A_f = fr["A"].mT.float()
    B_f = fr["B"].float()
    s2 = fr["scale_pre"].float().view(-1)
    V_b = V._sign(B_f)
    u_mag = B_f.abs().mean(dim=1)
    s_mid = u_mag / u_mag.mean()
    Vt = s_mid.unsqueeze(1) * V_b * s2.unsqueeze(0)
    # V~ reproduces their real B factor to bf16 precision: |B| is exactly rank-1
    # in exact arithmetic, so the only gap is the bf16 rounding of B itself.
    check("V~ equals their real B factor to bf16 precision", rel(Vt, B_f) < 5e-3,
          f"rel={rel(Vt, B_f):.3e}")

    G = Vt @ H
    Hr = G @ Vt.mT
    Hr = 0.5 * (Hr + Hr.mT)
    Hr.diagonal().add_(0.01 * Hr.diagonal().mean().abs())
    A_star = torch.linalg.solve(Hr, (Wf @ G.mT).mT).mT

    def J(A):
        M = Wf - A @ Vt
        return (M * (M @ H)).sum().item()
    j0, j1 = J(A_f), J(A_star)
    check("A* strictly lowers the H-weighted objective vs A_z", j1 < j0,
          f"A_z={j0:.6e} A*={j1:.6e}  gain={(j0-j1)/j0:.4f}")

    # and the compensated binarisation beats plain sign() at the same scales
    s1 = A_star.abs().mean(dim=1)
    coln = A_star.abs().mean(dim=0)
    s3 = coln / coln.mean()
    U_c = V._gptq_binarize_columns(A_star, Hr, s1, s3)
    U_n = V._sign(A_star)
    jc = J(s1.unsqueeze(1) * U_c * s3.unsqueeze(0))
    jn = J(s1.unsqueeze(1) * U_n * s3.unsqueeze(0))
    check("GPTQ-compensated binarisation beats plain sign()", jc < jn,
          f"plain={jn:.6e} compensated={jc:.6e}  gain={(jn-jc)/jn:.4f}")
    check("binarised A stays strictly worse than the real refit", j1 < jc, f"{j1:.4e} < {jc:.4e}")


if __name__ == "__main__":
    # the five layer shapes that actually occur in Llama-3.2-1B
    test_admm_identity(2048, 2048, 64, iters=20)   # q_proj / o_proj  (direct, square)
    test_admm_identity(2048, 512, 64, iters=20)    # (direct, tall)
    test_admm_identity(512, 2048, 64, iters=20)    # k_proj / v_proj  (transposed)
    test_admm_identity(8192, 2048, 64, iters=20)   # gate_proj / up_proj (direct)
    test_admm_identity(2048, 8192, 64, iters=20)   # down_proj (transposed)
    test_admm_H_changes_result()
    test_refit_lowers_objective()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
    sys.exit(1 if FAIL else 0)
