"""Diagnose: does the H=I path differ by the update rule, or by chaotic
amplification of float rounding through the sign() projection?"""
import torch
from nanoquant.modules.hub import NanoQuantConfigDataclass  # noqa: F401
from nanoquant.core.admm_nq import factorize_admm_nanoquant, _admm_solve_step
from nanoquant.utils.utils import set_seed
import variants as V

DEV = "cuda"
def rel(a, b): return ((a-b).norm()/b.norm().clamp(min=1e-30)).item()

# --- 1. the solve step itself, at H = I ---
torch.manual_seed(0)
X = torch.randn(96, 32, device=DEV); Y = torch.randn(96, 128, device=DEV)
Z = torch.randn(32, 128, device=DEV); U = torch.randn(32, 128, device=DEV)
Eye = torch.eye(96, device=DEV)
a = _admm_solve_step(X, Y, Z, U, 0.37, 3e-2)
b = V._admm_solve_step_H(X, Y, Z, U, 0.37, 3e-2, HY=Eye@Y, H=Eye)
print(f"solve-step at H=I:            rel={rel(b,a):.3e}")

# --- 2. full loop vs iteration count, float32 to remove bf16 noise ---
for (d_out, d_in, r) in [(128, 96, 32), (96, 128, 32)]:
    torch.manual_seed(7)
    W = torch.randn(d_out, d_in, device=DEV, dtype=torch.float32)
    i_norm = torch.rand(d_in, device=DEV) + 0.5
    o_norm = torch.rand(d_out, device=DEV) + 0.5
    is_t = d_out < d_in
    H = torch.diag(i_norm).float()
    Wp = W * (1.0 + 1e-7)                      # 1e-7 relative perturbation, control
    print(f"\n  {d_out}x{d_in} transpose={is_t}")
    print(f"  {'iters':>6} {'H=I vs stock':>14} {'stock vs stock+1e-7':>20}")
    for k in [1, 2, 5, 10, 20, 40, 80, 160]:
        set_seed(0); ref = factorize_admm_nanoquant(W, i_norm, o_norm, r, outer_iters=k, is_transpose=is_t, rho_scheduler='linear')
        set_seed(0); got = V.factorize_admm_hin(W, i_norm, o_norm, r, H_in=H, outer_iters=k, is_transpose=is_t, rho_scheduler='linear')
        set_seed(0); per = factorize_admm_nanoquant(Wp, i_norm, o_norm, r, outer_iters=k, is_transpose=is_t, rho_scheduler='linear')
        print(f"  {k:6d} {rel(got['W_final'], ref['W_final']):14.3e} {rel(per['W_final'], ref['W_final']):20.3e}")
