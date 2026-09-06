import torch
from nanoquant.modules.hub import NanoQuantConfigDataclass  # noqa: F401
from nanoquant.core.admm_nq import factorize_admm_nanoquant
from nanoquant.utils.utils import set_seed
import variants as V
DEV="cuda"
def rel(a,b): return ((a-b).float().norm()/b.float().norm().clamp(min=1e-30)).item()
for (d_out,d_in,r) in [(2048,2048,64),(512,2048,64),(2048,512,64),(8192,2048,64),(2048,8192,64),(128,96,32)]:
    torch.manual_seed(7)
    W = torch.randn(d_out,d_in,device=DEV,dtype=torch.bfloat16)
    i_norm = torch.rand(d_in,device=DEV)+0.5
    o_norm = torch.rand(d_out,device=DEV)+0.5
    is_t = d_out < d_in
    norm_i = i_norm.sqrt().clamp(1e-12)
    H = torch.diag(norm_i*norm_i).float()
    out=[]
    for k in [1,5,20]:
        set_seed(0); a = factorize_admm_nanoquant(W,i_norm,o_norm,r,outer_iters=k,is_transpose=is_t,rho_scheduler='linear')
        set_seed(0); b = V.factorize_admm_hin(W,i_norm,o_norm,r,H_in=H,outer_iters=k,is_transpose=is_t,rho_scheduler='linear')
        out.append(f"k={k}:{rel(b['W_final'],a['W_final']):.1e}")
    print(f"{d_out}x{d_in} transpose={is_t}  " + "  ".join(out))
