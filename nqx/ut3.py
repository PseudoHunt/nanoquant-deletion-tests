import torch
from nanoquant.modules.hub import NanoQuantConfigDataclass  # noqa: F401
import variants as V
DEV="cuda"
torch.manual_seed(7)
d_out,d_in,r = 128,96,32
W = torch.randn(d_out,d_in,device=DEV,dtype=torch.bfloat16)
i_norm = torch.rand(d_in,device=DEV)+0.5
o_norm = torch.rand(d_out,device=DEV)+0.5
norm_i = i_norm.sqrt().clamp(1e-12); norm_o = o_norm.sqrt().clamp(1e-12).unsqueeze(1)
H = torch.diag(norm_i*norm_i).float()
d = norm_i.float()
H_t = H / (d.unsqueeze(0)*d.unsqueeze(1)); H_t = 0.5*(H_t+H_t.mT)
print("H_t == I exactly:", bool(torch.equal(H_t, torch.eye(d_in,device=DEV))))
W_norm = (W * norm_i.unsqueeze(0) * norm_o)
Y = W_norm.mT.to(torch.float32)
with V.no_tf32():
    HY = H_t @ Y
print("HY == Y bitwise:", bool(torch.equal(HY, Y)), " Y contiguous:", Y.is_contiguous(), " HY contiguous:", HY.is_contiguous())
X = torch.randn(d_in, r, device=DEV)
p1 = X.mT @ Y
p2 = X.mT @ HY
print("X^T Y  == X^T HY bitwise:", bool(torch.equal(p1,p2)), " maxdiff:", (p1-p2).abs().max().item())
p3 = X.mT @ Y.contiguous()
print("X^T Y  == X^T Y.contiguous() bitwise:", bool(torch.equal(p1,p3)), " maxdiff:", (p1-p3).abs().max().item())
