import torch
from nanoquant.modules.hub import NanoQuantConfigDataclass  # noqa
from nanoquant.core.admm_nq import factorize_admm_nanoquant
from nanoquant.utils.load_utils import load_model
from nanoquant.utils.utils import get_decoder_layers, find_layers, set_seed
from nanoquant.core.importance import register_stats
import instrument as I, variants as V
MID="/home/work/hf_cache/hub/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da"
CACHE="/home/work/exp/artifacts/sbh"
m = load_model(MID, 2048, "cpu")
st = torch.load(f"{CACHE}/stats.pt"); st['stats_device']='cpu'
m = register_stats(m, st)
sub = find_layers(get_decoder_layers(m)[0])
for name, rank in [("self_attn.v_proj", 384), ("self_attn.q_proj", 992)]:
    lx = sub[name]; W = lx.weight.data.cuda()
    i_norm = lx.i_norm.cuda().float(); o_norm = lx.o_norm.cuda().float()
    H = I.prep_hin(f"model.layers.0.{name}", f"{CACHE}/hin", shrinkage=0.2, device="cuda")
    is_t = W.shape[0] < W.shape[1]
    Wf = W.float()
    def J(Wh):
        M = Wf - Wh.float(); return ((M*(M@H)).sum()/(Wf*(Wf@H)).sum()).item()
    # scale H so that H~ = D^-1 H D^-1 has unit MEAN diagonal, which is what the
    # stock code's unit-norm columns guarantee for X^T X
    d = i_norm.sqrt().clamp(1e-12)
    Ht = H / (d.unsqueeze(0)*d.unsqueeze(1))
    scale = Ht.diagonal().mean()
    print(f"\n=== {name} transpose={is_t}  mean diag(H~)={scale:.4f} -> rescaling H by 1/{scale:.4f}")
    Hn = H / scale
    print(f"{'iters':>6} {'stock J':>12} {'H raw J':>12} {'H norm J':>12}")
    for it in [60, 150, 400]:
        set_seed(0); a = factorize_admm_nanoquant(W, i_norm, o_norm, rank, outer_iters=it, is_transpose=is_t, rho_scheduler='linear')
        set_seed(0); b = V.factorize_admm_hin(W, i_norm, o_norm, rank, H_in=H,  outer_iters=it, is_transpose=is_t, rho_scheduler='linear')
        set_seed(0); c = V.factorize_admm_hin(W, i_norm, o_norm, rank, H_in=Hn, outer_iters=it, is_transpose=is_t, rho_scheduler='linear')
        print(f"{it:6d} {J(a['W_final']):12.4e} {J(b['W_final']):12.4e} {J(c['W_final']):12.4e}", flush=True)
    del H, Hn, Ht; torch.cuda.empty_cache()
