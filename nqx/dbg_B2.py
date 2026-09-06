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
for name, rank in [("self_attn.q_proj", 992), ("self_attn.v_proj", 384), ("mlp.down_proj", 1600)]:
    lx = sub[name]; W = lx.weight.data.cuda()
    i_norm = lx.i_norm.cuda().float(); o_norm = lx.o_norm.cuda().float()
    H = I.prep_hin(f"model.layers.0.{name}", f"{CACHE}/hin", shrinkage=0.2, device="cuda")
    is_t = W.shape[0] < W.shape[1]
    Wf = W.float()
    def J(Wh):   # H-weighted objective, relative
        M = Wf - Wh.float()
        return ((M*(M@H)).sum() / (Wf*(Wf@H)).sum()).item()
    def F(Wh):
        M = Wf - Wh.float()
        return (M.square().sum()/Wf.square().sum()).item()
    print(f"\n=== {name} {tuple(W.shape)} transpose={is_t} r={rank}")
    print(f"{'iters':>6} {'stock J':>12} {'stock fro':>12} {'H-wtd J':>12} {'H-wtd fro':>12}")
    for it in [20, 60, 150, 400]:
        set_seed(0); a = factorize_admm_nanoquant(W, i_norm, o_norm, rank, outer_iters=it, is_transpose=is_t, rho_scheduler='linear')
        set_seed(0); b = V.factorize_admm_hin(W, i_norm, o_norm, rank, H_in=H, outer_iters=it, is_transpose=is_t, rho_scheduler='linear')
        print(f"{it:6d} {J(a['W_final']):12.4e} {F(a['W_final']):12.4e} {J(b['W_final']):12.4e} {F(b['W_final']):12.4e}", flush=True)
    del H
    torch.cuda.empty_cache()
