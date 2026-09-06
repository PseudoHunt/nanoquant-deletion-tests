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
blk = get_decoder_layers(m)[0]
sub = find_layers(blk)
for name, rank in [("self_attn.q_proj", 992), ("self_attn.v_proj", 384)]:
    lx = sub[name]
    W = lx.weight.data.cuda()
    i_norm = lx.i_norm.cuda().float(); o_norm = lx.o_norm.cuda().float()
    H = I.prep_hin(f"model.layers.0.{name}", f"{CACHE}/hin", shrinkage=0.2, device="cuda")
    is_t = W.shape[0] < W.shape[1]
    print(f"\n=== {name} shape={tuple(W.shape)} transpose={is_t} rank={rank}")
    print(f"  i_norm: mean={i_norm.mean():.4e} min={i_norm.min():.4e} max={i_norm.max():.4e}")
    print(f"  diag(H): mean={H.diagonal().mean():.4e} min={H.diagonal().min():.4e} max={H.diagonal().max():.4e}")
    print(f"  ratio diag(H)/i_norm: mean={(H.diagonal()/i_norm).mean():.4f} "
          f"min={(H.diagonal()/i_norm).min():.4f} max={(H.diagonal()/i_norm).max():.4f}")
    d = i_norm.sqrt().clamp(1e-12)
    Ht = H / (d.unsqueeze(0)*d.unsqueeze(1))
    ev = torch.linalg.eigvalsh(Ht.double())
    print(f"  H~ eig: min={ev.min():.4e} max={ev.max():.4e} cond={(ev.max()/ev.min().clamp(1e-30)):.3e}")
    for it in [5, 20, 60]:
        set_seed(0); a = factorize_admm_nanoquant(W, i_norm, o_norm, rank, outer_iters=it, is_transpose=is_t, rho_scheduler='linear')
        set_seed(0); b = V.factorize_admm_hin(W, i_norm, o_norm, rank, H_in=H, outer_iters=it, is_transpose=is_t, rho_scheduler='linear')
        na = (a['W_final'].float()-W.float()).square().sum()/W.float().square().sum()
        nb = (b['W_final'].float()-W.float()).square().sum()/W.float().square().sum()
        print(f"  iters={it:3d}  stock recon={na:.4e}   H-weighted recon={nb:.4e}")
