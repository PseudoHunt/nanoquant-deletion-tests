import torch, time
from nanoquant.modules.hub import NanoQuantConfigDataclass  # noqa
from nanoquant.utils.load_utils import load_model
MID="/home/work/hf_cache/hub/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da"
m = load_model(MID, 2048, "cpu"); blk = m.model.layers[0].cuda()
for p in m.parameters(): p.requires_grad_(False)
rot = m.model.rotary_emb.cuda()
xreal = torch.load('/home/work/exp/artifacts/sbh/in_b0.pt')[:4].cuda()
with torch.no_grad():
    pos = tuple(t.detach().clone() for t in rot(xreal[:1], torch.arange(2048, device="cuda").unsqueeze(0)))
kw = {"position_embeddings": pos, "attention_mask": None, "use_cache": False}
@torch.no_grad()
def fwd(x, n=8):
    for i in range(3): blk(x[:1], **kw)
    torch.cuda.synchronize(); t=time.time()
    for i in range(n): blk(x[:1], **kw)
    torch.cuda.synchronize(); return (time.time()-t)/n*1000
xr = torch.randn_like(xreal)
std = xreal.float().std().item()
cases = [("random std1", xr), ("real", xreal), ("real*100", xreal*100), ("real*1000", xreal*1000),
         ("random*real_std", xr*std), ("real fp32 roundtrip", xreal.float().bfloat16())]
print("real std:", std, "absmax:", xreal.float().abs().max().item())
for tag, x in cases:
    print(f"{tag:22s} fwd={fwd(x):7.2f} ms")
# where does forward time go? per-submodule
import torch.nn as nn
times={}
def mk(name):
    def pre(mod, inp): torch.cuda.synchronize(); times[name]=time.time()
    def post(mod, inp, out): torch.cuda.synchronize(); times[name]=time.time()-times[name]
    return pre, post
hs=[]
for n_, mod in blk.named_modules():
    if isinstance(mod,(nn.Linear,)) or n_ in ("self_attn","mlp","input_layernorm","post_attention_layernorm"):
        pre,post=mk(n_); hs.append(mod.register_forward_pre_hook(pre)); hs.append(mod.register_forward_hook(post))
for tag, x in [("random std1", xr), ("real", xreal)]:
    with torch.no_grad(): blk(x[:1], **kw)
    print(f"--- {tag} submodule ms ---", {k: round(v*1000,2) for k,v in sorted(times.items(), key=lambda z:-z[1])[:6]})
