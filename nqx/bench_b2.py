import torch, time
from nanoquant.modules.hub import NanoQuantConfigDataclass  # noqa
from nanoquant.utils.load_utils import load_model
from nanoquant.core.compress_block import fused_weighted_mse
from nanoquant.optimi import AdamW
MID="/home/work/hf_cache/hub/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da"
m = load_model(MID, 2048, "cpu"); blk = m.model.layers[0].cuda()
rot = m.model.rotary_emb.cuda()
xreal = torch.load('/home/work/exp/artifacts/sbh/in_b0.pt')[:8].cuda()
yreal = torch.load('/home/work/exp/artifacts/sbh/out_b0.pt')[:8].cuda()
for _p in m.parameters(): _p.requires_grad_(False)
with torch.no_grad():
    pos = tuple(t.detach() for t in rot(xreal[:1], torch.arange(2048, device="cuda").unsqueeze(0)))
kw = {"position_embeddings": pos, "attention_mask": None, "use_cache": False}
params=[mm.weight for mm in blk.modules() if isinstance(mm, torch.nn.Linear)]
for p in params: p.requires_grad=True
opt = AdamW(params, lr=1e-4, weight_decay=0)
imp = torch.ones(2048, device="cuda")
def bench(x,y,tag,n=6):
    for i in range(2):
        o=blk(x[i:i+1], **kw)[0]; fused_weighted_mse(o,y[i:i+1],imp).backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); t=time.time()
    for i in range(n):
        o=blk(x[i:i+1], **kw)[0]; fused_weighted_mse(o,y[i:i+1],imp).backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); print(f"{tag:38s}: {(time.time()-t)/n*1000:8.1f} ms/step")
xr = torch.randn_like(xreal); yr = torch.randn_like(yreal)
bench(xr, yr, "random x (std 1)")
bench(xreal, yreal, "real x")
bench(xreal*100, yreal*100, "real x * 100")
bench(xreal.clone().contiguous(), yreal.clone().contiguous(), "real x, cloned contiguous")
bench(xreal + torch.randn_like(xreal)*1e-3, yreal, "real x + small noise")
std = xreal.float().std().item()
print("real x std:", std, " random std:", xr.float().std().item())
bench(xr*std, yr*std, "random x scaled to real std")
