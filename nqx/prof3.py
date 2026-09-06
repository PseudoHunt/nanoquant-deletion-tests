import torch, time
from nanoquant.modules.hub import NanoQuantConfigDataclass  # noqa
from nanoquant.utils.load_utils import load_model
from nanoquant.core.compress_block import fused_weighted_mse
MID="/home/work/hf_cache/hub/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da"
m = load_model(MID, 2048, "cpu"); blk = m.model.layers[0].cuda()
rot = m.model.rotary_emb.cuda()
xreal = torch.load('/home/work/exp/artifacts/sbh/in_b0.pt')[:4].cuda()
yreal = torch.load('/home/work/exp/artifacts/sbh/out_b0.pt')[:4].cuda()
with torch.no_grad():
    pos = tuple(t.detach().clone() for t in rot(xreal[:1], torch.arange(2048, device="cuda").unsqueeze(0)))
kw = {"position_embeddings": pos, "attention_mask": None, "use_cache": False}
params=[mm.weight for mm in blk.modules() if isinstance(mm, torch.nn.Linear)]
for p in params: p.requires_grad=True
imp = torch.ones(2048, device="cuda")
xr = torch.randn_like(xreal); yr = torch.randn_like(yreal)
def timeit(x,y,tag,n=5):
    def step(i):
        o=blk(x[i:i+1], **kw)[0]; fused_weighted_mse(o,y[i:i+1],imp).backward(); blk.zero_grad(set_to_none=True)
    for i in range(2): step(i)
    torch.cuda.synchronize(); t=time.time()
    for i in range(n): step(i%4)
    torch.cuda.synchronize(); print(f"{tag:40s}: {(time.time()-t)/n*1000:8.1f} ms/step", flush=True)
timeit(xr,    yr,    "random x, random y   (grad O(1))")
timeit(xreal, yreal, "real x,   real y     (grad ~4e-5)")
timeit(xreal, yr,    "real x,   RANDOM y   (grad O(1))")
timeit(xr,    yreal, "random x, real y     (grad O(1))")
timeit(xreal, yreal + torch.randn_like(yreal)*0.5, "real x, real y + big noise")
# scale the real input up so activations are O(1)
timeit(xreal*50, (yreal*50), "real x*50, real y*50")
