import torch, time
from torch.profiler import profile, ProfilerActivity
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
def step(x,y,i=0):
    o=blk(x[i:i+1], **kw)[0]; fused_weighted_mse(o,y[i:i+1],imp).backward(); blk.zero_grad(set_to_none=True)
for tag,(x,y) in [("RANDOM",(xr,yr)), ("REAL",(xreal,yreal))]:
    for i in range(3): step(x,y,i)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as pr:
        for i in range(3): step(x,y,i)
        torch.cuda.synchronize()
    print(f"########## {tag} ##########")
    print(pr.key_averages().table(sort_by="cuda_time_total", row_limit=8, max_name_column_width=70))
