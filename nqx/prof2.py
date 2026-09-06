import torch, time, warnings, traceback
from nanoquant.modules.hub import NanoQuantConfigDataclass  # noqa
from nanoquant.utils.load_utils import load_model
from nanoquant.core.compress_block import fused_weighted_mse
MID="/home/work/hf_cache/hub/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da"
m = load_model(MID, 2048, "cpu"); blk = m.model.layers[0].cuda()
rot = m.model.rotary_emb.cuda()
raw = torch.load('/home/work/exp/artifacts/sbh/in_b0.pt')
rawy = torch.load('/home/work/exp/artifacts/sbh/out_b0.pt')
xreal = raw[:4].cuda(); yreal = rawy[:4].cuda()
print("storage nbytes of loaded slice-view source:", raw.untyped_storage().nbytes()/1e6, "MB")
print("xreal storage:", xreal.untyped_storage().nbytes()/1e6, "MB  numel*2:", xreal.numel()*2/1e6)
with torch.no_grad():
    pos = tuple(t.detach().clone() for t in rot(xreal[:1], torch.arange(2048, device="cuda").unsqueeze(0)))
kw = {"position_embeddings": pos, "attention_mask": None, "use_cache": False}
params=[mm.weight for mm in blk.modules() if isinstance(mm, torch.nn.Linear)]
for p in params: p.requires_grad=True
imp = torch.ones(2048, device="cuda")
xr = torch.randn_like(xreal); yr = torch.randn_like(yreal)
def step(x,y,i=0):
    o=blk(x[i:i+1], **kw)[0]; fused_weighted_mse(o,y[i:i+1],imp).backward(); blk.zero_grad(set_to_none=True)
def timeit(x,y,tag,n=5):
    for i in range(2): step(x,y,i)
    torch.cuda.synchronize(); t=time.time()
    for i in range(n): step(x,y,i%4)
    torch.cuda.synchronize(); print(f"{tag:32s}: {(time.time()-t)/n*1000:8.1f} ms/step", flush=True)
timeit(xr,yr,"random")
timeit(xreal,yreal,"real (loaded)")
timeit(xreal.clone(),yreal.clone(),"real cloned")
del raw, rawy
import gc; gc.collect()
timeit(xreal,yreal,"real after freeing CPU source")
warnings.simplefilter("error")
torch.cuda.set_sync_debug_mode(1)
try:
    step(xreal,yreal,0)
    print("no sync warning raised")
except Exception as e:
    print("SYNC:", type(e).__name__, str(e)[:200])
    traceback.print_exc(limit=12)
torch.cuda.set_sync_debug_mode(0)
