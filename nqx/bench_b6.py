import torch, time
from nanoquant.modules.hub import NanoQuantConfigDataclass  # noqa
from nanoquant.utils.load_utils import load_model, cache_inputs_and_kwargs, load_tokenizer
from nanoquant.utils.data_utils import get_calib_loader, prepare_dataset
from nanoquant.core.compress_block import fused_weighted_mse
from nanoquant.optimi import AdamW
MID="/home/work/hf_cache/hub/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da"
qd = NanoQuantConfigDataclass(model_id=MID, seed=0).to_dict()
m = load_model(MID, 2048, "cpu"); m.eval(); m.config.use_cache=False
m.gradient_checkpointing_disable()
data = prepare_dataset(MID, qd); tok = load_tokenizer(MID)
calib = get_calib_loader(data, tok, 8, 0, 2048)
live_x, live_kw = cache_inputs_and_kwargs(m, calib, "cuda")
live_kw = {k:(v.to("cuda") if isinstance(v,torch.Tensor) else v) for k,v in live_kw.items()}
live_kw['use_cache']=False
print("LIVE kwargs:", {k:(f"{type(v).__name__}{tuple(v.shape) if isinstance(v,torch.Tensor) else ''}") for k,v in live_kw.items()})
blk = m.model.layers[0].cuda()
with torch.no_grad():
    live_y = torch.stack([blk(live_x[j:j+1].cuda(), **live_kw)[0][0] for j in range(8)]).cpu()
cached_x = torch.load('/home/work/exp/artifacts/sbh/in_b0.pt')[:8]
cached_kw = torch.load('/home/work/exp/artifacts/sbh/kwargs.pt')
cached_kw = {k:(v.to("cuda") if isinstance(v,torch.Tensor) else v) for k,v in cached_kw.items()}
print("CACHED kwargs:", {k:(f"{type(v).__name__}{tuple(v.shape) if isinstance(v,torch.Tensor) else ''}") for k,v in cached_kw.items()})
print("live_x vs cached_x maxdiff:", (live_x[:8].float()-cached_x.float()).abs().max().item(),
      " dtypes:", live_x.dtype, cached_x.dtype, " contig:", live_x.is_contiguous(), cached_x.is_contiguous())
params=[mm.weight for mm in blk.modules() if isinstance(mm, torch.nn.Linear)]
for p in params: p.requires_grad=True
opt = AdamW(params, lr=1e-4, weight_decay=0)
imp = torch.ones(2048, device="cuda")
def bench(x,y,kw,tag,n=6):
    x=x.cuda(); y=y.cuda()
    for i in range(2):
        o=blk(x[i:i+1], **kw)[0]; fused_weighted_mse(o,y[i:i+1],imp).backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); t=time.time()
    for i in range(n):
        o=blk(x[i:i+1], **kw)[0]; fused_weighted_mse(o,y[i:i+1],imp).backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); print(f"{tag:34s}: {(time.time()-t)/n*1000:8.1f} ms/step", flush=True)
bench(live_x[:8], live_y, live_kw,   "live x, live kwargs")
bench(cached_x,   live_y, live_kw,   "cached x, live kwargs")
bench(live_x[:8], live_y, cached_kw, "live x, cached kwargs")
bench(cached_x,   live_y, cached_kw, "cached x, cached kwargs")
