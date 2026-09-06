import torch, time
from nanoquant.modules.hub import NanoQuantConfigDataclass  # noqa
from nanoquant.utils.load_utils import load_model
from nanoquant.core.compress_block import tune_nonfact, fused_weighted_mse
from nanoquant.optimi import AdamW
MID="/home/work/hf_cache/hub/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da"
m = load_model(MID, 2048, "cpu"); blk = m.model.layers[0].cuda()
kw = torch.load('/home/work/exp/artifacts/sbh/kwargs.pt')
kw = {k:(v.cuda() if isinstance(v,torch.Tensor) else v) for k,v in kw.items()}
N=8
x = torch.load('/home/work/exp/artifacts/sbh/in_b0.pt')[:N].cuda()
y = torch.load('/home/work/exp/artifacts/sbh/out_b0.pt')[:N].cuda()
imp = torch.ones(2048, device="cuda")
qc = NanoQuantConfigDataclass(seed=0, num_calib_samples=N, nonfact_epochs=1).to_dict()
torch.cuda.synchronize(); t=time.time()
tune_nonfact(blk, x, y, imp, kw, qc)
torch.cuda.synchronize(); dt=time.time()-t
print(f"tune_nonfact          : {dt/N*1000:8.1f} ms/step")
params=[mm.weight for mm in blk.modules() if isinstance(mm, torch.nn.Linear)]
for p in params: p.requires_grad=True
opt = AdamW(params, lr=1e-4, weight_decay=0)
def bench(kwargs, tag, n=8):
    torch.cuda.synchronize(); t=time.time()
    for i in range(n):
        out = blk(x[i:i+1], **kwargs)[0]
        fused_weighted_mse(out, y[i:i+1], imp).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); print(f"{tag:22s}: {(time.time()-t)/n*1000:8.1f} ms/step")
bench(kw, "hand-rolled, dense mask")
kn = dict(kw); kn['attention_mask']=None
bench(kn, "hand-rolled, mask=None")
