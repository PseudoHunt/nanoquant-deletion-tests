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
xrand = torch.randn_like(xreal)

def fwd_only(x, n=6):
    with torch.no_grad():
        for i in range(2): blk(x[i%x.shape[0]:i%x.shape[0]+1], **kw)
        torch.cuda.synchronize(); t=time.time()
        for i in range(n): blk(x[i%x.shape[0]:i%x.shape[0]+1], **kw)
        torch.cuda.synchronize(); return (time.time()-t)/n*1000

def fwd_bwd(x, n=6):
    for p in [mm.weight for mm in blk.modules() if isinstance(mm, torch.nn.Linear)]: p.requires_grad_(True)
    def one(i):
        o = blk(x[i%x.shape[0]:i%x.shape[0]+1], **kw)[0]
        o.float().square().sum().backward()
        blk.zero_grad(set_to_none=True)
    for i in range(2): one(i)
    torch.cuda.synchronize(); t=time.time()
    for i in range(n): one(i)
    torch.cuda.synchronize(); r=(time.time()-t)/n*1000
    for p in [mm.weight for mm in blk.modules() if isinstance(mm, torch.nn.Linear)]: p.requires_grad_(False)
    return r

for tag, x in [("random", xrand), ("real", xreal), ("real*100", xreal*100),
               ("real cloned", xreal.clone()), ("real->fp32->bf16", xreal.float().bfloat16()),
               ("randn*real_std", torch.randn_like(xreal)*xreal.float().std())]:
    print(f"{tag:20s} fwd={fwd_only(x):7.1f} ms   fwd+bwd={fwd_bwd(x):7.1f} ms")
