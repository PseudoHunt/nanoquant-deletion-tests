import torch, time
from nanoquant.modules.hub import NanoQuantConfigDataclass  # noqa
from nanoquant.utils.load_utils import load_model
MID="/home/work/hf_cache/hub/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da"
m = load_model(MID, 2048, "cpu"); blk = m.model.layers[0].cuda()
rot = m.model.rotary_emb.cuda()
xreal = torch.load('/home/work/exp/artifacts/sbh/in_b0.pt')[:4].cuda()
with torch.no_grad():
    pos = tuple(t.detach().clone() for t in rot(xreal[:1], torch.arange(2048, device="cuda").unsqueeze(0)))
kw = {"position_embeddings": pos, "attention_mask": None, "use_cache": False}
params=[mm.weight for mm in blk.modules() if isinstance(mm, torch.nn.Linear)]
for p in params: p.requires_grad=True
xr = torch.randn_like(xreal)
def timeit(x,tag,n=5):
    def step(i):
        o=blk(x[i:i+1], **kw)[0]; o.float().square().sum().backward(); blk.zero_grad(set_to_none=True)
    for i in range(2): step(i)
    torch.cuda.synchronize(); t=time.time()
    for i in range(n): step(i%4)
    torch.cuda.synchronize(); print(f"{tag:34s}: {(time.time()-t)/n*1000:8.1f} ms/step", flush=True)
timeit(xr, "random")
for s in [1, 10, 100, 1000]:
    timeit(xreal*s, f"real x * {s}")
# inspect attention statistics for real vs random
with torch.no_grad():
    for tag, x in [("random", xr), ("real", xreal)]:
        h = blk.input_layernorm(x[:1])
        q = blk.self_attn.q_proj(h); k = blk.self_attn.k_proj(h)
        print(f"{tag}: |h| max={h.float().abs().max():.3f} std={h.float().std():.3f} | "
              f"|q| max={q.float().abs().max():.3f} |k| max={k.float().abs().max():.3f}")
