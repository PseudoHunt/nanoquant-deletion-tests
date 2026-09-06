"""Two diagnostics for Test D:
(1) dynamic range of the raw dL/dU that the mirror step uses, against |Theta|
(2) magnitude of the relaxed forward tanh(beta*Theta) that hardening replaces with 1
"""
import torch, json
import sys
sys.argv = [sys.argv[0], "run"]
import testD as D
import instrument as I
from nanoquant.core.compress_block import fused_weighted_mse

model, qd, kwargs, cal_in, cal_out, ho_in, ho_out = D.load_harness(0)
blob = torch.load(f"{D.DCACHE}/block0_postadmm.pt", weights_only=False)
blk = blob["block"].to("cuda")
sub = dict(D.nq_modules(blk))
imp = sub['mlp.down_proj'].o_norm.to("cuda")
cal_in_d, cal_out_d = cal_in[:4].to("cuda"), cal_out[:4].to("cuda")

print("=== (2) relaxed-forward magnitude |tanh(beta*Theta)| ===")
D.to_theta(blk)
print(f"{'layer':22s} {'beta=1 V':>10s} {'beta=10 V':>10s} {'beta=1 U':>10s} {'beta=10 U':>10s}")
for n in D.NAMES:
    m = sub[n]
    row = []
    for attr in ["V_latent", "U_latent"]:
        t = getattr(m, attr).data
        row += [torch.tanh(1.0*t).abs().mean().item(), torch.tanh(10.0*t).abs().mean().item()]
    print(f"{n:22s} {row[0]:10.4f} {row[1]:10.4f} {row[2]:10.4f} {row[3]:10.4f}")

print("\n=== (1) raw |dL/dU| that the mirror step adds to Theta, vs |Theta| ===")
D.MIRROR["on"] = True; D.MIRROR["hard"] = True; D.MIRROR["beta"] = 1.0
for _, m in D.nq_modules(blk):
    for n_, p in m.named_parameters():
        if "latent" in n_: p.requires_grad_(True)
y = blk(cal_in_d[0:1], **kwargs)[0]
fused_weighted_mse(y, cal_out_d[0:1], imp).backward()
print(f"{'layer':22s} {'|g| median':>12s} {'|g| p99':>12s} {'|g| max':>12s} {'|Theta| med':>12s} "
      f"{'eta*|g|/|Th| med':>17s} {'p99':>10s}")
for n in D.NAMES:
    m = sub[n]
    for attr in ["V_latent"]:
        g = getattr(m, attr).grad.abs().flatten()
        th = getattr(m, attr).data.abs().flatten()
        r = (0.03*g/th.clamp(min=1e-12))
        print(f"{n+'.'+attr[0]:22s} {g.median():12.3e} {g.quantile(0.99):12.3e} {g.max():12.3e} "
              f"{th.median():12.3e} {r.median():17.3e} {r.quantile(0.99):10.3e}")
