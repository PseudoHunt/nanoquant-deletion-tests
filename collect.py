"""Aggregate single-block harness runs into a table."""
import glob, json, sys
import numpy as np

rows = {}
for f in sorted(glob.glob("/home/work/exp/artifacts/sbh_runs/*.json")):
    r = json.load(open(f))
    if "block_post" not in r:
        continue
    tag = r["tag"]
    arm, b, s = tag.rsplit("_b", 1)[0], int(tag.rsplit("_b", 1)[1].split("_s")[0]), int(tag.split("_s")[-1])
    rows.setdefault((arm, b), []).append((s, r))

def agg(v):
    a = np.array(v, dtype=float)
    return a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0

print(f"{'arm':16s} {'blk':>3} {'n':>2} {'pre (mean+-sd)':>26} {'post (mean+-sd)':>26} {'ppl wiki2':>18} {'KL-to-FP':>16}")
for (arm, b), v in sorted(rows.items()):
    v.sort()
    pre = agg([r["block_pre"] for _, r in v]); post = agg([r["block_post"] for _, r in v])
    ppl = agg([r.get("ppl_wiki2", float('nan')) for _, r in v])
    kl = agg([r.get("kl_to_fp", float('nan')) for _, r in v])
    print(f"{arm:16s} {b:3d} {len(v):2d} {pre[0]:12.5e}+-{pre[1]:8.1e} {post[0]:12.5e}+-{post[1]:8.1e} "
          f"{ppl[0]:10.4f}+-{ppl[1]:5.3f} {kl[0]:9.5f}+-{kl[1]:5.4f}")
if len(sys.argv) > 1 and sys.argv[1] == "--json":
    out = {f"{a}_b{b}": {"n": len(v),
                         "pre": [r["block_pre"] for _, r in v], "post": [r["block_post"] for _, r in v],
                         "ppl": [r.get("ppl_wiki2") for _, r in v], "kl": [r.get("kl_to_fp") for _, r in v]}
           for (a, b), v in sorted(rows.items())}
    json.dump(out, open("/home/work/exp/artifacts/sbh_summary.json", "w"), indent=1)
