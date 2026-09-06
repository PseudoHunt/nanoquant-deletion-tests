"""Promotion decision + per-layer breakdown for the single-block harness.

Verdict criterion (block level, as briefed): an arm is promoted only if its
`post` block-output error beats the baseline on BOTH blocks beyond seed spread.
Seeds are paired -- the same seed means the same ADMM init and the same tuning
order -- which is what makes n=3 usable.

Per-layer `post` is printed alongside, because the block error is dominated by
the three MLP layers; an MLP-only gain has to be visible as such.
"""
import glob, json, sys
from statistics import mean, stdev

NAMES = ['self_attn.q_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'self_attn.k_proj',
         'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']

rows = {}
for f in sorted(glob.glob("/home/work/exp/artifacts/sbh_runs/*.json")):
    r = json.load(open(f))
    if "block_post" not in r:
        continue
    tag = r["tag"]
    arm = tag.rsplit("_b", 1)[0]
    b = int(tag.rsplit("_b", 1)[1].split("_s")[0])
    s = int(tag.split("_s")[-1])
    rows[(arm, b, s)] = r

arms = sorted({a for a, _, _ in rows})
blocks = sorted({b for _, b, _ in rows})
seeds = sorted({s for _, _, s in rows})


def sd(v):
    return stdev(v) if len(v) > 1 else 0.0


def col(arm, b, key):
    return [rows[(arm, b, s)][key] for s in seeds if (arm, b, s) in rows]


def lcol(arm, b, name, key):
    out = []
    for s in seeds:
        r = rows.get((arm, b, s))
        if r and f"{b}.{name}" in r["layers"] and r["layers"][f"{b}.{name}"].get(key) is not None:
            out.append(r["layers"][f"{b}.{name}"][key])
    return out


print("=== block level ===")
print(f"{'arm':8s} {'blk':>3} {'n':>2} {'pre':>19} {'post':>19} {'ppl(1 blk)':>15} {'KL-to-FP':>10}")
for arm in arms:
    for b in blocks:
        post = col(arm, b, "block_post")
        if not post:
            continue
        pre = col(arm, b, "block_pre")
        ppl = [x for x in (rows[(arm, b, s)].get("ppl_wiki2") for s in seeds if (arm, b, s) in rows) if x]
        kl = [x for x in (rows[(arm, b, s)].get("kl_to_fp") for s in seeds if (arm, b, s) in rows) if x]
        print(f"{arm:8s} {b:3d} {len(post):2d} {mean(pre):11.5e}+-{sd(pre):6.0e} {mean(post):11.5e}+-{sd(post):6.0e} "
              f"{(f'{mean(ppl):8.4f}+-{sd(ppl):5.3f}' if ppl else '-'):>15} {(f'{mean(kl):.5f}' if kl else '-'):>10}")

print("\n=== per-layer post block error (mean over seeds) ===")
for b in blocks:
    if not any((a, b, seeds[0]) in rows for a in arms):
        continue
    print(f"-- block {b} --")
    print(f"{'layer':22s} " + "".join(f"{a:>13s}" for a in arms))
    for nm in NAMES:
        cells = []
        for a in arms:
            v = lcol(a, b, nm, "post")
            cells.append(f"{mean(v):13.4e}" if v else f"{'-':>13s}")
        print(f"{nm:22s} " + "".join(cells))

print("\n=== per-layer H-weighted reconstruction objective J (pre -> post), mean over seeds ===")
for b in blocks:
    if not any((a, b, seeds[0]) in rows for a in arms):
        continue
    print(f"-- block {b} --")
    print(f"{'layer':22s} " + "".join(f"{a+' pre':>13s}{a+' post':>13s}" for a in arms))
    for nm in NAMES:
        cells = []
        for a in arms:
            jp, jq = lcol(a, b, nm, "J_pre"), lcol(a, b, nm, "J_post")
            cells.append(f"{mean(jp):13.4e}" if jp else f"{'-':>13s}")
            cells.append(f"{mean(jq):13.4e}" if jq else f"{'-':>13s}")
        print(f"{nm:22s} " + "".join(cells))

base = "base"
if base in arms:
    print("\n=== paired vs baseline, post block error (the verdict number) ===")
    for arm in arms:
        if arm in (base, "dup"):
            continue
        wins = []
        for b in blocks:
            ss = [s for s in seeds if (arm, b, s) in rows and (base, b, s) in rows]
            if not ss:
                continue
            d = [rows[(arm, b, s)]["block_post"] - rows[(base, b, s)]["block_post"] for s in ss]
            dp = [rows[(arm, b, s)]["block_pre"] - rows[(base, b, s)]["block_pre"] for s in ss]
            bp = mean([rows[(base, b, s)]["block_post"] for s in ss])
            bq = mean([rows[(base, b, s)]["block_pre"] for s in ss])
            win = mean(d) < 0 and abs(mean(d)) > sd(d)
            wins.append(win)
            print(f"  {arm:8s} blk{b} n={len(ss)}: d_pre={mean(dp):+.3e} ({mean(dp)/bq:+7.2%})   "
                  f"d_post={mean(d):+.3e}+-{sd(d):.1e} ({mean(d)/bp:+7.2%})   {'WIN' if win else 'no'}")
        if wins:
            print(f"  -> {arm}: {'PROMOTE to full model' if all(wins) and len(wins) == len(blocks) else 'do not promote'}")

    if "dup" in arms:
        print("\n=== determinism floor (dup vs base, identical config and seed) ===")
        for b in blocks:
            for s in seeds:
                if ("dup", b, s) in rows and (base, b, s) in rows:
                    a1, a2 = rows[("dup", b, s)]["block_post"], rows[(base, b, s)]["block_post"]
                    print(f"  blk{b} seed{s}: base={a2:.9e}  dup={a1:.9e}  "
                          f"reldiff={abs(a1-a2)/a2:.3e}  {'BIT-IDENTICAL' if a1 == a2 else 'differs'}")
