"""Parse a stock nanoquant.main log into the numbers repro.md needs."""
import json, re, sys
import numpy as np

NAMES = ['self_attn.q_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'self_attn.k_proj',
         'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']

def parse(path):
    txt = open(path, errors='ignore').read()
    rec = re.findall(r'\[\s*([0-9.]+)\] \t\tADMM weight recon error: raw=([0-9.eE+-]+), norm=([0-9.eE+-]+), '
                     r'per_el=([0-9.eE+-]+), ADMM time=([0-9.]+)s', txt)
    layers = []
    for i, (t, raw, nrm, pel, at) in enumerate(rec):
        layers.append({"block": i // 7, "name": NAMES[i % 7], "t": float(t), "raw": float(raw),
                       "norm": float(nrm), "per_el": float(pel), "admm_s": float(at)})
    blk_ppl = [(int(b), float(p)) for b, p in re.findall(r'Block (\d+): Test Data PPL\s+= ([0-9.]+)', txt)]
    final = dict(re.findall(r'Perplexity on (\w+): ([0-9.]+)', txt))
    # stage boundaries: walk the (1/3)->(2/3)->(3/3) markers in order
    marks = [(float(t), int(k)) for t, k in re.findall(r'\[\s*([0-9.]+)\] \t\((\d)/3\)', txt)]
    ppl_marks = [float(t) for t in re.findall(r'\[\s*([0-9.]+)\] \t\tBlock \d+: Test Data PPL', txt)]
    kd = re.search(r'\[\s*([0-9.]+)\].*Performing model-level KD tuning', txt)
    ends = re.findall(r'\[\s*([0-9.]+)\].*Results:', txt)
    t_kd = float(kd.group(1)) if kd else None
    tn = tf = 0.0
    for i, (t, k) in enumerate(marks):
        nxt = marks[i + 1][0] if i + 1 < len(marks) else (t_kd or t)
        if k == 1:
            tn += marks[i + 1][0] - t if i + 1 < len(marks) else 0.0
        elif k == 3:
            # refinement runs until the next (1/3); a block boundary inserts a PPL eval
            gap_end = nxt
            for pt in ppl_marks:
                if t < pt < nxt:
                    gap_end = pt
                    break
            tf += gap_end - t
    out = {
        "n_layers": len(layers), "layers": layers, "block_ppl": blk_ppl,
        "final_ppl": {k: float(v) for k, v in final.items()},
        "admm_total_s": sum(l["admm_s"] for l in layers),
        "admm_mean_s": float(np.mean([l["admm_s"] for l in layers])) if layers else None,
        "recon_norm_mean": float(np.mean([l["norm"] for l in layers])) if layers else None,
        "t_kd_start": t_kd, "t_end": float(ends[-1]) if ends else None,
        "t_block_stage_start": marks[0][0] if marks else None,
        "tune_nonfact_s": tn, "block_refine_s": tf,
        "kd_s": (float(ends[-1]) - t_kd) if (ends and t_kd) else None,
    }
    by = {}
    for n in NAMES:
        sel = [l for l in layers if l["name"] == n]
        by[n] = {"n": len(sel), "recon_norm_mean": float(np.mean([s["norm"] for s in sel])),
                 "recon_norm_std": float(np.std([s["norm"] for s in sel])),
                 "admm_s_mean": float(np.mean([s["admm_s"] for s in sel]))}
    out["by_layer"] = by
    return out

if __name__ == "__main__":
    r = parse(sys.argv[1])
    json.dump(r, open(sys.argv[2], "w"), indent=1)
    print(f"layers={r['n_layers']}  recon_norm_mean={r['recon_norm_mean']}")
    print(f"final PPL: {r['final_ppl']}")
    print(f"ADMM total={r['admm_total_s']:.0f}s  tune_nonfact={r['tune_nonfact_s']:.0f}s  "
          f"block_refine={r['block_refine_s']:.0f}s  KD+eval={r['kd_s']:.0f}s  total={r['t_end']:.0f}s")
    for n, v in r["by_layer"].items():
        print(f"  {n:20s} n={v['n']:2d} recon_norm={v['recon_norm_mean']:.4f}+-{v['recon_norm_std']:.4f} "
              f"admm={v['admm_s_mean']:.2f}s")
