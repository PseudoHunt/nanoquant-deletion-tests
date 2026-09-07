"""Build results/nanoquant/gauge/stage1_down.md from the arm JSONs."""
import glob
import json
import os

GC = "/home/work/exp/artifacts/gauge/runs"
OUT = "/home/work/exp/results/nanoquant/gauge"
DELTA = 0.0031
NAMES = ['self_attn.q_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'self_attn.k_proj',
         'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']

MATCH = {"3_100": "0b_100", "3_200": "0b_200",
         "2_refreshed": "0b_200", "2_frozen": "0b_200"}
ORDER = ["0a", "0a_dup", "0b_100", "0b_200", "2_refreshed", "2_frozen", "3_100", "3_200"]
LABEL = {
    "0a": "0a  baseline (R = I)",
    "0a_dup": "0a-dup  determinism replay",
    "0b_100": "0b_100  extra STE x100",
    "0b_200": "0b_200  extra STE x200",
    "2_refreshed": "2  ITQ, refreshed magnitudes",
    "2_frozen": "2  ITQ, frozen magnitudes",
    "3_100": "3  functional gauge @100",
    "3_200": "3  functional gauge @200",
}


def load():
    recs = {}
    for f in glob.glob(f"{GC}/*.json"):
        r = json.load(open(f))
        recs[r["arm"]] = r
    return recs


def fmt(x, n=6):
    return "—" if x is None else f"{x:.{n}f}"


def pct(a, b):
    if a is None or b is None:
        return "—"
    return f"{100.0 * (a - b) / b:+.2f}%"


def main():
    R = load()
    base = R.get("0a")
    lines = []
    A = lines.append

    A("| arm | E_ADMM | E_gauge pre-export | E_gauge | E_final | Δ vs 0a | Δ vs matched 0b | sign Δ U | sign Δ V | wall s |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for k in ORDER:
        if k not in R:
            continue
        r = R[k]
        m = MATCH.get(k)
        mb = R.get(m, {}).get("E_final") if m else None
        A(f"| **{LABEL[k]}** | {fmt(r.get('E_ADMM'))} | {fmt(r.get('E_gauge_pre_export'))} | "
          f"{fmt(r.get('E_gauge'))} | **{fmt(r.get('E_final'))}** | "
          f"{pct(r.get('E_final'), base.get('E_final')) if base else '—'} | "
          f"{pct(r.get('E_final'), mb)} | "
          f"{r.get('gauge_sign_delta_U', 0) * 100:.2f}% | {r.get('gauge_sign_delta_V', 0) * 100:.2f}% | "
          f"{r.get('wall_s', 0):.0f} |")
    A("")

    # gate
    A("## Stage-1 success gate")
    A("")
    A(f"A gauge checkpoint passes iff `E_final(gauge) < E_final(0a)·(1−δ)` **and** "
      f"`E_final(gauge) < E_final(0b_matched)·(1−δ)`, with δ = {DELTA}.")
    A("")
    A("| checkpoint | E_final | 0a threshold | matched 0b | 0b threshold | verdict |")
    A("|---|---|---|---|---|---|")
    for k in ["2_refreshed", "2_frozen", "3_100", "3_200"]:
        if k not in R:
            continue
        e = R[k]["E_final"]
        t0 = base["E_final"] * (1 - DELTA)
        mb = R.get(MATCH[k], {}).get("E_final")
        t1 = mb * (1 - DELTA) if mb else None
        ok = (e < t0) and (t1 is not None and e < t1)
        A(f"| {LABEL[k]} | {e:.6f} | {t0:.6f} | {MATCH[k]} = {fmt(mb)} | {fmt(t1)} | "
          f"**{'PASS' if ok else 'FAIL'}** |")
    A("")

    # Arm 1
    if "1" in R:
        r = R["1"]
        s = r["stats"]
        A("## Arm 1 — random gauge sensitivity (16 Haar block-rotations, no Step 3)")
        A("")
        A(f"Seed {r['seed']}, block size {r['block_size']}, "
          f"{len(r['samples'])} samples on `{r['layer']}`.")
        A("")
        A("| | E_gauge |")
        A("|---|---|")
        A(f"| identity (R = I) | {r['E_gauge_identity']:.6f} |")
        A(f"| min | {s['min']:.6f} |")
        A(f"| median | {s['median']:.6f} |")
        A(f"| max | {s['max']:.6f} |")
        A(f"| std | {s['std']:.3e} |")
        A(f"| best rotation id | {s['best_id']} |")
        A("")
        A("| id | E_gauge | E_gauge pre-export | sign Δ U | sign Δ V |")
        A("|---|---|---|---|---|")
        for x in r["samples"]:
            A(f"| {x['id']} | {x['E_gauge']:.6f} | {x['E_gauge_pre_export']:.6f} | "
              f"{x['sign_delta_U']*100:.2f}% | {x['sign_delta_V']*100:.2f}% |")
        A("")

    # Arm 2 traces
    for k in ["2_refreshed", "2_frozen"]:
        if k not in R:
            continue
        r = R[k]
        A(f"## Arm 2 — {LABEL[k]}, {r['rounds']}-round trace")
        A("")
        A("| round | " + " | ".join(str(t["round"]) for t in r["trace"]) + " |")
        A("|---" * (len(r["trace"]) + 1) + "|")
        A("| E_gauge | " + " | ".join(f"{t['E_gauge']:.5f}" for t in r["trace"]) + " |")
        A("")

    # Arm 3
    if "3" in R or os.path.exists(f"{GC}/arm3_search.json"):
        sr = json.load(open(f"{GC}/arm3_search.json"))
        A("## Arm 3 — functional latent gauge, 200-step trace")
        A("")
        A(f"Adam, lr {sr['lr']}, block size {sr['block_size']}, calibration split only, "
          f"{sr['steps']} steps.")
        A("")
        tr = sr["trace"]
        idx = [0] + list(range(9, len(tr), 10))
        A("| step | " + " | ".join(str(tr[i]["step"]) for i in idx) + " |")
        A("|---" * (len(idx) + 1) + "|")
        A("| L_func | " + " | ".join(f"{tr[i]['loss']:.5f}" for i in idx) + " |")
        A("")
        A("Section 4.2 logging — calibration functional loss at each checkpoint, "
          "before and after the export statistics are re-extracted:")
        A("")
        A("| checkpoint | L_func pre-export stats | L_func post-export stats |")
        A("|---|---|---|")
        for k, v in sorted(sr["checkpoints"].items(), key=lambda kv: int(kv[0])):
            A(f"| {k} | {v['calib_func_loss_pre_export_stats']:.6f} | "
              f"{v['calib_func_loss_post_export_stats']:.6f} |")
        A("")

    # per-layer post table
    A("## Per-layer post-Step-3 diagnostic (cumulative curve)")
    A("")
    A("Layers 1..k at their tuned signs, the rest back at their pre-Step-3 signs. "
      "Mechanism analysis only — the joint Step 3 may redistribute compensation "
      "across layers, so the final gain is not required to stay localised to `down_proj`.")
    A("")
    have = [k for k in ORDER if k in R and "layer_post_curve" in R[k]]
    A("| layer | " + " | ".join(LABEL[k].split()[0] for k in have) + " |")
    A("|---" * (len(have) + 1) + "|")
    for n in NAMES:
        A(f"| `{n}` | " + " | ".join(f"{R[k]['layer_post_curve'][n]:.5f}" for k in have) + " |")
    A("")

    # J table
    A("## J diagnostic (H-weighted layer reconstruction objective), pre / post Step 3")
    A("")
    A("| layer | " + " | ".join(f"{LABEL[k].split()[0]} pre / post" for k in have) + " |")
    A("|---" * (len(have) + 1) + "|")
    for n in NAMES:
        cells = []
        for k in have:
            j = R[k]["J"][n]
            cells.append(f"{j['J_pre']:.4f} / {j['J_post']:.4f}")
        A(f"| `{n}` | " + " | ".join(cells) + " |")
    A("")

    # Step-3 sign flips
    A("## Step-3 sign-flip fraction on `mlp.down_proj`")
    A("")
    A("| arm | U flips | V flips | block mean |")
    A("|---|---|---|---|")
    for k in have:
        f = R[k]["step3_flips"]
        A(f"| {LABEL[k].split()[0]} | {f['mlp.down_proj.U_latent']*100:.2f}% | "
          f"{f['mlp.down_proj.V_latent']*100:.2f}% | {R[k]['step3']['flip_mean']*100:.2f}% |")
    A("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(main())
