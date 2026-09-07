"""Build results/nanoquant/gauge/stage1_down.md from the arm JSONs."""
import glob
import json
import os

GC = "/home/work/exp/artifacts/gauge/runs"
OUT = "/home/work/exp/results/nanoquant/gauge"
DELTA = 0.0031
NAMES = ['self_attn.q_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'self_attn.k_proj',
         'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']

BASE_ORDER = ["0a", "0a_dup", "0b_100", "0b_200", "2_refreshed", "2_frozen",
              "2_refreshed_rand", "2_frozen_rand", "3_100", "3_200"]
BASE_LABEL = {
    "0a": "0a  baseline (R = I)",
    "0a_dup": "0a-dup  determinism replay",
    "0b_100": "0b_100  extra STE x100",
    "0b_200": "0b_200  extra STE x200",
    "2_refreshed": "2  ITQ, refreshed magnitudes, R0 = I",
    "2_frozen": "2  ITQ, frozen magnitudes, R0 = I",
    "2_refreshed_rand": "2  ITQ, refreshed magnitudes, R0 random",
    "2_frozen_rand": "2  ITQ, frozen magnitudes, R0 random",
    "3_100": "3  functional gauge lr=3e-3 @100",
    "3_200": "3  functional gauge lr=3e-3 @200",
}


def load():
    recs = {}
    for f in glob.glob(f"{GC}/*.json"):
        r = json.load(open(f))
        recs[r["arm"]] = r
    return recs


def build_index(R):
    """ORDER / LABEL / MATCH, extended with whatever gauge-lr cells exist."""
    order = [k for k in BASE_ORDER if k in R]
    label = dict(BASE_LABEL)
    match = {"3_100": "0b_100", "3_200": "0b_200",
             "2_refreshed": "0b_200", "2_frozen": "0b_200",
             "2_refreshed_rand": "0b_200", "2_frozen_rand": "0b_200"}
    extra = sorted([k for k in R if k.startswith("3lr") and k.endswith(("_100", "_200"))],
                   key=lambda k: (-float(k[3:].rsplit("_", 1)[0]), k))
    for k in extra:
        lr, ck = k[3:].rsplit("_", 1)
        label[k] = f"3  functional gauge lr={lr} @{ck}"
        match[k] = f"0b_{ck}"
    return order + extra, label, match


def fmt(x, n=6):
    return "—" if x is None else f"{x:.{n}f}"


def pct(a, b):
    if a is None or b is None:
        return "—"
    return f"{100.0 * (a - b) / b:+.2f}%"


def main():
    R = load()
    ORDER, LABEL, MATCH = build_index(R)
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
    for k in [x for x in ORDER if x.startswith(("2_", "3_", "3lr"))]:
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
    for k in [x for x in ORDER if x.startswith("2_")]:
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
    searches = sorted(glob.glob(f"{GC}/arm3*_search.json"),
                      key=lambda f: -json.load(open(f))["lr"])
    if searches:
        ident = R["0a"]["calib_func_loss_at_gauge"] if "0a" in R else None
        A("## Arm 3 — functional latent gauge: calibration objective vs gauge step size")
        A("")
        A("Adam on the block-Cayley parameters, 200 steps, calibration split only. "
          "`L_func` here is the **full-calibration** functional loss (all 128 sequences), "
          "probed every 25 steps. The identity gauge sits at "
          f"**{ident:.6f}**; a cell that never goes below that never descended.")
        A("")
        probes = [25, 50, 75, 100, 125, 150, 175, 200]
        A("| gauge lr | " + " | ".join(f"step {p}" for p in probes) + " |")
        A("|---" * (len(probes) + 1) + "|")
        for f in searches:
            sr = json.load(open(f))
            d = {p["step"]: p["calib_L_func"] for p in sr.get("calib_probe", [])}
            A(f"| {sr['lr']:g} | " + " | ".join(f"{d[p]:.5f}" if p in d else "—" for p in probes) + " |")
        A("")
        A("Section 4.2 logging — calibration functional loss at each checkpoint, "
          "before and after the export statistics are re-extracted:")
        A("")
        A("| gauge lr | checkpoint | L_func pre-export stats | L_func post-export stats |")
        A("|---|---|---|---|")
        for f in searches:
            sr = json.load(open(f))
            for k, v in sorted(sr["checkpoints"].items(), key=lambda kv: int(kv[0])):
                A(f"| {sr['lr']:g} | {k} | {v['calib_func_loss_pre_export_stats']:.6f} | "
                  f"{v['calib_func_loss_post_export_stats']:.6f} |")
        A("")

    # line search
    lsf = "/home/work/exp/artifacts/gauge/linesearch.json"
    if os.path.exists(lsf):
        ls = json.load(open(lsf))
        ts = [d["t"] for d in ls["descent"]]
        A("## Line search at `R = I` — is the ADMM basis a local optimum?")
        A("")
        A(f"Functional gradient accumulated over **all {ls['n_grad']}** calibration "
          f"sequences at `R = I` (so this is not mini-batch noise), "
          f"`||g||_F = {ls['grad_norm']:.3e}`, normalised to a unit direction. "
          "`t` is the total Frobenius displacement of the Cayley parameters — on "
          "the same scale the sweep uses (Adam at lr 1e-5 for 200 steps travels "
          "about 0.45; at lr 3e-3, about 135). Calibration data only.")
        A("")
        A("| t | " + " | ".join(f"{t:g}" for t in ts) + " |")
        A("|---" * (len(ts) + 1) + "|")
        A("| **−grad**, post-export | " +
          " | ".join(f"{d['calib_L_post_export']:.6f}" for d in ls["descent"]) + " |")
        A("| **−grad**, pre-export | " +
          " | ".join(f"{d['calib_L_pre_export']:.6f}" for d in ls["descent"]) + " |")
        for i, row in enumerate(ls["random"]):
            A(f"| random dir {i}, post-export | " +
              " | ".join(f"{d['calib_L_post_export']:.6f}" for d in row) + " |")
        A("")
        best = min(d["calib_L_post_export"] for d in ls["descent"])
        ident = ls["descent"][0]["calib_L_post_export"]
        A(f"Best point anywhere on the steepest-descent ray: **{best:.6f}** against "
          f"the identity's **{ident:.6f}** — an improvement of "
          f"**{100*(ident-best)/ident:.3f}%**, twenty times below the 0.19% replay "
          "floor. Across the whole range where the curve is actually moving "
          "(t = 0.3 to t = 10) the gradient direction rises **faster than a random "
          "one** — at t = 1, 0.1313 against 0.1255/0.1256/0.1255; at t = 3, 0.1754 "
          "against 0.1326/0.1329/0.1326 — so the STE gradient through the hard sign "
          "is not merely uninformative about the gauge, it is anti-correlated with "
          "the true objective at any step size large enough to move anything. "
          "(By t = 30 every direction has saturated near the random-gauge error "
          "and the ordering stops meaning anything.)")
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
