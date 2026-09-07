"""Build results/nanoquant/gauge/stage1d_givens.md from the discrete-search JSONs."""
import glob
import json
import math
import os

GC = "/home/work/exp/artifacts/gauge"
OUT = "/home/work/exp/results/nanoquant/gauge"
DELTA = 0.0031


def fmt(x, n=6):
    return "—" if x is None else f"{x:.{n}f}"


def main():
    A = [].append if False else None
    lines = []
    A = lines.append

    pilot = json.load(open(f"{GC}/givens_pilot_b0.json"))
    vf = f"{GC}/quad_verify.json"
    ver = json.load(open(vf)) if os.path.exists(vf) else None
    runs = {}
    for f in glob.glob(f"{GC}/givens_descent_*.json"):
        r = json.load(open(f))
        runs[os.path.basename(f)[len("givens_descent_"):-5]] = r
    arms = {}
    for f in glob.glob(f"{GC}/runs/arm4*.json"):
        r = json.load(open(f))
        arms[r["arm"]] = r
    base = json.load(open(f"{GC}/runs/arm0a.json"))
    dup = json.load(open(f"{GC}/runs/arm0a_dup.json"))

    A("## The oracle")
    A("")
    if ver:
        A("| case | real bf16 forward | quadratic oracle | relative gap |")
        A("|---|---|---|---|")
        for c in ver["cases"]:
            A(f"| {c['case']} | {c['real_forward']:.8f} | {c['oracle']:.8f} | "
              f"{c['rel_gap']:.2e} |")
        A("")
    A("The absolute gap is largely a common offset and is **not** the quantity that "
      "matters for ranking candidates; what matters is the accuracy of the "
      "*difference* `eps_Delta = |dE_oracle - dE_real|`, which is measured against "
      "the real forward at every checkpoint of every descent run below.")
    A("")

    A("## Pilot — one 32-block, every plane, every reachable binary model")
    A("")
    A("| | |")
    A("|---|---|")
    A(f"| planes (all pairs in rank block 0) | {pilot['n_planes']} |")
    A(f"| distinct binary models per plane (mod pi/2) | {pilot['n_intervals_per_plane']} |")
    A(f"| **planes containing an improving model** | **{pilot['n_improving']} "
      f"({100*pilot['n_improving']/pilot['n_planes']:.1f}%)** |")
    imp = [p for p in pilot["planes"] if p["improves"]]
    fl = sorted(p["n_flip"] for p in imp)
    df = sorted(p["delta_f"] for p in imp)
    med = lambda v: v[len(v) // 2]
    A(f"| sign flips at the best angle (relabel-aware) | median **{med(fl)}**, "
      f"min {fl[0]}, max {fl[-1]} |")
    A(f"| — as a fraction of the 20480 touched entries | {100*med(fl)/20480:.2f}% |")
    A(f"| gain per plane | median {100*(med(df)/1.059359e6)/pilot['E0_oracle']:+.5f}%, "
      f"best {100*(df[0]/1.059359e6)/pilot['E0_oracle']:+.5f}% |")
    A(f"| incremental-vs-direct gate | {pilot['gate_incremental_vs_direct_rel']:.2e} |")
    ns = sorted(p.get("noise", 0.0) for p in pilot["planes"])
    A(f"| fp64 residual arithmetic noise per plane | median {med(ns):.1e}, max {ns[-1]:.1e} |")
    A("")

    for name, r in sorted(runs.items()):
        A(f"## Coordinate descent — `{name}`")
        A("")
        A(f"Rank blocks {r['rank_blocks']}, {r['n_planes']} planes, strategy "
          f"`{r['strategy']}`. Calibration data only.")
        A("")
        if r.get("sweeps"):
            A("| sweep | accepted | E oracle | E real bf16 | cum vs start (oracle) | "
              "cum vs start (real) |")
            A("|---|---|---|---|---|---|")
            for sw in r["sweeps"]:
                A(f"| {sw['sweep']} | {sw['accepted']} | {sw['E_oracle']:.8f} | "
                  f"{sw['E_real']:.8f} | {sw['cum_gain_oracle_pct']:+.4f}% | "
                  f"**{sw['cum_gain_real_pct']:+.4f}%** |")
            A("")
        if r.get("real_checks"):
            A("| accepts | E oracle | E real bf16 | cum ΔE oracle | cum ΔE real | "
              "eps_Delta (cum) | sign Δ U | sign Δ V | reversals |")
            A("|---|---|---|---|---|---|---|---|---|")
            acc = {a["k"]: a for a in r["accepts"]}
            for c in r["real_checks"]:
                a = acc.get(c["k"], {})
                A(f"| {c['k']} | {c['E_oracle']:.8f} | {c['E_real']:.8f} | "
                  f"{c['cum_dE_oracle']:+.3e} | {c['cum_dE_real']:+.3e} | "
                  f"{c['eps_delta_cum']:.2e} | {a.get('sign_delta_U',0)*100:.3f}% | "
                  f"{a.get('sign_delta_V',0)*100:.3f}% | {a.get('reversals',0)} |")
            A("")
            xs = [math.log(c["k"]) for c in r["real_checks"] if c["cum_dE_real"] < 0]
            ys = [math.log(-c["cum_dE_real"]) for c in r["real_checks"] if c["cum_dE_real"] < 0]
            if len(xs) > 2:
                n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
                sl = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / sum((x-mx)**2 for x in xs)
                A(f"Log-log fit of cumulative real gain against accepted moves: "
                  f"**gain ∝ m^{sl:.2f}** (1.0 = independent/additive, "
                  f"0.5 = random-walk/interfering).")
                A("")
        A("| | held-out block error |")
        A("|---|---|")
        A(f"| `E_ADMM` (start) | {fmt(r.get('E_ADMM_heldout'))} |")
        A(f"| `E_gauge`, fixed ADMM scales | {fmt(r.get('E_gauge_heldout_pre_export'))} |")
        A(f"| `E_gauge`, full `Q_NQ` export | {fmt(r.get('E_gauge_heldout_qnq'))} |")
        A(f"| planes visited / moves accepted | {r.get('planes_visited')} / "
          f"{r.get('accepted_total')} |")
        A("")

    if arms:
        A("## Through the identical common Step 3")
        A("")
        A("| arm | E_ADMM | E_gauge pre-export | E_gauge | E_final | Δ vs 0a | "
          "Δ vs 0b_200 | sign Δ U | sign Δ V |")
        A("|---|---|---|---|---|---|---|---|---|")
        b0 = base["E_final"]
        try:
            b200 = json.load(open(f"{GC}/runs/arm0b_200.json"))["E_final"]
        except Exception:
            b200 = None
        A(f"| **0a baseline** | {base['E_ADMM']:.6f} | — | {base['E_gauge']:.6f} | "
          f"**{b0:.6f}** | — | — | 0.00% | 0.00% |")
        A(f"| 0a duplicate (replay floor) | {dup['E_ADMM']:.6f} | — | {dup['E_gauge']:.6f} | "
          f"**{dup['E_final']:.6f}** | {100*(dup['E_final']-b0)/b0:+.2f}% | — | 0.00% | 0.00% |")
        for k, r in sorted(arms.items()):
            A(f"| **{k}** | {r['E_ADMM']:.6f} | {fmt(r.get('E_gauge_pre_export'))} | "
              f"{r['E_gauge']:.6f} | **{r['E_final']:.6f}** | "
              f"{100*(r['E_final']-b0)/b0:+.2f}% | "
              f"{(f'{100*(r[chr(69)+chr(95)+chr(102)+chr(105)+chr(110)+chr(97)+chr(108)]-b200)/b200:+.2f}%') if b200 else '—'} | "
              f"{r.get('gauge_sign_delta_U',0)*100:.3f}% | {r.get('gauge_sign_delta_V',0)*100:.3f}% |")
        A("")
        A(f"Gate: a cell passes only if it beats **both** 0a and its matched 0b by more "
          f"than δ = {DELTA}. 0a threshold = {b0*(1-DELTA):.6f}.")
        A("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(main())
