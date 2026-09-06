"""Marchenko-Pastur bulk-edge estimation for FP weight matrices (Test C gate)."""
import json, math, sys, time
import numpy as np
import torch


def mp_median_sv(beta, n_grid=200001):
    """Median of the MP singular-value law for an m x n matrix (beta=m/n<=1),
    in units of s/sqrt(n) with unit-variance entries."""
    lo, hi = (1 - math.sqrt(beta))**2, (1 + math.sqrt(beta))**2
    # midpoints: avoids the 1/lambda singularity at lo=0 when beta==1
    edges = np.linspace(lo, hi, n_grid)
    lam = 0.5 * (edges[1:] + edges[:-1])
    dens = np.sqrt(np.clip((hi - lam) * (lam - lo), 0, None)) / (2 * math.pi * beta * lam)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(lam))])
    cdf /= cdf[-1]
    lam_med = float(np.interp(0.5, cdf, lam))
    return math.sqrt(lam_med)


def rmt_stats(W, thresh_mult=1.05):
    """W: (d_out, d_in) float32 tensor. Returns dict of RMT diagnostics."""
    d_out, d_in = W.shape
    m, n = min(d_out, d_in), max(d_out, d_in)
    beta = m / n
    s = torch.linalg.svdvals(W.double()).cpu().numpy()
    s_med = float(np.median(s))
    mu_beta = mp_median_sv(beta)
    sigma_hat = s_med / (math.sqrt(n) * mu_beta)
    edge = sigma_hat * (math.sqrt(d_out) + math.sqrt(d_in))
    thr = thresh_mult * edge
    k_rmt = int((s > thr).sum())
    return dict(d_out=d_out, d_in=d_in, beta=beta, s_max=float(s[0]), s_med=s_med,
                mu_beta=mu_beta, sigma_hat=sigma_hat, edge=edge, thr=thr, k_rmt=k_rmt,
                energy_top_k=float((s[:k_rmt]**2).sum() / (s**2).sum()) if k_rmt > 0 else 0.0,
                s_head=[float(x) for x in s[:12]])


def main():
    from nanoquant.utils.load_utils import load_model
    from nanoquant.utils.utils import get_layers_to_factorize, get_decoder_layers, find_layers, calculate_ranks
    MID = sys.argv[1]
    out = sys.argv[2]
    model = load_model(MID, 2048, "cpu")
    names = get_layers_to_factorize(model.config.model_type)
    qc = {'bits': 1.0, 'admm_type': 'nanoquant'}
    ranks = calculate_ranks(model, names, qc)
    res = {}
    t0 = time.time()
    for i, blk in enumerate(get_decoder_layers(model)):
        sub = find_layers(blk)
        for nm in names:
            if nm not in sub:
                continue
            W = sub[nm].weight.data.float()
            st = rmt_stats(W)
            st['rank_r'] = ranks[f"{i}.{nm}"]
            st['residual_rank_cost'] = 16 * st['k_rmt']
            res[f"{i}.{nm}"] = st
            print(f"{i}.{nm:20s} {st['d_out']}x{st['d_in']} r={st['rank_r']:4d} "
                  f"sigma={st['sigma_hat']:.5f} edge={st['edge']:.3f} smax={st['s_max']:.3f} "
                  f"k_RMT={st['k_rmt']:4d} 16k={st['residual_rank_cost']:5d} "
                  f"E_top={st['energy_top_k']:.4f}  [{time.time()-t0:.0f}s]", flush=True)
    json.dump(res, open(out, 'w'), indent=1)
    ks = [v['k_rmt'] for v in res.values()]
    print(f"\nk_RMT: min={min(ks)} med={int(np.median(ks))} max={max(ks)} mean={np.mean(ks):.1f}")
    print(f"layers with k_RMT<=2: {sum(1 for k in ks if k<=2)}/{len(ks)}")


if __name__ == "__main__":
    main()
