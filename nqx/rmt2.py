"""Follow-up spectral diagnostics for Test C: is the spectrum MP-bulk-plus-spikes,
or heavy-tailed with no well-defined split?"""
import json, math, sys
import numpy as np, torch
sys.path.insert(0, '/home/work/exp/nqx')
from rmt import mp_median_sv


def hill_alpha(s, frac):
    """Hill estimator of the power-law tail exponent of the ESD (eigenvalues lam=s^2).
    Martin & Mahoney report alpha in ~[2,6] for well-trained layers."""
    lam = np.sort(s.astype(np.float64) ** 2)[::-1]
    k = max(10, int(len(lam) * frac))
    k = min(k, len(lam) - 1)
    top = lam[:k]
    return 1.0 + k / np.sum(np.log(top / lam[k]))


def main():
    from nanoquant.utils.load_utils import load_model
    from nanoquant.utils.utils import get_layers_to_factorize, get_decoder_layers, find_layers, calculate_ranks
    MID, out = sys.argv[1], sys.argv[2]
    model = load_model(MID, 2048, "cpu")
    names = get_layers_to_factorize(model.config.model_type)
    ranks = calculate_ranks(model, names, {'bits': 1.0, 'admm_type': 'nanoquant'})
    res = {}
    for i, blk in enumerate(get_decoder_layers(model)):
        sub = find_layers(blk)
        for nm in names:
            if nm not in sub:
                continue
            W = sub[nm].weight.data.float()
            d_out, d_in = W.shape
            m, n = min(d_out, d_in), max(d_out, d_in)
            beta = m / n
            s = torch.linalg.svdvals(W.double()).cpu().numpy()
            sigma = np.median(s) / (math.sqrt(n) * mp_median_sv(beta))
            edge = sigma * (math.sqrt(d_out) + math.sqrt(d_in))
            k = int((s > 1.05 * edge).sum())
            # largest multiplicative gap in the top half of the spectrum
            top = s[:max(k * 2, 64)]
            ratios = top[:-1] / np.maximum(top[1:], 1e-30)
            gi = int(np.argmax(ratios[4:])) + 4          # ignore the first few
            r = ranks[f"{i}.{nm}"]
            res[f"{i}.{nm}"] = dict(
                d_out=d_out, d_in=d_in, r=r, k_rmt=k, edge=edge, s_max=float(s[0]),
                alpha_10=hill_alpha(s, 0.10), alpha_25=hill_alpha(s, 0.25), alpha_50=hill_alpha(s, 0.50),
                max_gap_ratio=float(ratios[4:].max()), gap_at=gi,
                gap_at_k=float(s[k - 1] / s[k]) if 0 < k < len(s) else None,
                frac_above_edge=k / len(s),
                resid_bpw=16 * k * (d_out + d_in) / (d_out * d_in),
                nq_bpw=(r * (d_out + d_in) + 16 * (d_out + d_in)) / (d_out * d_in),
            )
        print(f"block {i} done", flush=True)
    json.dump(res, open(out, 'w'), indent=1)


if __name__ == "__main__":
    main()
