"""Discrete gauge search: Givens-plane coordinate descent on the TRUE binary
objective, with exact enumeration of the sign-pattern breakpoints.

No STE.  No surrogate gradient.  Stage 1 showed the continuous surrogate is
uninformative where the binary objective is constant and anti-correlated with it
where signs start to change, so this searches the discrete structure directly.

The move class
--------------
A Givens rotation R_ij(theta) in the rank space mixes only latent coordinates
i and j, so it touches only columns i, j of U and rows i, j of V:

    (U R)[:, i] =  cos.u_i + sin.u_j        (V R)[i, :] =  cos.v_i + sin.v_j
    (U R)[:, j] = -sin.u_i + cos.u_j        (V R)[j, :] = -sin.v_i + cos.v_j

and `U R (V R)^T = U V^T` exactly.  The deployed sign patterns are therefore
piecewise constant in theta, changing only where some entry crosses zero:

    cos.u_i[m] + sin.u_j[m] = 0   =>   theta = atan2(-u_i[m], u_j[m])  (mod pi)

Each of the four affected vectors contributes one breakpoint per coordinate, so a
plane has 2*out + 2*in breakpoints (20480 for down_proj) dividing [0, pi) into
that many intervals of constant sign pattern.  The objective is pi-periodic:
R(theta+pi) negates both columns of U and both rows of V for coordinates i, j
simultaneously, and b_k c_k is invariant under negating both.  So enumerating
[0, pi) is exhaustive, and one representative angle per interval evaluates every
distinct binary model this plane can reach.

The cheap exact objective
-------------------------
With W = D_o M D_p, M = B_U B_V, and only rank coordinates i, j changing,
M = M_rest + b_i c_i + b_j c_j, and the oracle's quadratic collapses to a handful
of inner products (constants dropped):

    f = -2 sum_k b_k^T G~ c_k + 2 sum_k b_k^T N c_k + sum_{k,l} S_lk P_kl

    G~ = D_o G D_p,  H~ = D_p H D_p,  N = D_o^2 M_rest H~
    S_lk = b_l . (so^2 * b_k),  P_kl = c_k^T H~ c_l,   k, l in {i, j}

S_ii = S_jj = ||so||^2 is constant because b in {+-1}, so only S_ij varies.  And
N never has to be rebuilt per plane: with N_full = D_o^2 M H~ computed once,

    b_k^T N c_k = b_k^T N_full c_k - S_ki (c_i^T H~ c_k) - S_kj (c_j^T H~ c_k)

so a whole plane costs six batched matmuls against the candidate sign matrices.

Scales.  so / sp are held fixed during a plane sweep: a Givens rotation changes
2 of 1600 rank coordinates, so NanoQuant's mean-magnitude export scale moves by
O(1/800), and Step 3 re-learns both scale vectors anyway.  Accepted moves are
re-exported through the real `Q_NQ` (which does recompute them) and every
reported number comes from the real bf16 forward, never from this oracle.
"""
import argparse
import json
import math
import os
import time

import torch

from nanoquant.modules.hub import NanoQuantConfigDataclass  # noqa: F401  import-order guard

import instrument as I

from . import core as C
from . import harness as H
from . import quad as Q
from . import state as S

LAYER = "mlp.down_proj"


HALF_PI = math.pi / 2


def breakpoints(a_i, a_j):
    """Sign-change angles of the pair, reduced mod pi/2.

    The two crossings of a coordinate -- `cos.a_i + sin.a_j = 0` and
    `-sin.a_i + cos.a_j = 0` -- are exactly pi/2 apart (writing a_i = r cos(phi),
    a_j = r sin(phi), they are phi - pi/2 and phi), so mod pi/2 each coordinate
    contributes a single breakpoint at `atan2(a_j, a_i)`.
    """
    return torch.remainder(torch.atan2(a_j, a_i), HALF_PI)


class PlaneSearcher:
    """Holds everything shared across planes; `sweep(i, j)` searches one plane."""

    def __init__(self, oracle, base, device="cuda", chunk=1024, dtype=torch.float64):
        """fp64 throughout.

        `f` is a difference of terms of order 1e4 (`c^T H~ c` over 8192 entries of
        +-1) that cancel down to order 1e2, so fp32 leaves ~1.2e-3 of absolute
        noise -- about 0.25% of a median improving move.  That is tolerable for
        one move and not tolerable when several hundred are being accumulated,
        because banked numerical noise is indistinguishable from a banked gain.
        fp64 costs roughly 2x and removes the concern; the residual is still
        measured per plane and used to calibrate acceptance.
        """
        self.dev = device
        self.chunk = chunk
        self.dtype = dtype
        o = oracle.to(device)
        so, sp = o.so0.to(device).to(dtype), o.sp0.to(device).to(dtype)
        with C.no_tf32():
            self.Ht = sp.unsqueeze(1) * o.H.to(dtype) * sp.unsqueeze(0)   # [in, in]
            self.Gt = so.unsqueeze(1) * o.G.to(dtype) * sp.unsqueeze(0)   # [out, in]
        self.so2 = (so * so)
        self.so2_sum = float(self.so2.sum())
        self.T_sq, self.Y_sq = o.T_sq, o.Y_sq
        self.base = base
        # working factors, and the accumulated rotation relative to the ADMM basis
        self.U = base.U0.clone().to(dtype)                             # [out, rank]
        self.Vm = base.V0.transpose(0, 1).contiguous().to(dtype)       # math V [in, rank]
        self.R = torch.eye(base.rank, device=device, dtype=dtype)
        self.refresh()

    @torch.no_grad()
    def refresh(self):
        """Recompute B_U, B_V and N_full from the current U/Vm."""
        self.BU = C.pos_sign(self.U)                                   # [out, rank]
        self.BVm = C.pos_sign(self.Vm)                                 # [in, rank]
        with C.no_tf32():
            M = self.BU @ self.BVm.transpose(0, 1)                     # [out, in]
            self.N_full = (self.so2.unsqueeze(1) * M) @ self.Ht
        del M

    @torch.no_grad()
    def plane_ref(self, i, j):
        """The *current* (theta = 0) quantities the N-correction needs.

        N = D_o^2 M_rest H~ removes the current rank-i and rank-j outer products
        from M, so the correction crosses the **current** b_k0 / c_k0 with the
        **candidate** b_k(theta) / c_k(theta).  Getting this wrong (crossing
        candidate with candidate) silently returns the right answer at theta = 0
        and the wrong one everywhere else, which is why `sweep` asserts that
        f(0) reproduces the oracle.
        """
        b_i0 = C.pos_sign(self.U[:, i])
        b_j0 = C.pos_sign(self.U[:, j])
        c_i0 = C.pos_sign(self.Vm[:, i])
        c_j0 = C.pos_sign(self.Vm[:, j])
        with C.no_tf32():
            return {"w_i0": self.so2 * b_i0, "w_j0": self.so2 * b_j0,
                    "h_i0": self.Ht @ c_i0, "h_j0": self.Ht @ c_j0}

    @torch.no_grad()
    def f_terms(self, i, j, thetas, ref):
        """f(theta) for a batch of angles, up to a plane-dependent constant."""
        u_i, u_j = self.U[:, i], self.U[:, j]
        v_i, v_j = self.Vm[:, i], self.Vm[:, j]
        cs, sn = torch.cos(thetas).unsqueeze(1), torch.sin(thetas).unsqueeze(1)
        with C.no_tf32():
            Bi = C.pos_sign(cs * u_i + sn * u_j)                        # [T, out]
            Bj = C.pos_sign(-sn * u_i + cs * u_j)
            Ci = C.pos_sign(cs * v_i + sn * v_j)                        # [T, in]
            Cj = C.pos_sign(-sn * v_i + cs * v_j)

            HCi = self.Ht @ Ci.transpose(0, 1)                          # [in, T]
            HCj = self.Ht @ Cj.transpose(0, 1)
            P_ii = (Ci.transpose(0, 1) * HCi).sum(0)
            P_jj = (Cj.transpose(0, 1) * HCj).sum(0)
            P_ij = (Cj.transpose(0, 1) * HCi).sum(0)

            bGc = (Bi * (self.Gt @ Ci.transpose(0, 1)).transpose(0, 1)).sum(1) \
                + (Bj * (self.Gt @ Cj.transpose(0, 1)).transpose(0, 1)).sum(1)
            bNc_full = (Bi * (self.N_full @ Ci.transpose(0, 1)).transpose(0, 1)).sum(1) \
                     + (Bj * (self.N_full @ Cj.transpose(0, 1)).transpose(0, 1)).sum(1)

            # N-correction: CURRENT b_k0 / c_k0 crossed with CANDIDATE b_k / c_k
            corr = (Bi @ ref["w_i0"]) * (Ci @ ref["h_i0"]) \
                 + (Bi @ ref["w_j0"]) * (Ci @ ref["h_j0"]) \
                 + (Bj @ ref["w_i0"]) * (Cj @ ref["h_i0"]) \
                 + (Bj @ ref["w_j0"]) * (Cj @ ref["h_j0"])

            S_ij = ((self.so2.unsqueeze(0) * Bi) * Bj).sum(1)
            quad = self.so2_sum * (P_ii + P_jj) + 2.0 * S_ij * P_ij
            f = -2.0 * bGc + 2.0 * (bNc_full - corr) + quad

            # Flip count, modulo the coordinate relabelling.  R(theta + pi/2)
            # swaps rank coordinates i and j (with a sign) and leaves
            # b_i c_i + b_j c_j -- hence the model and f -- exactly invariant, so
            # a naive column-wise count reports ~half the touched entries as
            # "flipped" for a bit-identical model.  Score both identifications
            # and keep the smaller.
            b_i0 = C.pos_sign(u_i).unsqueeze(0); b_j0 = C.pos_sign(u_j).unsqueeze(0)
            c_i0 = C.pos_sign(v_i).unsqueeze(0); c_j0 = C.pos_sign(v_j).unsqueeze(0)
            flips_id = ((Bi != b_i0).sum(1) + (Bj != b_j0).sum(1)
                        + (Ci != c_i0).sum(1) + (Cj != c_j0).sum(1))
            flips_sw = ((Bi != b_j0).sum(1) + (Bj != -b_i0).sum(1)
                        + (Ci != c_j0).sum(1) + (Cj != -c_i0).sum(1))
            nflip = torch.minimum(flips_id, flips_sw)
        return f, nflip

    @torch.no_grad()
    def sweep(self, i, j):
        """Exact search over every distinct sign pattern this plane can reach."""
        bp = torch.cat([breakpoints(self.U[:, i], self.U[:, j]),
                        breakpoints(self.Vm[:, i], self.Vm[:, j])])
        bp = torch.sort(torch.unique(bp)).values
        # f is pi/2-periodic (see `breakpoints`), so [0, pi/2) is exhaustive:
        # every distinct binary model this plane can reach appears exactly once.
        mids = torch.cat([(bp[:-1] + bp[1:]) / 2,
                          torch.tensor([(bp[-1] + bp[0] + HALF_PI) / 2 % HALF_PI],
                                       device=bp.device)])
        mids = torch.cat([torch.zeros(1, device=bp.device), mids])      # index 0 = current
        ref = self.plane_ref(i, j)
        fs, nf = [], []
        for s in range(0, mids.numel(), self.chunk):
            f, n = self.f_terms(i, j, mids[s:s + self.chunk], ref)
            fs.append(f); nf.append(n)
        f = torch.cat(fs); n = torch.cat(nf)
        f0 = float(f[0])
        # Consistency: any candidate angle landing in the same sign-pattern cell as
        # theta = 0 must score exactly f0.  This catches an incorrect correction
        # term, which would agree at theta = 0 and be wrong everywhere else.
        same = (n == 0)
        drift = 0.0
        if bool(same.any()):
            drift = float((f[same] - f0).abs().max())
            scale = max(abs(f0), 1e-30)
            # Gross-bug gate only.  A candidate scoring the same model as theta = 0
            # must return f0 up to arithmetic noise; the candidate-x-candidate
            # correction bug this catches produced O(1) relative errors.  The
            # residual itself is returned as `noise` and used to calibrate
            # acceptance, so it is measured rather than assumed.
            assert drift / scale < 1e-3, \
                f"zero-flip candidates disagree with f(0) by {drift/scale:.2e} at plane ({i},{j})"
        k = int(torch.argmin(f))
        # Canonicalise the angle.  f is pi/2-periodic, so theta and theta - pi/2
        # give a bit-identical binary model; the second differs only by swapping
        # rank coordinates i and j.  Taking the representative in [-pi/4, pi/4)
        # keeps the accumulated factors in the original labelling, which is what
        # makes cumulative sign-flip counts mean anything over many moves.
        th = float(mids[k])
        if th >= math.pi / 4:
            th -= HALF_PI
        return {"i": i, "j": j, "n_intervals": int(mids.numel()) - 1,
                "f0": f0, "f_best": float(f[k]),
                "delta_f": float(f[k]) - f0,
                "theta": th, "n_flip": int(n[k]), "noise": drift,
                "n_zero_flip_cells": int(same.sum()),
                "improves": bool(float(f[k]) < f0)}

    @torch.no_grad()
    def apply(self, i, j, theta, track_R=True):
        """Rotate the working U/Vm by R_ij(theta) and refresh the cached state.

        The accumulated rotation R is tracked so the final state can be
        materialised through the *same* `materialize_gauge` / `Q_NQ` path every
        other arm uses, rather than by writing rotated factors in by hand.
        """
        c, s = math.cos(theta), math.sin(theta)
        with C.no_tf32():
            ui, uj = self.U[:, i].clone(), self.U[:, j].clone()
            self.U[:, i] = c * ui + s * uj
            self.U[:, j] = -s * ui + c * uj
            vi, vj = self.Vm[:, i].clone(), self.Vm[:, j].clone()
            self.Vm[:, i] = c * vi + s * vj
            self.Vm[:, j] = -s * vi + c * vj
            if track_R and self.R is not None:
                ri, rj = self.R[:, i].clone(), self.R[:, j].clone()
                self.R[:, i] = c * ri + s * rj
                self.R[:, j] = -s * ri + c * rj
        self.refresh()

    @torch.no_grad()
    def orthogonality_error(self):
        with C.no_tf32():
            I_ = torch.eye(self.R.shape[0], device=self.R.device, dtype=self.R.dtype)
            return float((self.R.transpose(0, 1) @ self.R - I_).norm())

    @torch.no_grad()
    def factor_consistency(self):
        """||U0 R - U_working||_F / ||U_working||_F -- the accumulated R must
        reproduce the factors the sweeps were actually scored against."""
        with C.no_tf32():
            Uc = self.base.U0.to(self.dtype) @ self.R
            return float((Uc - self.U).norm() / max(self.U.norm(), 1e-30))

    @torch.no_grad()
    def deployed_W(self):
        with C.no_tf32():
            M = self.BU @ self.BVm.transpose(0, 1)
            W = (self.base.so0.to(self.dtype).unsqueeze(1) * M) \
                * self.base.sp0.to(self.dtype).unsqueeze(0)
        return W.float()


# ---------------------------------------------------------------------------
def pilot(args):
    o = Q.Oracle(torch.load(f"{H.GCACHE}/quad_oracle.pt", weights_only=False))
    st = H.load_post_admm(seed=args.seed)
    sub = H.nq_sub(st["blk"])
    base = C.LayerBase(LAYER, sub[LAYER]).to("cuda")
    ps = PlaneSearcher(o, base)

    E0 = o.error(ps.deployed_W())
    print(f"[givens] oracle E at current point = {E0:.8f}", flush=True)

    # gate: f(0) + const must reproduce the oracle, and the incremental algebra
    # must agree with a from-scratch evaluation of a rotated weight
    i0, j0 = args.block * 32, args.block * 32 + 1
    r0 = ps.sweep(i0, j0)
    const = E0 * o.Y_sq - r0["f0"]
    ps_theta = r0["theta"]
    U_save, V_save = ps.U.clone(), ps.Vm.clone()
    ps.apply(i0, j0, ps_theta)
    E_direct = o.error(ps.deployed_W())
    E_pred = (const + r0["f_best"]) / o.Y_sq
    ps.U, ps.Vm = U_save, V_save
    ps.refresh()
    gate_rel = abs(E_pred - E_direct) / max(E_direct, 1e-30)
    print(f"[givens] GATE incremental-vs-direct: predicted {E_pred:.8f}  "
          f"direct {E_direct:.8f}  rel {gate_rel:.2e}", flush=True)
    assert gate_rel < 1e-5, "incremental Givens objective disagrees with the oracle"

    lo, hi = args.block * 32, args.block * 32 + 32
    pairs = [(i, j) for i in range(lo, hi) for j in range(i + 1, hi)]
    if args.max_planes:
        pairs = pairs[:args.max_planes]
    rec = {"mode": "pilot", "layer": LAYER, "rank_block": args.block,
           "coords": [lo, hi], "n_planes": len(pairs), "E0_oracle": E0, "planes": []}

    t0 = time.time()
    for n, (i, j) in enumerate(pairs):
        r = ps.sweep(i, j)
        rec["planes"].append(r)
        if (n + 1) % 50 == 0 or n == 0:
            imp = [p for p in rec["planes"] if p["improves"]]
            best = min((p["delta_f"] for p in rec["planes"]), default=0.0)
            print(f"[givens] plane {n+1:4d}/{len(pairs)}  improving so far: {len(imp)}  "
                  f"best df = {best:.6e}  ({time.time()-t0:.0f}s)", flush=True)
    rec["wall_s"] = time.time() - t0

    imp = [p for p in rec["planes"] if p["improves"]]
    rec["n_improving"] = len(imp)
    if imp:
        b = min(imp, key=lambda p: p["delta_f"])
        rec["best_plane"] = b
        rec["best_delta_E"] = b["delta_f"] / o.Y_sq
        rec["best_rel_gain_pct"] = 100.0 * (b["delta_f"] / o.Y_sq) / E0
    rec["n_intervals_per_plane"] = rec["planes"][0]["n_intervals"] if rec["planes"] else None
    rec["gate_incremental_vs_direct_rel"] = gate_rel

    # end-to-end confirmation of the single best move through the real bf16 forward
    if imp:
        b = rec["best_plane"]
        m = sub[LAYER]
        real_before = H.func_loss_over(st["blk"], st["cal_in"].to("cuda"),
                                       st["cal_out"].to("cuda"), st["kwargs"])
        ps.apply(b["i"], b["j"], b["theta"])
        with torch.no_grad():
            m.U_latent.data.copy_(ps.U.to(torch.bfloat16))
            m.V_latent.data.copy_(ps.Vm.transpose(0, 1).contiguous().to(torch.bfloat16))
        real_after_fixed = H.func_loss_over(st["blk"], st["cal_in"].to("cuda"),
                                            st["cal_out"].to("cuda"), st["kwargs"])
        rec["confirm"] = {
            "real_calib_before": real_before,
            "real_calib_after_fixed_scales": real_after_fixed,
            "oracle_predicted_after": (const_b := (E0 * o.Y_sq - b["f0"]) + b["f_best"]) / o.Y_sq,
        }
        rec["confirm"]["rel_gap_pred_vs_real"] = abs(
            rec["confirm"]["oracle_predicted_after"] - real_after_fixed) / real_after_fixed
        print(f"[givens] CONFIRM best move through the real forward: "
              f"{real_before:.8f} -> {real_after_fixed:.8f}  "
              f"(oracle predicted {rec['confirm']['oracle_predicted_after']:.8f}, "
              f"rel gap {rec['confirm']['rel_gap_pred_vs_real']:.2e})", flush=True)
    S.atomic_json(rec, f"{H.GCACHE}/givens_pilot_b{args.block}.json")
    print(f"[givens] PILOT: {len(imp)}/{len(pairs)} planes improve the true binary "
          f"calibration objective", flush=True)
    if imp:
        b = rec["best_plane"]
        print(f"[givens] best: plane ({b['i']},{b['j']}) theta={b['theta']:.5f} "
              f"flips={b['n_flip']}  dE={rec['best_delta_E']:.3e} "
              f"({rec['best_rel_gain_pct']:+.3f}% of E)", flush=True)
    return rec


# ---------------------------------------------------------------------------
@torch.no_grad()
def _write_fixed_scales(m, ps):
    """Working factors into the module, keeping the ADMM export scales."""
    m.U_latent.data.copy_(ps.U.float().to(torch.bfloat16))
    m.V_latent.data.copy_(ps.Vm.transpose(0, 1).contiguous().float().to(torch.bfloat16))
    m.scale_pre.data.copy_(ps.base.sp0.view(1, -1).to(torch.bfloat16))
    m.scale_post.data.copy_(ps.base.so0.view(1, -1).to(torch.bfloat16))


def _accept_ok(r, o, E0, args):
    """Accept a move only if its predicted gain clears both a relative floor and
    `noise_k` times the plane's own measured arithmetic noise."""
    if not r["improves"]:
        return False
    gain = -r["delta_f"]
    if gain <= args.noise_k * r.get("noise", 0.0):
        return False
    return (gain / o.Y_sq) / E0 > args.tol


class SignBook:
    """Cumulative sign bookkeeping against the original ADMM binary factors."""

    def __init__(self, ps):
        self.o_u = C.pos_sign(ps.base.U0).clone()
        self.o_v = C.pos_sign(ps.base.V0.transpose(0, 1)).clone()
        self.p_u, self.p_v = self.o_u.clone(), self.o_v.clone()
        self.n_u, self.n_v = self.o_u.numel(), self.o_v.numel()
        self.events = 0
        self.reversals = 0

    @torch.no_grad()
    def update(self, ps):
        cu, cv = C.pos_sign(ps.U), C.pos_sign(ps.Vm)
        self.events += int((cu != self.p_u).sum() + (cv != self.p_v).sum())
        self.reversals += int((((self.p_u != self.o_u) & (cu == self.o_u)).sum()
                               + ((self.p_v != self.o_v) & (cv == self.o_v)).sum()))
        self.p_u, self.p_v = cu, cv
        net_u = int((cu != self.o_u).sum())
        net_v = int((cv != self.o_v).sum())
        return {"net_changed_U": net_u, "net_changed_V": net_v,
                "sign_delta_U": net_u / self.n_u, "sign_delta_V": net_v / self.n_v,
                "flip_events": self.events, "reversals": self.reversals}


def descent(args):
    """Coordinate descent over Givens planes on the true binary objective.

    `--mode cyclic` sweeps planes in order, accepting each improvement as it is
    found.  `--mode greedy` is best-improvement: it keeps a priority queue over
    planes and always applies the single best available move, re-scoring lazily
    (a popped plane is re-swept, and re-queued if its value moved), which is what
    makes best-improvement affordable at all.
    """
    import heapq
    o = Q.Oracle(torch.load(f"{H.GCACHE}/quad_oracle.pt", weights_only=False))
    st = H.load_post_admm(seed=args.seed)
    sub = H.nq_sub(st["blk"])
    m = sub[LAYER]
    base = C.LayerBase(LAYER, m).to("cuda")
    ps = PlaneSearcher(o, base)
    book = SignBook(ps)
    cal_in, cal_out = st["cal_in"].to("cuda"), st["cal_out"].to("cuda")
    C.GUARD.assert_not_heldout(cal_in, cal_out, where="givens.descent")

    def real_calib():
        _write_fixed_scales(m, ps)
        return H.func_loss_over(st["blk"], cal_in, cal_out, st["kwargs"])

    E0 = o.error(ps.deployed_W())
    real0 = real_calib()
    blocks = [int(b) for b in args.blocks.split(",")]
    planes = [(i, j) for b in blocks
              for i in range(b * 32, b * 32 + 32) for j in range(i + 1, b * 32 + 32)]
    rec = {"mode": "descent", "strategy": args.mode, "layer": LAYER,
           "rank_blocks": blocks, "max_passes": args.passes, "tol_rel": args.tol,
           "n_planes": len(planes), "E0_oracle": E0, "E0_real_calib": real0,
           "oracle_offset_at_E0": E0 - real0,
           "accepts": [], "sweeps": [], "real_checks": []}
    print(f"[descent/{args.mode}] {len(planes)} planes, blocks {blocks}; "
          f"E0 oracle {E0:.8f}  real {real0:.8f}", flush=True)

    t0 = [time.time()]
    state = {"accepted": 0, "visited": 0, "last_real_orc": E0, "last_real": real0}

    def do_accept(r):
        ps.apply(r["i"], r["j"], r["theta"])
        state["accepted"] += 1
        sb = book.update(ps)
        E_now = o.error(ps.deployed_W())
        ent = {"k": state["accepted"], "plane": [r["i"], r["j"]], "theta": r["theta"],
               "n_flip": r["n_flip"], "dE_pred": r["delta_f"] / o.Y_sq,
               "E_oracle": E_now, "planes_visited": state["visited"], **sb}
        rec["accepts"].append(ent)
        if args.real_every and state["accepted"] % args.real_every == 0:
            rl = real_calib()
            dO = E_now - state["last_real_orc"]
            dR = rl - state["last_real"]
            chk = {"k": state["accepted"], "E_oracle": E_now, "E_real": rl,
                   "cum_dE_oracle": E_now - E0, "cum_dE_real": rl - real0,
                   "seg_dE_oracle": dO, "seg_dE_real": dR,
                   "eps_delta_seg": abs(dO - dR),
                   "eps_delta_cum": abs((E_now - E0) - (rl - real0)),
                   "offset": E_now - rl}
            rec["real_checks"].append(chk)
            state["last_real_orc"], state["last_real"] = E_now, rl
            print(f"[descent] accept {state['accepted']:5d}  oracle {E_now:.8f} "
                  f"({100*(E_now-E0)/E0:+.4f}%)  real {rl:.8f} "
                  f"({100*(rl-real0)/real0:+.4f}%)  eps_D(cum) {chk['eps_delta_cum']:.2e}  "
                  f"signdU {sb['sign_delta_U']*100:.3f}%  rev {sb['reversals']}  "
                  f"({time.time()-t0[0]:.0f}s)", flush=True)
        return ent

    if args.mode == "cyclic":
        for p in range(args.passes):
            acc, E_before = 0, o.error(ps.deployed_W())
            for (i, j) in planes:
                r = ps.sweep(i, j); state["visited"] += 1
                if _accept_ok(r, o, E0, args):
                    do_accept(r); acc += 1
            E_after = o.error(ps.deployed_W())
            rl = real_calib()
            rec["sweeps"].append({"sweep": p + 1, "accepted": acc,
                                  "E_oracle": E_after, "E_real": rl,
                                  "sweep_gain_pct": 100 * (E_after - E_before) / E_before,
                                  "cum_gain_oracle_pct": 100 * (E_after - E0) / E0,
                                  "cum_gain_real_pct": 100 * (rl - real0) / real0,
                                  "orth_err": ps.orthogonality_error(),
                                  "factor_consistency": ps.factor_consistency(),
                                  "seconds": time.time() - t0[0]})
            print(f"[descent] SWEEP {p+1}: accepted {acc}  oracle {E_after:.8f} "
                  f"({100*(E_after-E0)/E0:+.4f}%)  real {rl:.8f} "
                  f"({100*(rl-real0)/real0:+.4f}%)", flush=True)
            S.atomic_json(rec, f"{H.GCACHE}/givens_descent_{args.mode}{args.tag}.json")
            S.atomic_save({"R": ps.R.cpu(), "blocks": blocks, "strategy": args.mode,
                           "sweeps_done": p + 1}, f"{H.GCACHE}/givens_R_{args.mode}{args.tag}.pt")
            if acc == 0 or abs(E_after - E_before) / E_before < args.tol:
                print("[descent] converged", flush=True); break
    else:
        heap = []
        for (i, j) in planes:
            r = ps.sweep(i, j); state["visited"] += 1
            if _accept_ok(r, o, E0, args):
                heapq.heappush(heap, (r["delta_f"], i, j, 0))
        version, resweeps = 0, 0
        while heap and state["accepted"] < args.max_moves:
            df, i, j, v = heapq.heappop(heap)
            r = ps.sweep(i, j); state["visited"] += 1
            if not _accept_ok(r, o, E0, args):
                continue
            if v < version and r["delta_f"] > df * 0.9:      # stale: re-queue
                heapq.heappush(heap, (r["delta_f"], i, j, version)); resweeps += 1
                if resweeps > 20 * len(planes):
                    break
                continue
            do_accept(r); version += 1
            heapq.heappush(heap, (r["delta_f"], i, j, version))
        E_after = o.error(ps.deployed_W()); rl = real_calib()
        rec["sweeps"].append({"sweep": 1, "accepted": state["accepted"],
                              "E_oracle": E_after, "E_real": rl,
                              "cum_gain_oracle_pct": 100 * (E_after - E0) / E0,
                              "cum_gain_real_pct": 100 * (rl - real0) / real0,
                              "orth_err": ps.orthogonality_error(),
                              "factor_consistency": ps.factor_consistency(),
                              "resweeps": resweeps, "seconds": time.time() - t0[0]})
        S.atomic_save({"R": ps.R.cpu(), "blocks": blocks, "strategy": args.mode,
                       "sweeps_done": 1}, f"{H.GCACHE}/givens_R_{args.mode}{args.tag}.pt")

    # the same materialise / export path every other arm uses
    d = H.materialize_gauge(m, base, [ps.R.float()])
    rec["materialize"] = d
    rec["E_gauge_real_calib_qnq"] = H.func_loss_over(st["blk"], cal_in, cal_out, st["kwargs"])
    rec["E_gauge_heldout_qnq"] = I.block_err(st["blk"], st["ho_in"], st["ho_out"], st["kwargs"])
    H.apply_gauge_signs_only(m, base, [ps.R.float()])
    rec["E_gauge_heldout_pre_export"] = I.block_err(st["blk"], st["ho_in"], st["ho_out"],
                                                   st["kwargs"])
    rec["E_ADMM_heldout"] = st["admm_pre"]
    rec["planes_visited"] = state["visited"]
    rec["accepted_total"] = state["accepted"]
    rec["wall_s"] = time.time() - t0[0]
    S.atomic_json(rec, f"{H.GCACHE}/givens_descent_{args.mode}{args.tag}.json")
    print(f"[descent] FINAL held-out: E_ADMM {rec['E_ADMM_heldout']:.6f} -> "
          f"E_gauge {rec['E_gauge_heldout_qnq']:.6f} (Q_NQ) / "
          f"{rec['E_gauge_heldout_pre_export']:.6f} (fixed scales)", flush=True)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pilot", "descent"])
    ap.add_argument("--blocks", default="0")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--mode_desc", dest="mode", default="cyclic",
                    choices=["cyclic", "greedy"])
    ap.add_argument("--real_every", type=int, default=25)
    ap.add_argument("--max_moves", type=int, default=100000)
    ap.add_argument("--noise_k", type=float, default=10.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_planes", type=int, default=0)
    a = ap.parse_args()
    (pilot if a.mode == "pilot" else descent)(a)
