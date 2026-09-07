"""Post-ADMM state description, Snapshot A (conversion) and Snapshot B (Step 3).

Snapshot A and Snapshot B are two *different* stochastic states and are stored
in two different files (brief section 1.3).

Snapshot A -- conversion / SVID state.
    The reproduced NanoQuant conversion path contains **no SVID call at all**:
    `admm_nq.svid` / `rank1_approx` are used only inside the ADMM outer loop, and
    the post-ADMM -> deployed conversion (`NanoQuantLinear._setup_path`,
    `binary_ste`, the mean-magnitude scale extraction) draws no random numbers.
    Q_NQ is therefore already a deterministic function of (U_R, V_R).  A dedicated
    `torch.Generator` is still created, seeded and its state banked so the claim
    is auditable and so a future path that does need SVID vectors has a home;
    `assert_qnq_deterministic` proves determinism by bit-comparison.

Snapshot B -- common Step-3 state.
    NanoQuant's Step 3 (`compress_block.tune_fact`, and the `mode="ste"` branch of
    `nqx/testD.py::step3` that reproduces it) calls `set_seed(quant_config['seed'])`
    on entry and then draws one `torch.randperm(num_calib_samples, device="cpu")`
    per epoch.  The batch order is therefore a pure function of the seed and is
    *not* inherited from whatever ran before.  Snapshot B banks the CPU and CUDA
    RNG states immediately after that `set_seed`, together with the resulting
    8 x 128 permutation array and a batch cursor, and every arm asserts its own
    Step 3 replays exactly that array.
"""
import hashlib
import json
import os

import torch

from nanoquant.utils.utils import set_seed


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def atomic_text(text, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
def describe_state(blk, names, ranks):
    """Per-layer record of exactly what the reproduced post-ADMM state holds."""
    from nanoquant.modules.linear import NanoQuantLinear
    sub = {n: m for n, m in blk.named_modules() if isinstance(m, NanoQuantLinear)}
    rec = {}
    for n in names:
        m = sub[n]
        e = {
            "layer": n,
            "rank": int(ranks[f"0.{n}"]) if f"0.{n}" in ranks else int(m.rank),
            "dtype": str(m.U_latent.dtype),
            "U_latent_shape": list(m.U_latent.shape),
            "V_latent_shape": list(m.V_latent.shape),
            "scale_pre_shape": list(m.scale_pre.shape),
            "scale_post_shape": list(m.scale_post.shape),
            "has_scale_mid": hasattr(m, "scale_mid"),
            "has_bias": m.bias is not None,
            "do_train": bool(m.do_train),
            "binarized": bool(m._binarized),
        }
        if e["has_scale_mid"]:
            s3 = m.scale_mid.data.float()
            e["scale_mid"] = {
                "shape": list(s3.shape),
                "min": s3.min().item(), "max": s3.max().item(),
                "n_negative": int((s3 < 0).sum().item()),
                "n_zero": int((s3 == 0).sum().item()),
            }
        U = m.U_latent.data.float()
        V = m.V_latent.data.float()
        e["sign_frac_pos_U"] = float((U >= 0).float().mean().item())
        e["sign_frac_pos_V"] = float((V >= 0).float().mean().item())
        e["U_absmean"] = float(U.abs().mean().item())
        e["V_absmean"] = float(V.abs().mean().item())
        rec[n] = e
    return rec


# ---------------------------------------------------------------------------
SNAP_A_SEED = 0xA5010000
STEP3_SEED = 0


def make_snapshot_A(path, layers):
    """Bank a dedicated conversion generator (never the global RNG)."""
    g_cpu = torch.Generator(device="cpu"); g_cpu.manual_seed(SNAP_A_SEED)
    g_cuda = torch.Generator(device="cuda"); g_cuda.manual_seed(SNAP_A_SEED)
    snap = {
        "kind": "snapshot_A_conversion",
        "seed": SNAP_A_SEED,
        "cpu_generator_state": g_cpu.get_state(),
        "cuda_generator_state": g_cuda.get_state(),
        "layers": list(layers),
        "svid_vectors": {},          # empty: the export path draws no random numbers
        "note": ("The reproduced NanoQuant conversion path performs no SVID / "
                 "power_iteration and draws no random numbers; Q_NQ is a pure "
                 "function of (U_R, V_R).  Determinism is asserted by "
                 "gauge.core.assert_qnq_deterministic, not by RNG replay."),
    }
    atomic_save(snap, path)
    return snap


def make_snapshot_B(path, n_samples=128, epochs=8, seed=STEP3_SEED):
    """Bank the RNG state and the exact calibration batch order Step 3 will use."""
    set_seed(seed)
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state()
    perms = torch.stack([torch.randperm(n_samples, device="cpu", dtype=torch.long)
                         for _ in range(epochs)])
    snap = {
        "kind": "snapshot_B_common_step3",
        "seed": seed,
        "n_samples": n_samples,
        "epochs": epochs,
        "cpu_rng_state": cpu_state,
        "cuda_rng_state": cuda_state,
        "batch_order": perms,          # [epochs, n_samples]
        "batch_cursor": 0,
        "note": ("Step 3 calls set_seed(seed) on entry, so the batch order is a "
                 "pure function of the seed.  Restoring this snapshot before "
                 "every arm makes that explicit and lets each arm assert it "
                 "replayed the identical order."),
    }
    atomic_save(snap, path)
    return snap


def restore_snapshot_B(snap):
    """Restore the exact RNG state Step 3 starts from."""
    set_seed(snap["seed"])
    torch.set_rng_state(snap["cpu_rng_state"])
    torch.cuda.set_rng_state(snap["cuda_rng_state"])


class PermRecorder:
    """Records every torch.randperm(n_samples, device='cpu') drawn while active,
    so an arm can prove its Step 3 saw Snapshot B's batch order."""

    def __init__(self, n_samples):
        self.n = n_samples
        self.perms = []
        self._orig = None

    def __enter__(self):
        self._orig = torch.randperm
        rec = self

        def patched(n, *a, **kw):
            out = rec._orig(n, *a, **kw)
            if n == rec.n and out.device.type == "cpu":
                rec.perms.append(out.clone())
            return out

        torch.randperm = patched
        return self

    def __exit__(self, *exc):
        torch.randperm = self._orig
        return False

    def as_tensor(self):
        return torch.stack(self.perms) if self.perms else torch.empty(0)

    def assert_matches(self, snap, where=""):
        got = self.as_tensor()
        want = snap["batch_order"]
        assert got.shape == want.shape, f"batch order shape {tuple(got.shape)} != {tuple(want.shape)} at {where}"
        assert torch.equal(got, want), f"batch order differs from Snapshot B at {where}"
