"""Create Snapshot A (conversion) and Snapshot B (common Step 3), plus the
checksum manifest for the post-ADMM state (brief sections 1.2-1.4)."""
import json
import os
import shutil
import time

import torch

from nanoquant.modules.hub import NanoQuantConfigDataclass  # noqa: F401  import-order guard

import instrument as I
import testD as D

from . import core as C
from . import harness as H
from . import state as S

MIRROR = "/home/jl_fs/gauge_mirror"


def main():
    os.makedirs(H.GCACHE, exist_ok=True)
    st = H.load_post_admm(seed=0)
    blk, ranks = st["blk"], st["ranks"]
    sub = H.nq_sub(blk)

    a = S.make_snapshot_A(f"{H.GCACHE}/snapshot_A.pt", H.NAMES)
    b = S.make_snapshot_B(f"{H.GCACHE}/snapshot_B.pt",
                          n_samples=st["qd"]["num_calib_samples"],
                          epochs=st["qd"]["fact_epochs"], seed=st["qd"]["seed"])
    print(f"[snap] A seed={a['seed']}  svid_vectors={len(a['svid_vectors'])} (export path draws no RNG)")
    print(f"[snap] B seed={b['seed']}  batch_order={tuple(b['batch_order'].shape)}")

    desc = S.describe_state(blk, H.NAMES, ranks)
    meta = {
        "post_admm_pre_heldout": I.block_err(blk, st["ho_in"], st["ho_out"], st["kwargs"]),
        "cached_reference_pre": 0.14366140267175126,
        "cached_reference_post_joint_step3": 0.11485157565362869,
        "ranks": {n: int(ranks[f"0.{n}"]) for n in H.NAMES},
        "layers": desc,
        "gauge_batch_order_seed": H.GAUGE_DATA_SEED,
        "gauge_batch_order_200": H.gauge_batch_order(200),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    S.atomic_json(meta, f"{H.GCACHE}/post_admm_meta.json")

    files = [f"{D.DCACHE}/block0_postadmm.pt", f"{H.GCACHE}/snapshot_A.pt",
             f"{H.GCACHE}/snapshot_B.pt", f"{H.GCACHE}/post_admm_meta.json",
             f"{H.CACHE}/in_b0.pt", f"{H.CACHE}/out_b0.pt", f"{H.CACHE}/stats.pt",
             f"{H.CACHE}/kwargs.pt", f"{H.CACHE}/sequences.pt"]
    man = {}
    os.makedirs(MIRROR, exist_ok=True)
    for f in files:
        man[os.path.basename(f)] = {"sha256": S.sha256(f), "bytes": os.path.getsize(f), "path": f}
        print(f"[snap] {os.path.basename(f)}  {man[os.path.basename(f)]['sha256'][:16]}  "
              f"{man[os.path.basename(f)]['bytes']/1e6:.1f} MB", flush=True)
    # persistent mirror: everything except the two 1.3 GB activation caches
    for f in files:
        if os.path.getsize(f) < 2e9 and "in_b0" not in f and "out_b0" not in f:
            shutil.copy2(f, os.path.join(MIRROR, os.path.basename(f)))
    S.atomic_json({"files": man, "mirror": MIRROR},
                  "/home/work/exp/artifacts/gauge_manifest.json")
    shutil.copy2("/home/work/exp/artifacts/gauge_manifest.json", MIRROR)
    print("[snap] manifest written")


if __name__ == "__main__":
    main()
