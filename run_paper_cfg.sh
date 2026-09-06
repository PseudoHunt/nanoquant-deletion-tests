#!/bin/bash
# Step 0 re-run with the Appendix C configuration:
#   gamma (calib_shrinkage) = 0.2   [repo default 0.4]
#   model_kd_lr             = 1e-6  [repo default 1e-5]
#   transformers            = 4.51.3 (the pinned/paper version, via the dtype shim)
set -u
SEED=${1:-0}
TAG=${2:-repro_paper_seed0}
export HF_HOME=/home/work/hf_cache PYTHONUNBUFFERED=1
MID=/home/work/hf_cache/hub/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da
cd /home/work/NanoQuant
/home/work/venv51/bin/python -u /home/work/exp/nqx/run_tf451.py \
  --model_id "$MID" --bits 1.0 --seed "$SEED" \
  --num_calib_samples 128 --calib_dataset wikitext2 \
  --calib_shrinkage 0.2 --model_kd_lr 1e-6 \
  --nonfact_epochs 8 --fact_epochs 8 --admm_outer_iters 400 \
  --ppl_task "wikitext2,c4" --zeroshot_task "" 2>&1 \
  | /home/work/venv/bin/python -u /home/work/exp/ts.py > /home/work/exp/logs/${TAG}.log
echo "EXIT=${PIPESTATUS[0]}" >> /home/work/exp/logs/${TAG}.log
