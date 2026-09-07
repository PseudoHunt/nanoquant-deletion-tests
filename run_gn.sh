#!/bin/bash
cd /home/work/exp; source env.sh
L=/home/work/logs
python3 -m nqx.gauge.gn_probe --layer self_attn.o_proj > $L/gn_o.log 2>&1
python3 -m nqx.gauge.gn_probe --layer self_attn.k_proj > $L/gn_k.log 2>&1
python3 -m nqx.gauge.gn_probe --layer self_attn.k_proj --doses 0 \
  --R_from /home/work/exp/artifacts/gauge/wholeblock_R_blkgate.pt > $L/gn_kben.log 2>&1
echo GN_DONE >> $L/gn_kben.log
