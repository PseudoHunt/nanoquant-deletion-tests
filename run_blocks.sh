#!/bin/bash
cd /home/work/exp; source env.sh
until grep -q ALLDONE /home/work/logs/arm4.log 2>/dev/null; do sleep 30; done
python3 -m nqx.gauge.givens descent --blocks 0,1,2,3 --passes 1 --mode_desc cyclic --tag _b0123 > /home/work/logs/desc_b0123.log 2>&1
python3 -m nqx.gauge.arms 4b0123 > /home/work/logs/arm4b.log 2>&1
echo ALLDONE >> /home/work/logs/arm4b.log
cd /home/work/exp; source env.sh
python3 -c "
import sys; sys.path.insert(0,'/home/work/exp/nqx')
from nanoquant.modules.hub import NanoQuantConfigDataclass
import torch, gauge.quad as Q, gauge.harness as H, gauge.core as C
o = Q.Oracle(torch.load(f'{H.GCACHE}/quad_oracle.pt', weights_only=False))
st = H.load_post_admm(seed=0); sub = H.nq_sub(st['blk'])
base = C.LayerBase('mlp.down_proj', sub['mlp.down_proj']).to('cuda')
Q.verify(o, st, base)
" >> /home/work/logs/arm4b.log 2>&1
echo VERIFYDONE >> /home/work/logs/arm4b.log
