#!/bin/bash
# Stop the whole-block run once sweep 1 reaches mlp.up_proj.  Results are written
# after every layer (atomic JSON + rotation snapshot), so everything through
# up_proj is preserved; only the 154 minutes of down_proj are skipped.
L=/home/work/logs
until grep -q 'sw1 mlp.up_proj' $L/wholeblock.log 2>/dev/null; do
  sleep 15
  pgrep -f "nqx.gauge.wholeblock" >/dev/null || exit 0
done
sleep 3
for p in $(pgrep -f "nqx.gauge.wholeblock"); do kill -TERM $p 2>/dev/null; done
sleep 5
for p in $(pgrep -f "nqx.gauge.wholeblock"); do kill -9 $p 2>/dev/null; done
echo "STOPPED_AFTER_UP_PROJ" >> $L/wholeblock.log
