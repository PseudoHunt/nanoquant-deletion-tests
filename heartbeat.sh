#!/bin/bash
# Gauge-NQ result heartbeat: commit+push results/ every 120s under a shared lock.
cd /home/work/exp || exit 1
while true; do
  flock /tmp/gauge_git.lock sh -c '
    cd /home/work/exp
    git add results/ artifacts/gauge_manifest.json 2>/dev/null
    git diff --cached --quiet || git commit -qm "hb $(date +%s)"
    git push -q origin HEAD
  ' >> /home/work/logs/heartbeat.log 2>&1
  sleep 120
done
