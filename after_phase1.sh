#!/bin/bash
set -u
cd /home/work/exp
while pgrep -f "[p]hase1.sh" > /dev/null; do sleep 20; done
echo "[after] phase1 done $(date -u)"
/home/work/venv/bin/python decide.py > artifacts/decide_seed0.txt 2>&1
git add -A && git commit -qm "phase 1 complete: A+B arm

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YAQwZnt4tN9kRonGomC5bi" || true
git push -q origin master || true
echo "[after] === Appendix C config repro run start $(date -u) ==="
/home/work/exp/run_paper_cfg.sh 0 repro_paper_seed0
/home/work/venv/bin/python /home/work/exp/parse_step0.py \
    /home/work/exp/logs/repro_paper_seed0.log \
    /home/work/exp/artifacts/step0_paper_seed0.json > /home/work/exp/logs/parse_paper.txt 2>&1
cat /home/work/exp/logs/parse_paper.txt
git add -A && git commit -qm "Appendix C config repro run (gamma=0.2, model_kd_lr=1e-6, transformers 4.51.3)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YAQwZnt4tN9kRonGomC5bi" || true
git push -q origin master || true
echo "[after] ALL DONE $(date -u)"
