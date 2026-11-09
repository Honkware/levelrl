#!/usr/bin/env bash
# Starts the unattended pipeline in tmux session "pipeline". Safe to rerun: finished stages are skipped and a
# training arm resumes from its last checkpoint. Retries up to three times, two minutes apart.
cd "$(dirname "$0")/.."
tmux has-session -t pipeline 2>/dev/null && { echo "already running: tmux attach -t pipeline"; exit 0; }
tmux new -d -s pipeline 'set -o pipefail; for i in 1 2 3; do .venv/bin/python -u -m levelrl.pipeline 2>&1 | grep --line-buffered -v Warn | tee -a runs/pipeline.log && break; sleep 120; done'
echo "started: tmux attach -t pipeline, or tail -f runs/pipeline.log"
