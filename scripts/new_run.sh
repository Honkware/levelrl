#!/usr/bin/env bash
# On the training box: moves the current run aside as runs/runN and results/runN, so the next pipeline starts
# clean. Nothing is deleted.
set -euo pipefail
cd "$(dirname "$0")/.."
tmux has-session -t pipeline 2>/dev/null && { echo "the pipeline is running, not archiving"; exit 1; }
[ -d runs/pipeline ] || { echo "no current run"; exit 0; }
n=1; while [ -e "runs/run$n" ] || [ -e "results/run$n" ]; do n=$((n + 1)); done
mkdir -p "runs/run$n/progress" "results/run$n"
for f in pipeline arms pipeline.log; do [ -e "runs/$f" ] && mv "runs/$f" "runs/run$n/"; done
cp runs/elo_calibration.json "runs/run$n/" 2>/dev/null || true
mv runs/progress/*.json "runs/run$n/progress/" 2>/dev/null || true
rm -f runs/progress/.run.*
for f in results/*; do
    case "$(basename "$f")" in run[0-9]*|parity.json|positions_manifest.json|elo_calibration.json) ;; *) mv "$f" "results/run$n/";; esac
done
echo "archived as run$n"
