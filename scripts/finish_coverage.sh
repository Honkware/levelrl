#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
while tmux list-sessions -F "#{session_name}" 2>/dev/null | grep -Fxq coverage-data || \
      tmux list-sessions -F "#{session_name}" 2>/dev/null | grep -Fxq coverage-repair; do
    sleep 30
done
.venv/bin/python -m levelrl.coverage finalize
.venv/bin/python -m levelrl.coverage audit
