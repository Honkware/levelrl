#!/usr/bin/env bash
# Copies this checkout to the training box, with a manifest of the commit it came from. Refuses a dirty tree: the
# first two runs trained on code that was not committed at the time, so their results cannot be traced to a commit.
set -euo pipefail
cd "$(dirname "$0")/.."
host="${1:-${LEVELRL_HOST:?usage: scripts/deploy.sh HOST, or set LEVELRL_HOST}}"
if [ -n "$(git status --porcelain)" ]; then
    echo "uncommitted changes, commit first:"; git status --short; exit 1
fi
printf '{"commit": "%s", "subject": "%s", "deployed": "%s"}\n' \
    "$(git rev-parse HEAD)" "$(git log -1 --format=%s | sed 's/"/\\"/g')" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > MANIFEST.json
rsync -rc --exclude __pycache__ --exclude target levelrl tests scripts native README.md requirements.txt MANIFEST.json "$host":levelrl/
rm MANIFEST.json
echo "deployed $(git rev-parse --short HEAD) to $host"
