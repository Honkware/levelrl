#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw runs

for month in "$@"; do
    file="data/raw/lichess_db_standard_rated_${month}.pgn.zst"
    if [ -s "$file" ]; then
        echo "$month already downloaded"
        continue
    fi
    url="https://database.lichess.org/standard/lichess_db_standard_rated_${month}.pgn.zst"
    echo "downloading $month from $url"
    wget -c -nv --tries=0 --waitretry=30 --timeout=60 -O "$file.part" "$url"
    mv "$file.part" "$file"
    echo "finished $month"
done
