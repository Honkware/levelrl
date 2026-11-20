#!/usr/bin/env bash
set -euo pipefail
if [ -d /w ]; then
    cd /w
else
    cd "$(dirname "$0")/.."
fi
MONTHS=${MONTHS:-"2025-09 2026-03 2026-09"}
BANDS=${BANDS:-"600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2100 2200 2300 2400 2500 2600"}
PYTHON=${PYTHON:-python}
for month in $MONTHS; do
    file="data/raw/lichess_db_standard_rated_${month}.pgn.zst"
    if [ ! -s "$file" ] || [ -f "$file.aria2" ]; then
        mkdir -p data/raw
        curl -sfL "https://database.lichess.org/standard/lichess_db_standard_rated_${month}.pgn.zst.torrent" \
            -o "data/raw/lichess_db_standard_rated_${month}.pgn.zst.torrent"
        "$PYTHON" scripts/torrent_download.py "data/raw/lichess_db_standard_rated_${month}.pgn.zst.torrent" --out data/raw
    fi
    "$PYTHON" -m levelrl.data --scan "$month" > "scan_${month}.log" 2>&1
    rm -rf data/raw
done
levelrl/native/target/release/band --months $MONTHS --bands $BANDS --out data/positions > build.log 2>&1
tar czf out.tgz data/positions data/scan/*/done *.log
touch FINISHED
