from __future__ import annotations

import gzip
import json
from collections import Counter

import chess
import chess.engine

from levelrl import data

# The fixed opening suite every gauntlet and group starts from: the most played 8-ply openings in the
# post-cutoff blitz games we scanned, kept only if Stockfish calls them level (|eval| <= MAX_CP at DEPTH) and the
# side to move is not in check, so no game starts already decided or forced.
PLIES = 8
N = 240
MAX_CP = 60
DEPTH = 14
OUT = data.ROOT / "openings.json"


def build(n: int = N) -> list[list[str]]:
    seen = Counter()
    for m in data.MONTHS:
        for line in gzip.open(data.ROOT / "scan" / m / "rows.jsonl.gz", "rt"):
            r = json.loads(line)
            if data.blitz(r["tc"]) and len(r["moves"]) >= PLIES:
                seen[tuple(r["moves"][:PLIES])] += 1
    out = []
    with chess.engine.SimpleEngine.popen_uci("stockfish") as sf:
        sf.configure({"Threads": 4, "Hash": 256})
        for moves, count in seen.most_common():
            b = chess.Board()
            for mv in moves:
                b.push_uci(mv)
            if b.is_check():
                continue
            cp = sf.analyse(b, chess.engine.Limit(depth=DEPTH))["score"].white().score(mate_score=10_000)
            if abs(cp) <= MAX_CP:
                out.append({"moves": list(moves), "games": count, "cp": cp})
            if len(out) == n:
                break
    OUT.write_text(json.dumps({"plies": PLIES, "max_cp": MAX_CP, "depth": DEPTH, "months": data.MONTHS,
                               "openings": out}, indent=1))
    return [o["moves"] for o in out]


def load() -> list[list[str]]:
    return [o["moves"] for o in json.loads(OUT.read_text())["openings"]]


if __name__ == "__main__":
    s = build()
    print(len(s), "openings")
