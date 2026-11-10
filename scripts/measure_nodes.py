import json
import os
import statistics
import sys
from multiprocessing import Pool

import chess
import chess.engine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from levelrl import ladder

# How many nodes a rung searches in the first run's 50 ms, with every worker busy, which is what a node budget has
# to match to keep the rungs about as strong as they were.


def job(args):
    name, moves = args
    b = chess.Board()
    for m in moves:
        b.push_uci(m)
    info = ladder.rung(name).eng.analyse(b, chess.engine.Limit(time=0.05))
    return name, info.get("nodes", 0)


def main():
    games = json.loads(open(sys.argv[1]).read())
    jobs = [(name, g["moves"][:cut]) for name in ("sk0", "elo1500", "elo2100")
            for g in games[:120] for cut in range(12, len(g["moves"]) - 1, 16)]
    with Pool(22) as p:
        res = p.map(job, jobs, chunksize=8)
    for name in ("sk0", "elo1500", "elo2100"):
        v = sorted(n for k, n in res if k == name)
        print(name, "n", len(v), "median", statistics.median(v), "p10", v[len(v) // 10], "p90", v[9 * len(v) // 10])


if __name__ == "__main__":
    main()
