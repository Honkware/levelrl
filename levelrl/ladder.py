from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import shutil
from multiprocessing import Pool
from pathlib import Path

import chess
import chess.engine

from levelrl import openings, progress

# Stockfish 19 rungs. Its strength limit stops at 1320, so the rungs below that are Skill 0 playing a uniformly
# random legal move with probability p, as the spec asks. One thread, a small fixed hash and a fixed move time,
# all written into elo_calibration.json so the ladder can be rebuilt exactly.
ENGINE = {"Threads": 1, "Hash": 16}
# A rung searches a fixed number of nodes per move, so it plays the same at any CPU load and on any machine. The
# first run gave rungs 50 ms instead, which made their strength depend on how busy the box was. LEVELRL_MOVETIME
# (seconds) brings that back, to reproduce that run's ladder.
NODES = 30_000
MOVETIME = float(os.environ["LEVELRL_MOVETIME"]) if os.environ.get("LEVELRL_MOVETIME") else None
LIMIT = {"time": MOVETIME} if MOVETIME else {"nodes": NODES}
MAX_PLIES = 400
RUNGS = {
    "sk0_r50": {"skill": 0, "random": 0.5},
    "sk0_r30": {"skill": 0, "random": 0.3},
    "sk0_r15": {"skill": 0, "random": 0.15},
    "sk0_r07": {"skill": 0, "random": 0.07},
    "sk0": {"skill": 0, "random": 0.0},
    **{f"elo{e}": {"elo": e, "random": 0.0} for e in (1320, 1500, 1700, 1900, 2100, 2300, 2500)},
}
# The ladder's mean is pinned rather than one rung, so no rung's interval is inflated by its distance from the pin.
# Only differences matter in the end, since the bot anchors fix the units.
PIN = 1500.0
MAX_HALF_CI = 60.0
SPAN = 2
OUT = Path(__file__).resolve().parent.parent / "runs" / "elo_calibration.json"


class Rung:
    def __init__(self, name: str, eng: chess.engine.SimpleEngine):
        self.name, self.cfg, self.eng = name, RUNGS[name], eng

    def reset(self):
        c = self.cfg
        if "elo" in c:
            self.eng.configure({**ENGINE, "UCI_LimitStrength": True, "UCI_Elo": c["elo"]})
        else:
            self.eng.configure({**ENGINE, "UCI_LimitStrength": False, "Skill Level": c["skill"]})

    def move(self, board: chess.Board, rng: random.Random, game=None) -> chess.Move:
        if self.cfg["random"] and rng.random() < self.cfg["random"]:
            return rng.choice(list(board.legal_moves))
        # A new game id sends ucinewgame, which clears the hash between games. Each ladder game has its own rng.
        return self.eng.play(board, chess.engine.Limit(**LIMIT), game=rng if game is None else game).move

    def close(self):
        try:
            self.eng.quit()
        except Exception:
            pass
        finally:
            self.eng.close()


_engines: dict = {}


def rung(name: str) -> Rung:
    if name not in _engines:
        _engines[name] = Rung(name, chess.engine.SimpleEngine.popen_uci("stockfish"))
        _engines[name].reset()
    return _engines[name]


def close_engines():
    for r in _engines.values():
        r.close()
    _engines.clear()


def ending(b: chess.Board) -> tuple[float, str]:
    # White's score (1, 0.5 or 0) and how the game ended.
    o = b.outcome(claim_draw=True)
    if o is None:
        return 0.5, "move cap"
    return {True: 1.0, False: 0.0, None: 0.5}[o.winner], o.termination.name.lower().replace("_", " ")


def play(white, black, opening: list[str], seed: int, show: str | None = None) -> tuple[float, list[str], str]:
    # Players are anything with move(board, rng). show names the job to stream this game's board for.
    rng = random.Random(seed)
    b = chess.Board()
    for mv in opening:
        b.push_uci(mv)
    while not b.is_game_over(claim_draw=True) and b.ply() < MAX_PLIES:
        b.push((white if b.turn else black).move(b, rng))
        if show:
            progress.board(show, seed, b, getattr(white, "name", "white"), getattr(black, "name", "black"))
    s, end = ending(b)
    if show:
        progress.board(show, seed, b, getattr(white, "name", "white"), getattr(black, "name", "black"),
                       result=f"{b.result(claim_draw=True)} · {end}")
    return s, [m.uci() for m in b.move_stack], end


def engine_move(args):
    name, b, seed = args
    return rung(name).move(b, random.Random(seed * 1000 + b.ply()), game=seed).uci()


def lc0_path():
    for p in (shutil.which("lc0"), os.path.expanduser("~/lc0/build/lc0")):
        if p and os.path.exists(p):
            return p
    return None


class Lc0:
    # A Maia-1 net through lc0 at one node, the way the Lichess maia bots play.
    def __init__(self, net):
        self.net, self.name = net, Path(net).stem
        self.eng = chess.engine.SimpleEngine.popen_uci([lc0_path(), f"--weights={net}", "--threads=1"])

    def move(self, board, rng, game=None):
        return self.eng.play(board, chess.engine.Limit(nodes=1), game=game or rng).move

    def close(self):
        try:
            self.eng.quit()
        except Exception:
            pass
        finally:
            self.eng.close()


_lc0 = None


def lc0_game(args):
    # One cross-check game in a pool worker, which keeps a single lc0 process and swaps it when the net changes.
    global _lc0
    net, opp, op, seed, white = args
    if _lc0 is None or _lc0.net != net:
        if _lc0 is not None:
            _lc0.close()
        _lc0 = Lc0(net)
    o = rung(opp)
    s, _, _ = play(_lc0 if white else o, o if white else _lc0, op, seed, show="crosscheck")
    return {"opp": opp, "score": s if white else 1 - s}


def _job(args):
    w, bl, op, seed = args
    s, moves, end = play(rung(w), rung(bl), op, seed, show="ladder")
    return {"white": w, "black": bl, "score": s, "plies": len(moves), "end": end}


def fit(games: list[dict], names: list[str], pin=PIN, iters: int = 2000) -> dict[str, float]:
    # Bradley-Terry on the Elo scale with draws as half points, fitted by MM (Hunter 2004). pin is the ladder mean,
    # or (rung, elo) to pin one rung.
    # A rung that never scores or never concedes has no finite rating, so every pair gets a tiny 0.5 prior.
    w = {n: 0.0 for n in names}
    n = {}
    for g in games:
        a, b = g["white"], g["black"]
        w[a] += g["score"]
        w[b] += 1 - g["score"]
        n[a, b] = n.get((a, b), 0) + 1
        n[b, a] = n.get((b, a), 0) + 1
    for a, b in list(n):
        w[a] += 0.5 / max(1, len(names) - 1) / 2
        n[a, b] += 0.5 / max(1, len(names) - 1)
    g_ = {x: 1.0 for x in names}
    for _ in range(iters):
        new = {}
        for a in names:
            den = sum(c / (g_[a] + g_[b]) for (x, b), c in n.items() if x == a)
            new[a] = w[a] / den if den else g_[a]
        s = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        g_ = {k: v / s for k, v in new.items()}
    elo = {k: 400 * math.log10(v) for k, v in g_.items()}
    off = pin[1] - elo[pin[0]] if isinstance(pin, tuple) else pin - sum(elo.values()) / len(elo)
    return {k: v + off for k, v in elo.items()}


def bootstrap(games: list[dict], names: list[str], n: int = 200, seed: int = 0) -> dict[str, list[float]]:
    rng = random.Random(seed)
    draws = {k: [] for k in names}
    for _ in range(n):
        e = fit(rng.choices(games, k=len(games)), names, iters=300)
        for k in names:
            draws[k].append(e[k])
    out = {}
    for k, v in draws.items():
        v.sort()
        out[k] = [v[int(0.025 * n)], v[int(0.975 * n)]]
    return out


def run(per_pair: int = 200, span: int = SPAN, procs: int = 10, seed: int = 1) -> dict:
    # Only rungs within span steps of each other play, since a lopsided pair says little. Games accumulate across
    # runs in ladder_games.json, so a rerun with a new seed only adds evidence.
    names = list(RUNGS)
    ops = openings.load()
    rng = random.Random(seed)
    jobs = []
    for i, a in enumerate(names):
        for b in names[i + 1:i + 1 + span]:
            for op in rng.sample(ops, per_pair // 2):
                jobs += [(a, b, op, rng.getrandbits(32)), (b, a, op, rng.getrandbits(32))]
    path = OUT.parent / ("ladder_games.json" if MOVETIME else f"ladder_games_nodes{NODES}.json")
    games = json.loads(path.read_text()) if path.exists() else []
    games = [g for g in games if g["white"] in RUNGS and g["black"] in RUNGS]
    order = {n: i for i, n in enumerate(names)}
    new = []
    with Pool(procs) as p:
        for k, g in enumerate(p.imap_unordered(_job, jobs, chunksize=4)):
            games.append(g)
            new.append(g)
            w, b = g["white"], g["black"]
            progress.event("ladder", f"{w} {'beat' if g['score'] == 1 else 'lost to' if g['score'] == 0 else 'drew'}"
                                     f" {b} · {g['plies']} plies · {g['end']}")
            progress.tick("ladder", k + 1, len(jobs), "games", note=f"{procs} boards at once · {len(jobs) - k - 1:,} queued",
                          stats={"white scores": f"{sum(x['score'] for x in new) / len(new):.0%}",
                                 "draws": f"{sum(x['score'] == 0.5 for x in new) / len(new):.0%}",
                                 "upsets": sum((x["score"] == 1) == (order[x["white"]] < order[x["black"]])
                                               for x in new if x["score"] != 0.5),
                                 "longest": f"{max(x['plies'] for x in new)} plies"})
            if k % 200 == 0:
                print(k, "/", len(jobs), flush=True)
    OUT.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(games))
    elo = fit(games, names)
    ci = bootstrap(games, names)
    res = {
        "engine": "Stockfish 19", "options": ENGINE, "limit": LIMIT, "max_plies": MAX_PLIES, "rungs": RUNGS,
        "pin_mean": PIN, "games": len(games), "elo": elo, "ci": ci,
        "refused": [k for k, (lo, hi) in ci.items() if (hi - lo) / 2 > MAX_HALF_CI],
    }
    OUT.write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-pair", type=int, default=200)
    ap.add_argument("--span", type=int, default=SPAN)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    r = run(a.per_pair, a.span, a.procs, a.seed)
    for k in RUNGS:
        print(f"{k:9s} {r['elo'][k]:7.0f}  [{r['ci'][k][0]:.0f}, {r['ci'][k][1]:.0f}]")
    print("refused:", r["refused"])
