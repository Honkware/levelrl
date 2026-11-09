from __future__ import annotations

import json
import random
from multiprocessing import Pool

import chess
import torch

from levelrl import ladder, maia, progress

# Maia against the Stockfish rungs, many games at once. Maia's moves are one batched forward over every game where
# it is to move, while the rungs' moves for the other games run in a process pool alongside it.


def _sf(args):
    name, moves, seed = args
    b = chess.Board()
    for m in moves:
        b.push_uci(m)
    return ladder.rung(name).move(b, random.Random(seed * 1000 + b.ply()), game=seed).uci()


def sample(logits: torch.Tensor, temp: float, top_p: float, gen: torch.Generator) -> torch.Tensor:
    # Batched maia3.uci.sample_from_logits: nucleus after temperature, top move always kept.
    if temp <= 0:
        return logits.argmax(-1)
    p = torch.softmax(logits / temp, -1)
    if top_p < 1.0:
        sp, si = p.sort(-1, descending=True)
        keep = sp.cumsum(-1) <= top_p
        keep[:, 0] = True
        p = torch.zeros_like(p).scatter(1, si, sp * keep)
    return torch.multinomial(p, 1, generator=gen)[:, 0]


def play(cfg, model, policy: dict, opponents: list[str], openings: list[list[str]], seed: int = 0,
         procs: int = 10, bsz: int = 256, name: str = "gauntlet", elo: dict[str, float] | None = None,
         log=lambda s: print(s, flush=True)) -> list[dict]:
    # policy: self_elo, oppo_elo, history, temp, top_p. One game per (opponent, opening, colour).
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    todo = [(o, op, white, rng.getrandbits(31)) for o in opponents for op in openings for white in (True, False)]
    live, done, total = [], [], len(todo)
    me = f"maia@{policy['self_elo']}"
    est = None
    with Pool(procs) as pool:
        while todo or live:
            while todo and len(live) < bsz:
                o, op, white, s = todo.pop()
                b = chess.Board()
                for m in op:
                    b.push_uci(m)
                live.append({"opp": o, "white": white, "seed": s, "board": b, "opening": op})
            mine = [g for g in live if g["board"].turn == g["white"]]
            theirs = [g for g in live if g["board"].turn != g["white"]]
            fut = pool.map_async(_sf, [(g["opp"], [m.uci() for m in g["board"].move_stack], g["seed"]) for g in theirs])
            if mine:
                logits, _ = maia.forward(cfg, model, [g["board"] for g in mine], [policy["self_elo"]] * len(mine),
                                         [policy["oppo_elo"]] * len(mine), history=policy["history"])
                for g, j in zip(mine, sample(logits, policy["temp"], policy["top_p"], gen).tolist()):
                    g["board"].push(maia.to_move(g["board"], j))
            for g, m in zip(theirs, fut.get()):
                g["board"].push_uci(m)
            keep = []
            for g in live:
                b = g["board"]
                names = (me, g["opp"]) if g["white"] else (g["opp"], me)
                if b.is_game_over(claim_draw=True) or b.ply() >= ladder.MAX_PLIES:
                    s, end = ladder.ending(b)
                    r = s if g["white"] else 1 - s
                    done.append({"opp": g["opp"], "white": g["white"], "score": r, "plies": b.ply(), "end": end,
                                 "moves": [m.uci() for m in b.move_stack]})
                    progress.board(name, g["seed"], b, *names, result=f"{b.result(claim_draw=True)} · {end}")
                    progress.event(name, f"{me} {'beat' if r == 1 else 'lost to' if r == 0 else 'drew'} {g['opp']}"
                                         f" · {b.ply()} plies · {end}")
                    if elo and len(done) % 20 == 0:
                        est = rate(done, elo)
                    stats = {"score": f"{sum(x['score'] for x in done) / len(done):.0%}",
                             "draws": f"{sum(x['score'] == 0.5 for x in done) / len(done):.0%}",
                             "longest": f"{max(x['plies'] for x in done)} plies"}
                    if est is not None:
                        stats = {"elo so far": f"{est:.0f}", **stats}
                    progress.tick(name, len(done), total, "games", stats=stats,
                                  note=f"{me} vs {', '.join(opponents)} · {len(live)} boards at once")
                    if len(done) % 100 == 0:
                        log(f"{len(done)} games")
                else:
                    keep.append(g)
            if keep:
                g = keep[0]
                progress.board(name, g["seed"], g["board"], *((me, g["opp"]) if g["white"] else (g["opp"], me)))
            live = keep
    return done


def rate(games: list[dict], elo: dict[str, float]) -> float:
    # Maximum-likelihood Elo of one player against rungs held at their calibrated ratings, draws as half points.
    # The score is monotone in the rating, so bisection finds it. Clamped so a clean sweep stays finite.
    def excess(r):
        return sum(g["score"] - 1 / (1 + 10 ** ((elo[g["opp"]] - r) / 400)) for g in games)
    lo, hi = -1000.0, 4500.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if excess(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def rate_ci(games: list[dict], elo: dict[str, float], n: int = 1000, seed: int = 0) -> tuple[float, list[float]]:
    rng = random.Random(seed)
    v = sorted(rate(rng.choices(games, k=len(games)), elo) for _ in range(n))
    return rate(games, elo), [v[int(0.025 * n)], v[int(0.975 * n)]]


def save(path, policy, games, elo):
    r, ci = rate_ci(games, elo)
    out = {"policy": policy, "elo": r, "ci": ci, "games": len(games),
           "score": sum(g["score"] for g in games) / len(games),
           "by_opp": {o: {"n": sum(g["opp"] == o for g in games),
                          "score": sum(g["score"] for g in games if g["opp"] == o)
                          / max(1, sum(g["opp"] == o for g in games))} for o in sorted({g["opp"] for g in games})}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    path.with_suffix(".games.json").write_text(json.dumps(games))
    return out


def closest(elo: dict[str, float], target: float, k: int = 3, refused=()) -> list[str]:
    ok = [n for n in elo if n not in refused]
    return sorted(ok, key=lambda n: abs(elo[n] - target))[:k]

