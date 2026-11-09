from __future__ import annotations

import random
from multiprocessing import Pool

import chess

from levelrl import ladder, maia, progress

# Many Maia-vs-rung games at once. Each step every live game makes one move: the Maia side through mover(), one
# batched call over all games where Maia is to move, and the rung side in a process pool running alongside it.
# Spec fields: opp (rung name), white (Maia plays white), opening (uci list), seed, plus anything the caller wants
# carried through to the finished record.

_pool = None


def pool(procs: int = 22) -> Pool:
    global _pool
    if _pool is None:
        _pool = Pool(procs)
    return _pool


def _sf(args):
    name, moves, seed = args
    b = chess.Board()
    for m in moves:
        b.push_uci(m)
    return ladder.rung(name).move(b, random.Random(seed * 1000 + b.ply()), game=seed).uci()


class Arena:
    def __init__(self, cfg, mover, feed, name: str, bsz: int = 256, me=lambda s: "maia"):
        self.cfg, self.mover, self.feed, self.name, self.bsz, self.me = cfg, mover, feed, name, bsz, me
        self.live, self.empty = [], False

    def _fill(self):
        while not self.empty and len(self.live) < self.bsz:
            s = self.feed()
            if s is None:
                self.empty = True
                break
            b = chess.Board()
            for m in s["opening"]:
                b.push_uci(m)
            self.live.append({"spec": s, "board": b, "hist": maia.Hist(self.cfg, b), "rec": [], "resign": False})

    def done(self) -> bool:
        self._fill()
        return not self.live

    def step(self) -> list[dict]:
        self._fill()
        mine = [g for g in self.live if g["board"].turn == g["spec"]["white"]]
        theirs = [g for g in self.live if g["board"].turn != g["spec"]["white"]]
        fut = pool().map_async(_sf, [(g["spec"]["opp"], [m.uci() for m in g["board"].move_stack], g["spec"]["seed"])
                                     for g in theirs])
        if mine:
            for g, mv in zip(mine, self.mover(mine)):
                if mv is not None:
                    g["board"].push(mv)
                    g["hist"].push(g["board"])
        for g, m in zip(theirs, fut.get()):
            g["board"].push_uci(m)
            g["hist"].push(g["board"])
        out, keep = [], []
        for g in self.live:
            b, s = g["board"], g["spec"]
            names = (self.me(s), s["opp"]) if s["white"] else (s["opp"], self.me(s))
            if g["resign"] or b.is_game_over(claim_draw=True) or b.ply() >= ladder.MAX_PLIES:
                if g["resign"]:
                    w, end = (0.0 if s["white"] else 1.0), "resigned (training adjudication)"
                else:
                    w, end = ladder.ending(b)
                r = w if s["white"] else 1 - w
                out.append({**s, "score": r, "plies": b.ply(), "end": end, "moves": [m.uci() for m in b.move_stack],
                            "rec": g["rec"]})
                res = {1.0: "1-0", 0.0: "0-1"}.get(w, "1/2-1/2")
                progress.board(self.name, s["seed"], b, *names, result=f"{res} · {end}")
                progress.event(self.name, f"{self.me(s)} {'beat' if r == 1 else 'lost to' if r == 0 else 'drew'}"
                                          f" {s['opp']} · {b.ply()} plies · {end}")
            else:
                keep.append(g)
        self.live = keep
        if keep:
            g = keep[0]
            s = g["spec"]
            progress.board(self.name, s["seed"], g["board"],
                           *((self.me(s), s["opp"]) if s["white"] else (s["opp"], self.me(s))))
        return out
