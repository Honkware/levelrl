from __future__ import annotations

import multiprocessing as mp
import queue
import random
import time

import chess

from levelrl import ladder, maia, progress

# Many Maia-vs-rung games at once. A rung's turn is a job in a process pool: the worker replays the game, plays the
# engine move, encodes the positions and checks for the end, so the main process only batches Maia's forwards and
# hands turns back out. Nothing waits for a whole round of moves, which keeps every engine worker busy. Before
# this the main process did the chess bookkeeping for all games between rounds and the engines sat idle about
# half the time (load 10 of 24 with one Python thread pinned).
# Spec fields: opp (rung name), white (Maia plays white), opening (uci list), seed, plus anything the caller wants
# carried through to the finished record.

_pool = None
ENGINE_TIMEOUT = 60
PROCS = 22  # of 24 threads: one stays free for this process and one for the GPU driver thread
BATCH_WAIT = 0.01  # seconds to let more engine replies arrive, so Maia's forward is not run for one game at a time


def pool(procs: int = PROCS) -> mp.pool.Pool:
    global _pool
    if _pool is None:
        _pool = mp.get_context("spawn").Pool(procs)
    return _pool


def close_pool():
    global _pool
    if _pool is not None:
        p, _pool = _pool, None
        p.terminate()
        p.join()


def _encode(b: chess.Board):
    if maia.native.available():
        t, m = maia.native.features(b)
    else:
        t, m = maia.encode_board(b), maia.legal(b)
    return t.numpy(), m.numpy()


def _over(b: chess.Board):
    if b.is_game_over(claim_draw=True) or b.ply() >= ladder.MAX_PLIES:
        return ladder.ending(b)
    return None


def _turn(args):
    # One rung turn, in a worker. after_maia says the last move in the list is Maia's and not yet encoded.
    gid, name, moves, seed, after_maia = args
    b = chess.Board()
    for m in moves:
        b.push_uci(m)
    pre = _encode(b)[0] if after_maia else None
    end = _over(b)
    if end:
        return gid, None, pre, None, None, end
    mv = ladder.rung(name).move(b, random.Random(seed * 1000 + b.ply()), game=seed)
    b.push(mv)
    tok, mask = _encode(b)
    return gid, mv.uci(), pre, tok, mask, _over(b)


class Arena:
    def __init__(self, cfg, mover, feed, name: str, bsz: int = 4 * PROCS, me=lambda s: "maia"):
        # bsz is how many games are in play at once. Four per worker keeps the engine queue from running dry, and
        # more only makes each game take longer, which in training means moves sampled from an older policy.
        self.cfg, self.mover, self.feed, self.name, self.bsz, self.me = cfg, mover, feed, name, bsz, me
        self.live, self.wait, self.empty = {}, [], False
        self.q, self.out, self.n = queue.SimpleQueue(), 0, 0

    def _send(self, g, after_maia: bool):
        s = g["spec"]
        pool().apply_async(_turn, ((g["id"], s["opp"], list(g["uci"]), s["seed"], after_maia),),
                           callback=self.q.put, error_callback=self.q.put)
        self.out += 1

    def _fill(self):
        while not self.empty and len(self.live) < self.bsz:
            s = self.feed()
            if s is None:
                self.empty = True
                break
            b = chess.Board()
            for m in s["opening"]:
                b.push_uci(m)
            self.n += 1
            g = {"id": self.n, "spec": s, "board": b, "uci": list(s["opening"]), "hist": maia.Hist(self.cfg, b),
                 "rec": [], "resign": False}
            self.live[g["id"]] = g
            if b.turn == s["white"]:
                self.wait.append(g)
            else:
                self._send(g, False)

    def done(self) -> bool:
        self._fill()
        return not self.live

    def _replies(self) -> list:
        # Blocks for the first engine reply when nothing else can be done, then takes what else arrives shortly.
        got = []
        if self.out and not self.wait:
            try:
                got.append(self.q.get(timeout=ENGINE_TIMEOUT))
            except queue.Empty:
                raise TimeoutError(f"no engine reply in {ENGINE_TIMEOUT} s") from None
            # With the engine queue full a short wait fills Maia's batch. Near the end of a gauntlet the workers
            # are waiting on us, and every millisecond here is a millisecond they sit idle.
            stop = time.monotonic() + (BATCH_WAIT if self.out >= 2 * PROCS else 0.0)
            while len(got) < min(self.out, 32) and time.monotonic() < stop:
                try:
                    got.append(self.q.get(timeout=max(0.0, stop - time.monotonic())))
                except queue.Empty:
                    break
        while True:
            try:
                got.append(self.q.get_nowait())
            except queue.Empty:
                return got

    def _names(self, s):
        return (self.me(s), s["opp"]) if s["white"] else (s["opp"], self.me(s))

    def _finish(self, g, w: float, end: str) -> dict:
        b, s = g["board"], g["spec"]
        del self.live[g["id"]]
        r = w if s["white"] else 1 - w
        res = {1.0: "1-0", 0.0: "0-1"}.get(w, "1/2-1/2")
        progress.board(self.name, s["seed"], b, *self._names(s), result=f"{res} · {end}")
        progress.event(self.name, f"{self.me(s)} {'beat' if r == 1 else 'lost to' if r == 0 else 'drew'}"
                                  f" {s['opp']} · {b.ply()} plies · {end}")
        return {**s, "score": r, "plies": b.ply(), "end": end, "moves": g["uci"], "rec": g["rec"]}

    def step(self) -> list[dict]:
        self._fill()
        out = []
        try:
            for r in self._replies():
                if isinstance(r, BaseException):
                    raise r
                gid, mv, pre, tok, mask, end = r
                self.out -= 1
                g = self.live[gid]
                if pre is not None:
                    g["hist"].add(pre)
                if mv is not None:
                    g["board"].push_uci(mv)
                    g["uci"].append(mv)
                    g["hist"].add(tok, mask)
                    progress.board(self.name, g["spec"]["seed"], g["board"], *self._names(g["spec"]))
                if end:
                    out.append(self._finish(g, *end))
                else:
                    self.wait.append(g)
            if self.wait:
                mine, self.wait = self.wait, []
                for g, mv in zip(mine, self.mover(mine)):
                    if mv is None:
                        out.append(self._finish(g, 0.0 if g["spec"]["white"] else 1.0,
                                                "resigned (training adjudication)"))
                    else:
                        g["board"].push(mv)
                        g["uci"].append(mv.uci())
                        self._send(g, True)
        except BaseException:
            close_pool()
            raise
        return out
