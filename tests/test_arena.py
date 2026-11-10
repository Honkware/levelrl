import random
import shutil

import chess
import pytest
import torch

from levelrl import arena, ladder, maia


def test_pool_uses_spawn_instead_of_inheriting_cuda_and_engine_threads(monkeypatch):
    worker_pool = object()

    class Context:
        def Pool(self, procs):
            assert procs == 3
            return worker_pool

    methods = []
    monkeypatch.setattr(arena, "_pool", None)
    monkeypatch.setattr(arena.mp, "get_context", lambda method: methods.append(method) or Context())
    assert arena.pool(3) is worker_pool
    assert methods == ["spawn"]


def test_engine_timeout_discards_the_pool(monkeypatch):
    class Pool:
        def apply_async(self, fn, args, callback, error_callback):
            assert fn.__name__ == "_turn"  # the reply never comes

    class Hist:
        def __init__(self, cfg, board):
            pass

    discarded = []
    monkeypatch.setattr(arena, "pool", lambda: Pool())
    monkeypatch.setattr(arena, "close_pool", lambda: discarded.append(True))
    monkeypatch.setattr(arena, "ENGINE_TIMEOUT", 0.05)
    monkeypatch.setattr(arena.maia, "Hist", Hist)
    specs = iter([{"opp": "elo1500", "opening": [], "white": False, "seed": 1}])
    a = arena.Arena(object(), lambda games: [], lambda: next(specs, None), "test")
    with pytest.raises(TimeoutError):
        a.step()
    assert discarded == [True]


@pytest.mark.skipif(shutil.which("stockfish") is None, reason="needs stockfish")
def test_games_played_through_the_workers_keep_the_history_a_full_replay_would_build(monkeypatch, tmp_path):
    monkeypatch.setattr(arena.progress, "ROOT", tmp_path)
    monkeypatch.setattr(arena, "_pool", None)
    cfg, _ = maia.load("cpu")
    rng = random.Random(0)
    seen = [0]

    def mover(games):
        out = []
        for g in games:
            b = g["board"]
            assert [m.uci() for m in b.move_stack] == g["uci"]
            assert torch.equal(g["hist"].tokens(), maia.tokens(cfg, b, True))
            assert torch.equal(g["hist"].legal_mask(), maia.legal(b))
            seen[0] += 1
            out.append(rng.choice(list(b.legal_moves)))
        return out

    todo = [{"opp": "sk0_r50", "opening": ["e2e4", "e7e5"], "white": w, "seed": i} for i, w in enumerate([True, False] * 3)]
    a = arena.Arena(cfg, mover, lambda: todo.pop() if todo else None, "test", bsz=4)
    try:
        arena.pool(2)
        done = []
        while not a.done():
            done += a.step()
    finally:
        arena.close_pool()
    assert len(done) == 6 and seen[0] > 60
    for g in done:
        b = chess.Board()
        for m in g["moves"]:
            b.push_uci(m)  # raises on an illegal move
        w, end = ladder.ending(b)
        assert g["plies"] == b.ply() and g["end"] == end and g["score"] == (w if g["white"] else 1 - w)


def test_close_pool_terminates_and_joins_workers(monkeypatch):
    class Pool:
        calls = []

        def terminate(self):
            self.calls.append("terminate")

        def join(self):
            self.calls.append("join")

    p = Pool()
    monkeypatch.setattr(arena, "_pool", p)
    arena.close_pool()
    assert p.calls == ["terminate", "join"]
    assert arena._pool is None
