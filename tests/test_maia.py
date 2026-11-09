from __future__ import annotations

import random

import chess
import pytest

from maia3.uci import Maia3UCIEngine, parse_args
from levelrl import maia

N = 1000


@pytest.fixture(scope="module")
def model():
    return maia.load("cpu")


def positions(seed: int) -> list[chess.Board]:
    rng, out = random.Random(seed), []
    while len(out) < N:
        b = chess.Board()
        for _ in range(rng.randrange(1, 120)):
            if b.is_game_over():
                break
            b.push(rng.choice(list(b.legal_moves)))
        if not b.is_game_over():
            out.append(b)
    return out


def promotions(seed: int) -> list[chess.Board]:
    # Pawns one step from promoting, for both colours, so every promotion index gets exercised.
    rng, out = random.Random(seed), []
    while len(out) < 200:
        f, turn = rng.randrange(8), rng.choice([chess.WHITE, chess.BLACK])
        b = chess.Board(None)
        b.set_piece_at(chess.E1 if turn == chess.BLACK else chess.E8, chess.Piece(chess.KING, not turn))
        b.set_piece_at(chess.A1 if turn == chess.WHITE else chess.A8, chess.Piece(chess.KING, turn))
        b.set_piece_at(chess.square(f, 6 if turn == chess.WHITE else 1), chess.Piece(chess.PAWN, turn))
        b.turn = turn
        if b.is_valid() and not b.is_game_over():
            out.append(b)
    return out


@pytest.mark.parametrize("history", [False, True])
def test_batched_argmax_matches_the_uci_engine(model, history):
    cfg, m = model
    ecfg = parse_args(["--model", "maia3-79m", "--device", "cpu", "--temperature", "0"]
                      + (["--use-uci-history"] if history else []))
    eng = Maia3UCIEngine(ecfg)
    eng.model = m
    boards = positions(1)
    elos = [random.Random(i).randrange(600, 2600, 100) for i in range(len(boards))]
    logits, _ = maia.forward(cfg, m, boards, elos, elos, history=history)
    for b, e, lg in zip(boards, elos, logits):
        eng.self_elo = eng.oppo_elo = e
        eng.cmd_position("position startpos moves " + " ".join(mv.uci() for mv in b.move_stack))
        assert eng.score_moves()[0] == maia.to_move(b, int(lg.argmax()))


def test_every_legal_move_maps_to_a_legal_index_and_back(model):
    for b in positions(2) + promotions(3):
        logits, _ = maia.forward(*model, [b], [1500], [1500])
        idx = (logits[0] > float("-inf")).nonzero().flatten().tolist()
        assert sorted(maia.to_move(b, i).uci() for i in idx) == sorted(mv.uci() for mv in b.legal_moves)
