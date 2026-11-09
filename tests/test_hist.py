import random

import chess
import torch

from levelrl import maia


def test_incremental_history_matches_full_replay_tokens():
    cfg, _ = maia.load("cpu")
    rng = random.Random(0)
    for _ in range(20):
        b = chess.Board()
        for _ in range(rng.randint(0, 12)):
            b.push(rng.choice(list(b.legal_moves)))
        h = maia.Hist(cfg, b)
        for _ in range(rng.randint(1, 30)):
            if b.is_game_over():
                break
            b.push(rng.choice(list(b.legal_moves)))
            h.push(b)
            assert torch.equal(h.tokens(), maia.tokens(cfg, b, True))
