import random

import chess
import torch

from maia3.dataset import get_legal_moves_mask, tokenize_board

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

    positions = [b, chess.Board("8/P6k/8/8/8/8/7p/4K2R w - - 0 1"), chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")]
    for pos in positions:
        assert torch.equal(maia.encode_board(pos), tokenize_board(pos))
        assert torch.equal(maia.legal(pos), get_legal_moves_mask(pos, maia.MOVE_IDX))
    replayed, x = maia.replay(cfg, [m.uci() for m in b.move_stack])
    assert replayed.fen() == b.fen()
    assert torch.equal(x, maia.tokens(cfg, b, True))
