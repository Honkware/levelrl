import chess
import pytest
import torch

from levelrl import maia, native
from tests.test_maia import positions, promotions

pytestmark = pytest.mark.skipif(not native.available(), reason="native library is not built")


def test_native_features_match_python_and_maia_on_1204_positions():
    special = [
        chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
        chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"),
        chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"),
        chess.Board("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1"),
    ]
    for board in positions(7) + promotions(9) + special:
        tokens, mask = native.features(board)
        assert torch.equal(tokens, maia.encode_board(board)), board.fen()
        assert torch.equal(mask, maia.legal(board)), board.fen()


def test_persistent_native_board_matches_stateless_features_across_games():
    import random

    rng = random.Random(3)
    for _ in range(100):
        board = chess.Board()
        native_board = native.NativeBoard(board)
        try:
            for _ in range(12):
                move = rng.choice(list(board.legal_moves))
                board.push(move)
                native_board.push_uci(move.uci())
                expected_tokens, expected_mask = native.features(board)
                tokens, mask = native_board.features()
                assert torch.equal(tokens, expected_tokens)
                assert torch.equal(mask, expected_mask)
        finally:
            native_board.close()
