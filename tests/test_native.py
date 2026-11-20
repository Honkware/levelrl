import gzip
import json
import subprocess
from pathlib import Path

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


def test_rust_scan_and_band_build_preserve_schema_and_all_bands(tmp_path):
    from tests.test_data import GAME

    scan = tmp_path / "data" / "scan" / "2026-03"
    scan.mkdir(parents=True)
    pgn = tmp_path / "games.pgn"
    pgn.write_text("\n".join(GAME.format(late="0:02:35").replace("/abc", f"/g{i}") for i in range(60)))
    archive = tmp_path / "games.zst"
    subprocess.run(["zstd", "-q", "-o", str(archive), str(pgn)], check=True)
    binary = Path(__file__).parent.parent / "native" / "target" / "release" / "scan"
    subprocess.run(f"zstd -dc '{archive}' | '{binary}' --month 2026-03 --keep-bp 10000 --out '{scan}'",
                   shell=True, check=True)
    counts = json.loads((scan / "counts.json").read_text())
    assert counts == {"a": 60, "b": 60}

    band = Path(__file__).parent.parent / "native" / "target" / "release" / "band"
    out = tmp_path / "positions"
    subprocess.run([str(band), "--root", str(tmp_path), "--months", "2026-03", "--bands", "1450 1500",
                    "--n-cal", "0", "--out", str(out)], check=True)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["bands"] == [1450, 1500] and manifest["months"] == ["2026-03"]
    for target, expected in ((1450, 60), (1500, 0)):
        rows = [json.loads(line) for line in gzip.open(out / f"fid_{target}.jsonl.gz", "rt")]
        assert len(rows) == expected
        assert all({"site", "moves", "move", "elo", "oppo_elo", "white", "black", "tc", "month"} <= set(r)
                   for r in rows)
        assert all("-" not in move for r in rows for move in r["moves"] + [r["move"]])
