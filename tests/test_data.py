from __future__ import annotations

import gzip
import io
import json
import random

import chess.pgn
import pytest

from levelrl import data

GAME = """[Event "Rated Blitz game"]
[Site "https://lichess.org/abc"]
[White "a"]
[Black "b"]
[WhiteElo "1500"]
[BlackElo "1480"]
[TimeControl "180+0"]

1. e4 {{ [%clk 0:03:00] }} e5 {{ [%clk 0:03:00] }} 2. Nf3 {{ [%clk 0:02:58] }} Nc6 {{ [%clk 0:02:57] }}
3. Bb5 {{ [%clk 0:02:55] }} a6 {{ [%clk 0:02:50] }} 4. Ba4 {{ [%clk 0:02:50] }} Nf6 {{ [%clk 0:02:45] }}
5. O-O {{ [%clk 0:02:40] }} Be7 {{ [%clk 0:02:40] }} 6. Re1 {{ [%clk {late}] }} b5 {{ [%clk 0:02:30] }}
7. Bb3 {{ [%clk 0:02:30] }} d6 {{ [%clk 0:02:20] }} 1-0
"""


def game(late: str = "0:02:35") -> chess.pgn.Game:
    return chess.pgn.read_game(io.StringIO(GAME.format(late=late)))


def test_a_month_inside_the_training_window_stops_the_run():
    for m in ("2023-01", "2025-07"):
        with pytest.raises(SystemExit, match="leakage"):
            data.check_month(m)
    data.check_month("2025-08")


def test_loading_a_pre_cutoff_row_stops_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    (tmp_path / "positions").mkdir()
    with gzip.open(tmp_path / "positions" / "calib.jsonl.gz", "wt") as f:
        f.write(json.dumps({"month": "2025-07", "moves": []}) + "\n")
    with pytest.raises(SystemExit, match="leakage"):
        data.load("calib")


def test_bot_and_casual_games_are_not_human():
    assert data.human({"Event": "Rated Blitz game"})
    assert not data.human({"Event": "Rated Blitz game", "BlackTitle": "BOT"})
    assert not data.human({"Event": "Casual Blitz game"})


def test_picked_positions_skip_book_plies_and_low_clocks():
    for seed in range(50):
        row = data.pick(game(), random.Random(seed))
        assert data.SKIP_PLIES <= len(row["moves"]) < 14
        assert data.board(row).is_legal(chess.Move.from_uci(row["move"]))
    assert data.pick(game(late="0:00:20"), random.Random(0)) is None


def test_the_mover_and_opponent_ratings_follow_the_side_to_move():
    for seed in range(20):
        row = data.pick(game(), random.Random(seed))
        white = len(row["moves"]) % 2 == 0
        assert (row["elo"], row["oppo_elo"]) == ((1500, 1480) if white else (1480, 1500))
