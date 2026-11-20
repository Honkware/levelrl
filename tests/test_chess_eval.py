"""Local fixtures cover probe schemas, metrics, ratings, and adapter contracts."""

from __future__ import annotations

import gzip
import json
import math
import random

import chess
import pytest

from chess_eval import (
    ChessEvalError,
    DataError,
    DatasetManifest,
    Engine,
    EngineEvaluation,
    GameResult,
    InvalidPositionError,
    InvalidPredictionError,
    MissingDataError,
    MoveAnalysis,
    MoveScore,
    ProbePosition,
    RatingPoint,
    UnsupportedMetricError,
    ValidationError,
    analyse_move,
    bootstrap_rating,
    estimate_rating,
    evaluate_probe_set,
    fit_rating_scale,
    human_likeness,
    load_probe_set,
    paired_ci,
    split_probe_set,
    summarize,
)


class Predictor:
    def __init__(self, target_probability=0.7):
        self.target_probability = target_probability

    def predict(self, position):
        moves = [move.uci() for move in position.board().legal_moves]
        remainder = (1 - self.target_probability) / (len(moves) - 1)
        return {move: self.target_probability if move == position.move else remainder for move in moves}


def test_probe_scoring_is_model_agnostic_and_summarizes_both_metrics():
    positions = [ProbePosition.from_moves([], "e2e4"), ProbePosition.from_moves(["e2e4"], "e7e5")]
    scores = evaluate_probe_set(positions, Predictor())
    result = summarize(scores)
    assert result["n"] == 2
    assert result["top1"] == 1.0
    assert result["log_loss"] == pytest.approx(-math.log(0.7))


def test_invalid_positions_and_predictions_have_structured_errors():
    with pytest.raises(InvalidPositionError) as caught:
        ProbePosition(chess.STARTING_FEN, "e2e5")
    assert caught.value.as_dict()["code"] == "invalid_position"
    assert caught.value.context["move"] == "e2e5"

    class ZeroPredictor:
        def predict(self, position):
            return {position.move: 0.0, "d2d4": 1.0}

    with pytest.raises(InvalidPredictionError):
        evaluate_probe_set([ProbePosition.from_moves([], "e2e4")], ZeroPredictor())


def test_summary_rejects_unsupported_metrics_and_missing_scores():
    with pytest.raises(UnsupportedMetricError) as caught:
        summarize([MoveScore(True, 0.0)], metrics=["accuracy"])
    assert caught.value.context["metrics"] == ["accuracy"]
    with pytest.raises(MissingDataError):
        summarize([])


def test_paired_ci_is_deterministic_and_validates_pairing():
    assert paired_ci([1.0] * 20, [0.0] * 20).as_list() == [1.0, 1.0]
    a, b = [0.0, 1.0, 2.0, 3.0], [0.0, 0.5, 1.0, 1.5]
    assert paired_ci(a, b, samples=200) == paired_ci(a, b, samples=200)
    with pytest.raises(ValidationError):
        paired_ci([1.0], [])


def test_jsonl_probe_loading_retains_context_and_enforces_cutoff(tmp_path):
    path = tmp_path / "positions.jsonl.gz"
    rows = [
        {"moves": [], "move": "e2e4", "site": "game-1", "month": "2025-09", "elo": 1500},
        {"fen": chess.STARTING_FEN, "move": "d2d4", "month": "2026-03", "context": {"source": "test"}},
    ]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    positions = load_probe_set(path, cutoff="2025-07")
    assert len(positions) == 2
    assert positions[0].game_id == "game-1"
    assert positions[0].context["elo"] == 1500
    assert positions[1].context["source"] == "test"

    rows[0]["month"] = "2025-07"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(rows[0]) + "\n")
    with pytest.raises(DataError) as caught:
        load_probe_set(path, cutoff="2025-07")
    assert caught.value.context["line"] == 1


def test_probe_loading_reports_missing_and_malformed_data(tmp_path):
    with pytest.raises(MissingDataError):
        load_probe_set(tmp_path / "absent.jsonl")
    path = tmp_path / "bad.jsonl"
    path.write_text('{"fen": "not a fen", "move": "e2e4"}\n', encoding="utf-8")
    with pytest.raises(InvalidPositionError):
        load_probe_set(path)


def test_probe_split_is_disjoint_and_never_silently_overlaps():
    positions = [ProbePosition.from_moves([], "e2e4", game_id=str(index)) for index in range(8)]
    split = split_probe_set(positions, probe_size=3, validation_size=2, fidelity_size=3)
    assert [row.game_id for row in split["probe"]] == ["0", "1", "2"]
    assert [row.game_id for row in split["validation"]] == ["3", "4"]
    assert [row.game_id for row in split["fidelity"]] == ["5", "6", "7"]
    with pytest.raises(MissingDataError):
        split_probe_set(positions[:5], probe_size=3, validation_size=1, fidelity_size=2)


def test_manifest_round_trip_and_leakage_guard(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = DatasetManifest("2025-07", ("2026-03", "2025-09"), datasets={"test": 2})
    manifest.save(path)
    loaded = DatasetManifest.load(path)
    assert loaded == manifest
    assert loaded.months == ("2025-09", "2026-03")
    loaded.require_month("2025-09")
    with pytest.raises(MissingDataError):
        loaded.require_month("2026-09")
    with pytest.raises(DataError):
        DatasetManifest("2025-07", ("2025-07",))
    with pytest.raises(ChessEvalError):
        DatasetManifest.load(tmp_path / "missing.json")


def test_manifest_schema_rejects_unknown_fields_and_string_months():
    with pytest.raises(DataError):
        DatasetManifest.from_dict({"cutoff": "2025-07", "months": ["2025-09"], "extra": True})
    with pytest.raises(DataError):
        DatasetManifest.from_dict({"cutoff": "2025-07", "months": "2025-09"})


def test_probe_schema_rejects_ambiguous_positions_and_invalid_context():
    row = {"fen": chess.STARTING_FEN, "moves": [], "move": "e2e4"}
    with pytest.raises(DataError):
        ProbePosition.from_row(row)
    with pytest.raises(DataError):
        ProbePosition.from_row({"fen": chess.STARTING_FEN, "move": "e2e4", "context": []})


def test_engine_protocol_and_human_likeness_summary():
    class FakeEngine:
        def evaluate(self, board, *, nodes):
            assert nodes == 10
            return EngineEvaluation(100 if board.turn == chess.WHITE else 150)

    assert isinstance(FakeEngine(), Engine)
    analysis = analyse_move(FakeEngine(), ProbePosition.from_moves([], "e2e4"), nodes=10)
    assert analysis == MoveAnalysis(250, "opening", "tactical oversight")
    result = human_likeness([analysis, MoveAnalysis(0, "opening")])
    assert result["n"] == 2
    assert result["blunder_rate"] == 0.5
    assert result["by_phase"]["opening"]["mean_cp_loss"] == 125


def test_game_rating_recovers_known_strength_and_reports_interval():
    ratings = {"a": 1200.0, "b": 1500.0, "c": 1800.0}
    rng = random.Random(0)
    games = []
    for opponent, rating in ratings.items():
        for _ in range(400):
            expected = 1 / (1 + 10 ** ((rating - 1450) / 400))
            games.append(GameResult(opponent, float(rng.random() < expected)))
    assert estimate_rating(games, ratings) == pytest.approx(1450, abs=35)
    estimate = bootstrap_rating(games[::20], ratings, samples=50)
    assert estimate.games == 60
    assert estimate.ci is not None and estimate.ci.lower < estimate.rating < estimate.ci.upper
    with pytest.raises(MissingDataError):
        estimate_rating([GameResult("unknown", 1.0)], ratings)


def test_rating_scale_fit_and_uncertainty():
    rng = random.Random(0)
    points = [RatingPoint(x, 20.0, 300 + 1.6 * x + rng.gauss(0, 15), 40.0)
              for x in range(700, 2100, 150)]
    fit = fit_rating_scale(points, samples=400)
    assert fit.use == "affine"
    assert fit.selected.slope == pytest.approx(1.6, abs=0.15)
    assert fit.selected.invert(fit.selected.convert(1400)) == pytest.approx(1400)
    near = fit.selected.interval(1400, [1370, 1430])
    far = fit.selected.interval(2200, [2170, 2230])
    assert near.lower < fit.selected.convert(1400) < near.upper
    assert far.upper - far.lower > near.upper - near.lower
