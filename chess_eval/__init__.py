from __future__ import annotations

from .analysis import Engine, EngineEvaluation, MoveAnalysis, analyse_move, game_phase, human_likeness
from .errors import (
    ChessEvalError,
    DataError,
    InvalidPositionError,
    InvalidPredictionError,
    MissingDataError,
    UnsupportedMetricError,
    ValidationError,
)
from .manifest import DatasetManifest, check_month
from .metrics import ConfidenceInterval, MoveScore, evaluate_probe_set, paired_ci, score_move, summarize
from .positions import MovePredictor, ProbePosition, load_probe_set, split_probe_set
from .rating import (
    GameResult,
    LinearFit,
    RatingEstimate,
    RatingFit,
    RatingPoint,
    bootstrap_rating,
    estimate_rating,
    fit_rating_scale,
)

__all__ = [
    "ChessEvalError", "ConfidenceInterval", "DataError", "DatasetManifest", "Engine", "EngineEvaluation",
    "GameResult", "InvalidPositionError", "InvalidPredictionError", "LinearFit", "MissingDataError",
    "MoveAnalysis", "MovePredictor", "MoveScore", "ProbePosition", "RatingEstimate", "RatingFit", "RatingPoint",
    "UnsupportedMetricError", "ValidationError", "analyse_move", "bootstrap_rating", "check_month",
    "estimate_rating", "evaluate_probe_set", "fit_rating_scale", "game_phase", "human_likeness", "load_probe_set",
    "paired_ci", "score_move", "split_probe_set", "summarize",
]

__version__ = "0.1.0"
