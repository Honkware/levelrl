from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

import chess

from .errors import InvalidPredictionError, MissingDataError, UnsupportedMetricError, ValidationError
from .positions import MovePredictor, ProbePosition


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    lower: float
    upper: float
    confidence: float = 0.95

    def __post_init__(self) -> None:
        try:
            lower, upper, confidence = float(self.lower), float(self.upper), float(self.confidence)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValidationError("confidence interval values must be numeric") from error
        if not all(math.isfinite(value) for value in (lower, upper)):
            raise ValidationError("confidence interval bounds must be finite", lower=self.lower, upper=self.upper)
        if lower > upper:
            raise ValidationError("confidence interval bounds are reversed", lower=self.lower, upper=self.upper)
        if not 0 < confidence < 1:
            raise ValidationError("confidence must be between zero and one", confidence=self.confidence)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "confidence", confidence)

    def as_list(self) -> list[float]:
        return [self.lower, self.upper]


@dataclass(frozen=True, slots=True)
class MoveScore:
    top1: bool
    log_loss: float


def score_move(position: ProbePosition, predictor: MovePredictor) -> MoveScore:
    raw = predictor.predict(position)
    if not isinstance(raw, Mapping) or not raw:
        raise InvalidPredictionError("predictor must return a non-empty move-probability mapping")
    board = position.board()
    probabilities: dict[str, float] = {}
    for move, value in raw.items():
        try:
            parsed = chess.Move.from_uci(move)
            probability = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise InvalidPredictionError("predictions must contain UCI moves and numeric probabilities",
                                         move=move, value=value) from error
        if parsed not in board.legal_moves:
            raise InvalidPredictionError("predictor returned an illegal move", move=move, fen=position.fen)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise InvalidPredictionError("probabilities must be finite and between zero and one",
                                         move=move, probability=probability)
        probabilities[move] = probability
    if position.move not in probabilities:
        raise MissingDataError("predictor omitted the human move", move=position.move)
    if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise InvalidPredictionError("move probabilities must sum to one", total=sum(probabilities.values()))
    target = probabilities[position.move]
    if target == 0:
        raise InvalidPredictionError("the human move must have non-zero probability", move=position.move)
    predicted = max(probabilities, key=lambda move: probabilities[move])
    return MoveScore(predicted == position.move, -math.log(target))


def evaluate_probe_set(positions: Sequence[ProbePosition], predictor: MovePredictor) -> tuple[MoveScore, ...]:
    if not positions:
        raise MissingDataError("cannot evaluate an empty probe set")
    return tuple(score_move(position, predictor) for position in positions)


def summarize(scores: Sequence[MoveScore], *, metrics: Sequence[str] = ("top1", "log_loss")) -> dict[str, int | float]:
    supported = {"top1", "log_loss"}
    unsupported = set(metrics) - supported
    if unsupported:
        raise UnsupportedMetricError("unsupported summary metric", metrics=sorted(unsupported),
                                     supported=sorted(supported))
    if not scores:
        raise MissingDataError("cannot summarize an empty score sequence")
    result: dict[str, int | float] = {"n": len(scores)}
    for metric in metrics:
        if metric == "top1":
            result[metric] = sum(score.top1 for score in scores) / len(scores)
        else:
            result[metric] = sum(score.log_loss for score in scores) / len(scores)
    return result


def _interval(sorted_values: Sequence[float], confidence: float) -> ConfidenceInterval:
    tail = (1 - confidence) / 2
    last = len(sorted_values) - 1
    lower = min(last, int(tail * len(sorted_values)))
    upper = min(last, int((1 - tail) * len(sorted_values)))
    return ConfidenceInterval(sorted_values[lower], sorted_values[upper], confidence)


def paired_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    if len(a) != len(b):
        raise ValidationError("paired samples must have equal lengths", a=len(a), b=len(b))
    if not a:
        raise MissingDataError("cannot bootstrap an empty paired sample")
    if samples <= 0:
        raise ValidationError("bootstrap sample count must be positive", samples=samples)
    if not 0 < confidence < 1:
        raise ValidationError("confidence must be between zero and one", confidence=confidence)
    differences = []
    for index, (left, right) in enumerate(zip(a, b)):
        try:
            difference = float(left) - float(right)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValidationError("paired samples must be numeric", index=index) from error
        if not math.isfinite(difference):
            raise ValidationError("paired samples must be finite", index=index)
        differences.append(difference)
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        draws.append(sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences))
    return _interval(sorted(draws), confidence)
