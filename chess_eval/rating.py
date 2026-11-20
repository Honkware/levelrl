from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import NormalDist
from typing import Mapping, Sequence

from .errors import MissingDataError, ValidationError
from .metrics import ConfidenceInterval, _interval


@dataclass(frozen=True, slots=True)
class GameResult:
    opponent: str
    score: float

    def __post_init__(self) -> None:
        if not self.opponent:
            raise ValidationError("game result needs an opponent")
        try:
            score = float(self.score)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValidationError("game score must be numeric", score=self.score) from error
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValidationError("game score must be between zero and one", score=self.score)
        object.__setattr__(self, "score", score)


@dataclass(frozen=True, slots=True)
class RatingEstimate:
    rating: float
    games: int
    score: float
    ci: ConfidenceInterval | None = None


def _checked_games(
    games: Sequence[GameResult], opponent_ratings: Mapping[str, float]
) -> dict[str, float]:
    if not games:
        raise MissingDataError("cannot estimate a rating without games")
    ratings = {}
    for game in games:
        if game.opponent not in opponent_ratings:
            raise MissingDataError("opponent rating is missing", opponent=game.opponent)
        try:
            rating = float(opponent_ratings[game.opponent])
        except (TypeError, ValueError, OverflowError) as error:
            raise ValidationError("opponent rating must be numeric", opponent=game.opponent) from error
        if not math.isfinite(rating):
            raise ValidationError("opponent rating must be finite", opponent=game.opponent)
        ratings[game.opponent] = rating
    return ratings


def estimate_rating(
    games: Sequence[GameResult],
    opponent_ratings: Mapping[str, float],
    *,
    minimum: float = -1000.0,
    maximum: float = 4500.0,
) -> float:
    ratings = _checked_games(games, opponent_ratings)
    if not minimum < maximum:
        raise ValidationError("rating bounds are invalid", minimum=minimum, maximum=maximum)

    def excess(rating: float) -> float:
        return sum(game.score - 1 / (1 + 10 ** ((ratings[game.opponent] - rating) / 400)) for game in games)

    low, high = minimum, maximum
    for _ in range(60):
        middle = (low + high) / 2
        if excess(middle) > 0:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def bootstrap_rating(
    games: Sequence[GameResult],
    opponent_ratings: Mapping[str, float],
    *,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    minimum: float = -1000.0,
    maximum: float = 4500.0,
) -> RatingEstimate:
    _checked_games(games, opponent_ratings)
    if samples <= 0:
        raise ValidationError("bootstrap sample count must be positive", samples=samples)
    if not 0 < confidence < 1:
        raise ValidationError("confidence must be between zero and one", confidence=confidence)
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        sample = rng.choices(games, k=len(games))
        draws.append(estimate_rating(sample, opponent_ratings, minimum=minimum, maximum=maximum))
    return RatingEstimate(
        estimate_rating(games, opponent_ratings, minimum=minimum, maximum=maximum),
        len(games),
        sum(game.score for game in games) / len(games),
        _interval(sorted(draws), confidence),
    )


@dataclass(frozen=True, slots=True)
class RatingPoint:
    source: float
    source_se: float
    target: float
    target_se: float

    def __post_init__(self) -> None:
        try:
            values = tuple(float(value) for value in (self.source, self.source_se, self.target, self.target_se))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValidationError("rating points and their standard errors must be numeric") from error
        if not all(math.isfinite(value) for value in values):
            raise ValidationError("rating points and their standard errors must be finite")
        for name, value in zip(("source", "source_se", "target", "target_se"), values):
            object.__setattr__(self, name, value)
        if self.source_se < 0 or self.target_se < 0 or self.source_se == self.target_se == 0:
            raise ValidationError("standard errors must be non-negative and not both zero")


@dataclass(frozen=True, slots=True)
class LinearFit:
    offset: float
    slope: float
    offset_ci: ConfidenceInterval
    slope_ci: ConfidenceInterval
    residuals: tuple[float, ...]
    draws: tuple[tuple[float, float], ...] = ()

    def convert(self, source: float) -> float:
        return self.offset + self.slope * source

    def invert(self, target: float) -> float:
        if self.slope == 0:
            raise ValidationError("cannot invert a zero-slope rating conversion")
        return (target - self.offset) / self.slope

    def interval(
        self, source: float, source_ci: ConfidenceInterval | Sequence[float]
    ) -> ConfidenceInterval:
        if isinstance(source_ci, ConfidenceInterval):
            low, high = source_ci.lower, source_ci.upper
            confidence = source_ci.confidence
        else:
            if len(source_ci) != 2:
                raise ValidationError("source interval must have two bounds")
            low, high = source_ci
            confidence = 0.95
        if not math.isfinite(source) or not all(math.isfinite(value) for value in (low, high)) or low > high:
            raise ValidationError("source rating and interval are invalid", source=source, interval=[low, high])
        fitted = [offset + slope * source for offset, slope in self.draws]
        if fitted:
            mean = sum(fitted) / len(fitted)
            sd = (sum((value - mean) ** 2 for value in fitted) / len(fitted)) ** 0.5
        else:
            sd = 0.0
        z = NormalDist().inv_cdf((1 + confidence) / 2)
        half = ((abs(self.slope) * (high - low) / 2) ** 2 + (z * sd) ** 2) ** 0.5
        center = self.convert(source)
        return ConfidenceInterval(center - half, center + half, confidence)


@dataclass(frozen=True, slots=True)
class RatingFit:
    offset: LinearFit
    affine: LinearFit
    use: str

    def __post_init__(self) -> None:
        if self.use not in {"offset", "affine"}:
            raise ValidationError("rating fit selection must be offset or affine", use=self.use)

    @property
    def selected(self) -> LinearFit:
        return self.affine if self.use == "affine" else self.offset


def _wls(points: Sequence[RatingPoint], slope: bool) -> tuple[float, float]:
    weights = [1 / (point.source_se ** 2 + point.target_se ** 2) for point in points]
    total = sum(weights)
    mean_x = sum(weight * point.source for weight, point in zip(weights, points)) / total
    mean_y = sum(weight * point.target for weight, point in zip(weights, points)) / total
    if not slope:
        return mean_y - mean_x, 1.0
    sxx = sum(weight * (point.source - mean_x) ** 2 for weight, point in zip(weights, points))
    if sxx == 0:
        raise ValidationError("affine rating fit needs at least two distinct source ratings")
    covariance = sum(weight * (point.source - mean_x) * (point.target - mean_y)
                     for weight, point in zip(weights, points))
    fitted_slope = covariance / sxx
    return mean_y - fitted_slope * mean_x, fitted_slope


def fit_rating_scale(
    points: Sequence[RatingPoint], *, samples: int = 4000, confidence: float = 0.95, seed: int = 0
) -> RatingFit:
    if len(points) < 2:
        raise MissingDataError("rating-scale fit needs at least two points")
    if samples <= 0:
        raise ValidationError("bootstrap sample count must be positive", samples=samples)
    if not 0 < confidence < 1:
        raise ValidationError("confidence must be between zero and one", confidence=confidence)
    rng = random.Random(seed)
    fits = {}
    for name, with_slope in (("offset", False), ("affine", True)):
        offset, slope = _wls(points, with_slope)
        draws = []
        for _ in range(samples):
            sampled = [RatingPoint(rng.gauss(point.source, point.source_se), point.source_se,
                                   rng.gauss(point.target, point.target_se), point.target_se) for point in points]
            draws.append(_wls(sampled, with_slope))
        fits[name] = LinearFit(
            offset,
            slope,
            _interval(sorted(draw[0] for draw in draws), confidence),
            _interval(sorted(draw[1] for draw in draws), confidence),
            tuple(point.target - (offset + slope * point.source) for point in points),
            tuple(draws[::max(1, samples // 400)]),
        )
    slope_ci = fits["affine"].slope_ci
    use = "affine" if not slope_ci.lower <= 1.0 <= slope_ci.upper else "offset"
    return RatingFit(fits["offset"], fits["affine"], use)
