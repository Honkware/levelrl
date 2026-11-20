from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .errors import MapValidationError, UnsupportedSchemaVersion, UnsupportedTargetError

SCHEMA_VERSION = 1
MIN_TARGET = 600
MAX_TARGET = 2600
MIN_DIAL = 0
MAX_DIAL = 5000
BAND = 50
UNITS = "estimated Lichess blitz strength from the current fixed-node ladder calibration"
INTERPOLATION = "exact measured settings only; unmeasured targets use the nearest measured configuration without claiming validation"


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MapValidationError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise MapValidationError(f"{name} must be a finite number")
    return value


def _target(value: object, *, measured: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise UnsupportedTargetError("target must be an integer")
    if not MIN_TARGET <= value <= MAX_TARGET:
        raise UnsupportedTargetError(f"target {value} is outside the supported range {MIN_TARGET}..{MAX_TARGET}")
    if measured and value % 100:
        raise MapValidationError("measured targets must be multiples of 100")
    return value


def _interval(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise MapValidationError(f"{name} must contain exactly two numbers")
    low = _finite(value[0], f"{name}[0]")
    high = _finite(value[1], f"{name}[1]")
    if low > high:
        raise MapValidationError(f"{name} lower bound exceeds its upper bound")
    return low, high


@dataclass(frozen=True, slots=True)
class Setting:
    method: str
    dial: int
    temp: float

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method.strip():
            raise MapValidationError("method must be a non-empty string")
        if isinstance(self.dial, bool) or not isinstance(self.dial, int):
            raise MapValidationError("dial must be an integer")
        if not MIN_DIAL <= self.dial <= MAX_DIAL:
            raise MapValidationError(f"dial must be in the range {MIN_DIAL}..{MAX_DIAL}")
        object.__setattr__(self, "temp", _finite(self.temp, "temp"))
        if self.temp <= 0:
            raise MapValidationError("temp must be greater than zero")

    @classmethod
    def from_dict(cls, row: dict) -> Setting:
        if not isinstance(row, dict):
            raise MapValidationError("setting must be an object")
        try:
            return cls(method=row["method"], dial=row["dial"], temp=row["temp"])
        except KeyError as exc:
            raise MapValidationError(f"setting is missing {exc.args[0]}") from exc


@dataclass(frozen=True, slots=True)
class Candidate:
    method: str
    dial: int
    temp: float
    validation_elo: float
    validation_sce: float | None = None
    reaches: bool | None = None

    def __post_init__(self) -> None:
        setting = Setting(self.method, self.dial, self.temp)
        object.__setattr__(self, "temp", setting.temp)
        object.__setattr__(self, "validation_elo", _finite(self.validation_elo, "validation_elo"))
        if self.validation_sce is not None:
            object.__setattr__(self, "validation_sce", _finite(self.validation_sce, "validation_sce"))
        if self.reaches is not None and not isinstance(self.reaches, bool):
            raise MapValidationError("reaches must be a boolean or null")

    @property
    def setting(self) -> Setting:
        return Setting(self.method, self.dial, self.temp)


@dataclass(frozen=True, slots=True)
class Measurement:
    target: int
    setting: Setting
    measured_elo: float
    elo_ci: tuple[float, float]
    elo_ci_units: tuple[float, float]

    def __post_init__(self) -> None:
        _target(self.target, measured=True)
        if not isinstance(self.setting, Setting):
            raise MapValidationError("setting must be a Setting")
        object.__setattr__(self, "measured_elo", _finite(self.measured_elo, "measured_elo"))
        object.__setattr__(self, "elo_ci", _interval(self.elo_ci, "elo_ci"))
        object.__setattr__(self, "elo_ci_units", _interval(self.elo_ci_units, "elo_ci_units"))

    @property
    def strength_in_band(self) -> bool:
        return abs(self.measured_elo - self.target) <= BAND

    @classmethod
    def from_dict(cls, row: dict) -> Measurement:
        if not isinstance(row, dict):
            raise MapValidationError("validated entries must be objects")
        try:
            target = row["target"]
            measured_elo = row["measured_elo"]
            elo_ci = row["elo_ci"]
            elo_ci_units = row["elo_ci_units"]
        except KeyError as exc:
            raise MapValidationError(f"validated entry is missing {exc.args[0]}") from exc
        if row.get("validated") is not True:
            raise MapValidationError(f"target {target} is not marked as validated")
        measurement = cls(target, Setting.from_dict(row), measured_elo, elo_ci, elo_ci_units)
        if "strength_in_band" in row and row["strength_in_band"] is not measurement.strength_in_band:
            raise MapValidationError(f"target {target} has an inconsistent strength_in_band flag")
        return measurement

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "method": self.setting.method,
            "dial": self.setting.dial,
            "temp": self.setting.temp,
            "measured_elo": self.measured_elo,
            "elo_ci": list(self.elo_ci),
            "elo_ci_units": list(self.elo_ci_units),
            "strength_in_band": self.strength_in_band,
            "validated": True,
        }


@dataclass(frozen=True, slots=True)
class ResolvedSetting:
    target: int
    setting: Setting
    nearest_target: int
    validated: bool
    nearest: bool
    interpolated: bool
    extrapolated: bool

    @property
    def method(self) -> str:
        return self.setting.method

    @property
    def dial(self) -> int:
        return self.setting.dial

    @property
    def temp(self) -> float:
        return self.setting.temp

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "method": self.method,
            "dial": self.dial,
            "temp": self.temp,
            "nearest_target": self.nearest_target,
            "nearest": self.nearest,
            "validated": self.validated,
            "interpolated": self.interpolated,
            "extrapolated": self.extrapolated,
        }


def select(candidates: Iterable[Candidate], target: int) -> Candidate:
    _target(target)
    rows = list(candidates)
    if not rows:
        raise ValueError("cannot select from an empty candidate list")
    if not all(isinstance(row, Candidate) for row in rows):
        raise TypeError("candidates must contain Candidate instances")
    inside = [row for row in rows if abs(row.validation_elo - target) <= BAND]
    if inside:
        return min(inside, key=lambda row: (
            math.inf if row.validation_sce is None else row.validation_sce,
            abs(row.validation_elo - target),
        ))
    return min(rows, key=lambda row: abs(row.validation_elo - target))


class CalibrationMap:
    def __init__(self, measurements: Iterable[Measurement] = ()) -> None:
        rows = tuple(sorted(measurements, key=lambda row: row.target))
        if not all(isinstance(row, Measurement) for row in rows):
            raise TypeError("measurements must contain Measurement instances")
        targets = [row.target for row in rows]
        if len(targets) != len(set(targets)):
            raise MapValidationError("validated targets must be unique")
        self._measurements = rows

    @property
    def measurements(self) -> tuple[Measurement, ...]:
        return self._measurements

    @classmethod
    def from_rows(cls, rows: Iterable[dict]) -> CalibrationMap:
        return cls(Measurement.from_dict(row) for row in rows)

    def lookup(self, target: int) -> ResolvedSetting:
        _target(target)
        if not self._measurements:
            raise UnsupportedTargetError("an empty calibration map has no measured settings")
        nearest = min(self._measurements, key=lambda row: (abs(row.target - target), row.target))
        exact = nearest.target == target
        return ResolvedSetting(
            target=target,
            setting=nearest.setting,
            nearest_target=nearest.target,
            validated=exact,
            nearest=not exact,
            interpolated=False,
            extrapolated=target < self._measurements[0].target or target > self._measurements[-1].target,
        )

    nearest = lookup

    def settings(self) -> list[ResolvedSetting]:
        if not self._measurements:
            return []
        return [self.lookup(target) for target in range(self._measurements[0].target, self._measurements[-1].target + 1, 100)]

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "units": UNITS,
            "interpolation": INTERPOLATION,
            "validated": [row.to_dict() for row in self._measurements],
            "settings": [row.to_dict() for row in self.settings()],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> CalibrationMap:
        if not isinstance(payload, dict):
            raise MapValidationError("calibration map must be a JSON object")
        legacy = "schema_version" not in payload
        version = payload.get("schema_version", SCHEMA_VERSION)
        if isinstance(version, bool) or not isinstance(version, int):
            raise MapValidationError("schema_version must be an integer")
        if version != SCHEMA_VERSION:
            raise UnsupportedSchemaVersion(f"unsupported calibration map schema version {version}")
        if payload.get("units") != UNITS:
            raise MapValidationError("calibration map has inconsistent units")
        if payload.get("interpolation") != INTERPOLATION:
            raise MapValidationError("calibration map has inconsistent interpolation metadata")
        rows = payload.get("validated")
        if not isinstance(rows, list):
            raise MapValidationError("validated must be a list")
        calibration = cls.from_rows(rows)
        calibration._check_settings(payload.get("settings"), legacy)
        return calibration

    def _check_settings(self, rows: object, legacy: bool) -> None:
        if not isinstance(rows, list):
            raise MapValidationError("settings must be a list")
        expected = [row.to_dict() for row in self.settings()]
        if len(rows) != len(expected):
            raise MapValidationError("settings do not match the validated measurements")
        legacy_keys = ("target", "method", "dial", "temp", "nearest_target", "interpolated", "extrapolated")
        for index, (actual, wanted) in enumerate(zip(rows, expected)):
            if not isinstance(actual, dict):
                raise MapValidationError(f"settings[{index}] must be an object")
            matches = all(actual.get(key) == wanted[key] for key in legacy_keys)
            if not legacy:
                matches = actual == wanted
            if not matches:
                raise MapValidationError(f"settings[{index}] is inconsistent with nearest-measured lookup")

    @classmethod
    def load(cls, path: str | Path) -> CalibrationMap:
        return load_map(path)

    def save(self, path: str | Path) -> None:
        save_map(self, path)


def nearest_measured(calibration: CalibrationMap, target: int) -> ResolvedSetting:
    if not isinstance(calibration, CalibrationMap):
        raise TypeError("calibration must be a CalibrationMap")
    return calibration.lookup(target)


def load_map(path: str | Path) -> CalibrationMap:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MapValidationError(f"invalid JSON in {path}: {exc.msg}") from exc
    return CalibrationMap.from_dict(payload)


def save_map(calibration: CalibrationMap, path: str | Path) -> None:
    if not isinstance(calibration, CalibrationMap):
        raise TypeError("calibration must be a CalibrationMap")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
            temporary_name = tmp.name
            json.dump(calibration.to_dict(), tmp, indent=1, allow_nan=False)
            tmp.write("\n")
            tmp.flush()
            os.fsync(tmp.fileno())
        Path(temporary_name).replace(path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
