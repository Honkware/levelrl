from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .errors import ChessEvalError, DataError, MissingDataError, ValidationError

_MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def _validate_month(value: str, field: str) -> None:
    if not isinstance(value, str) or not _MONTH.fullmatch(value):
        raise ValidationError(f"{field} must use YYYY-MM format", **{field: value})


def check_month(month: str, cutoff: str) -> None:
    _validate_month(month, "month")
    _validate_month(cutoff, "cutoff")
    if month <= cutoff:
        raise DataError("dataset month is inside the training window", month=month, cutoff=cutoff)


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    cutoff: str
    months: tuple[str, ...]
    seed: int = 0
    datasets: Mapping[str, int] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_month(self.cutoff, "cutoff")
        if not isinstance(self.months, (tuple, list)):
            raise ValidationError("manifest months must be a sequence")
        if not self.months:
            raise MissingDataError("manifest must contain at least one month")
        for month in self.months:
            check_month(month, self.cutoff)
        if len(set(self.months)) != len(self.months):
            raise ValidationError("manifest months must be unique", months=self.months)
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValidationError("seed must be an integer", seed=self.seed)
        if not isinstance(self.datasets, Mapping) or not isinstance(self.metadata, Mapping):
            raise ValidationError("datasets and metadata must be mappings")
        for name, count in self.datasets.items():
            if not isinstance(name, str) or not name or not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValidationError("dataset counts must be non-negative integers", dataset=name, count=count)
        try:
            json.dumps(self.metadata)
        except (TypeError, ValueError) as error:
            raise ValidationError("manifest metadata must be JSON serializable") from error
        object.__setattr__(self, "months", tuple(sorted(self.months)))
        object.__setattr__(self, "datasets", MappingProxyType(dict(self.datasets)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def require_month(self, month: str) -> None:
        check_month(month, self.cutoff)
        if month not in self.months:
            raise MissingDataError("month is not covered by this manifest", month=month)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cutoff": self.cutoff,
            "months": list(self.months),
            "seed": self.seed,
            "datasets": dict(self.datasets),
            "metadata": dict(self.metadata),
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DatasetManifest:
        if not isinstance(value, Mapping):
            raise DataError("manifest must be a JSON object")
        fields = {"cutoff", "months", "seed", "datasets", "metadata"}
        missing = {"cutoff", "months"} - value.keys()
        if missing:
            raise MissingDataError("manifest is missing required fields", fields=sorted(missing))
        if unknown := value.keys() - fields:
            raise DataError("manifest contains unknown fields", fields=sorted(unknown))
        if not isinstance(value["months"], list):
            raise DataError("manifest months must be a JSON array")
        try:
            return cls(
                cutoff=value["cutoff"],
                months=tuple(value["months"]),
                seed=value.get("seed", 0),
                datasets=value.get("datasets", {}),
                metadata=value.get("metadata", {}),
            )
        except (TypeError, ValueError) as error:
            if isinstance(error, ChessEvalError):
                raise
            raise DataError("manifest has an invalid schema", detail=str(error)) from error

    @classmethod
    def load(cls, path: str | Path) -> DatasetManifest:
        source = Path(path)
        if not source.is_file():
            raise MissingDataError("manifest file does not exist", path=str(source))
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise DataError("could not read manifest", path=str(source), detail=str(error)) from error
        try:
            return cls.from_dict(value)
        except ChessEvalError as error:
            error.context.setdefault("path", str(source))
            raise

