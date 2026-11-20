from __future__ import annotations

from typing import Any


class ChessEvalError(ValueError):
    code = "chess_eval_error"

    def __init__(self, message: str, **context: Any) -> None:
        self.message = message
        self.context = context
        super().__init__(message)

    def __str__(self) -> str:
        if not self.context:
            return self.message
        details = ", ".join(f"{key}={value!r}" for key, value in self.context.items())
        return f"{self.message} ({details})"

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": dict(self.context)}


class ValidationError(ChessEvalError):
    code = "validation_error"


class InvalidPositionError(ValidationError):
    code = "invalid_position"


class InvalidPredictionError(ValidationError):
    code = "invalid_prediction"


class MissingDataError(ChessEvalError):
    code = "missing_data"


class DataError(ChessEvalError):
    code = "data_error"


class UnsupportedMetricError(ChessEvalError):
    code = "unsupported_metric"
