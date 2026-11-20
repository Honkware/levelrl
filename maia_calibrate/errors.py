from __future__ import annotations


class CalibrationError(ValueError):
    pass


class MapValidationError(CalibrationError):
    pass


class UnsupportedSchemaVersion(CalibrationError):
    pass


class UnsupportedTargetError(CalibrationError):
    pass
