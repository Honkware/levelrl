import copy
import json

import pytest

from maia_calibrate import (
    SCHEMA_VERSION,
    CalibrationMap,
    Candidate,
    MapValidationError,
    Setting,
    UnsupportedSchemaVersion,
    UnsupportedTargetError,
    load_map,
    nearest_measured,
    save_map,
    select,
)


def measured_rows():
    return [
        {
            "target": 1100,
            "method": "temperature",
            "dial": 1100,
            "temp": 1.25,
            "measured_elo": 1078.0,
            "elo_ci": [1040.0, 1110.0],
            "elo_ci_units": [1000.0, 1140.0],
            "strength_in_band": True,
            "validated": True,
        },
        {
            "target": 1300,
            "method": "remap",
            "dial": 1200,
            "temp": 0.8,
            "measured_elo": 1310.0,
            "elo_ci": [1270.0, 1350.0],
            "elo_ci_units": [1200.0, 1400.0],
            "strength_in_band": True,
            "validated": True,
        },
    ]


def test_exact_1100_setting_survives_json_round_trip(tmp_path):
    original = measured_rows()
    calibration = CalibrationMap.from_rows(copy.deepcopy(original))
    path = tmp_path / "maps" / "calibration_map.json"

    save_map(calibration, path)
    restored = load_map(path)
    exact = restored.lookup(1100)

    assert (exact.method, exact.dial, exact.temp) == ("temperature", 1100, 1.25)
    assert exact.validated is True and exact.nearest is False
    assert exact.nearest_target == 1100
    assert restored.to_dict()["validated"] == original
    assert json.loads(path.read_text())["schema_version"] == SCHEMA_VERSION


def test_nearest_fallback_is_explicit_and_never_validated():
    calibration = CalibrationMap.from_rows(measured_rows())

    resolved = nearest_measured(calibration, 1200)

    assert (resolved.method, resolved.dial, resolved.temp) == ("temperature", 1100, 1.25)
    assert resolved.target == 1200 and resolved.nearest_target == 1100
    assert resolved.nearest is True and resolved.validated is False
    assert resolved.interpolated is False and resolved.extrapolated is False
    assert calibration.lookup(1100).setting is calibration.measurements[0].setting


def test_nearest_fallback_marks_extrapolation_inside_supported_domain():
    calibration = CalibrationMap.from_rows(measured_rows())

    resolved = calibration.lookup(1000)

    assert resolved.nearest_target == 1100
    assert resolved.validated is False and resolved.extrapolated is True


@pytest.mark.parametrize("target", [599, 2601, 1100.0, True, "1100"])
def test_unsupported_targets_raise_explicit_errors(target):
    calibration = CalibrationMap.from_rows(measured_rows())

    with pytest.raises(UnsupportedTargetError):
        calibration.lookup(target)


def test_empty_map_does_not_invent_a_setting():
    with pytest.raises(UnsupportedTargetError, match="empty"):
        CalibrationMap().lookup(1100)


def test_selection_reuses_fidelity_rule_and_nearest_is_not_selection():
    rows = [
        Candidate("closest", 1100, 1.0, 1099, 1.5),
        Candidate("fidelity", 1100, 1.25, 1120, 1.2),
        Candidate("outside", 1000, 1.0, 1000, 1.0),
    ]

    assert select(rows, 1100).method == "fidelity"


def test_legacy_map_loads_without_changing_measured_values():
    payload = CalibrationMap.from_rows(measured_rows()).to_dict()
    del payload["schema_version"]
    payload["settings"] = [
        {key: value for key, value in row.items() if key not in ("nearest", "validated")}
        for row in payload["settings"]
    ]

    restored = CalibrationMap.from_dict(payload)

    assert restored.to_dict()["validated"] == measured_rows()


@pytest.mark.parametrize(
    "change",
    [
        lambda row: row.update(temp=0),
        lambda row: row.update(temp=float("nan")),
        lambda row: row.update(dial=-1),
        lambda row: row.update(dial=5001),
        lambda row: row.update(strength_in_band=False),
        lambda row: row.update(validated=False),
        lambda row: row.update(elo_ci=[1100, 1000]),
    ],
)
def test_invalid_validated_entries_are_rejected(change):
    rows = measured_rows()
    change(rows[0])

    with pytest.raises(MapValidationError):
        CalibrationMap.from_rows(rows)


def test_duplicate_targets_and_inconsistent_settings_are_rejected():
    rows = measured_rows()
    payload = CalibrationMap.from_rows(rows).to_dict()
    payload["validated"].append(copy.deepcopy(payload["validated"][0]))
    with pytest.raises(MapValidationError, match="unique"):
        CalibrationMap.from_dict(payload)

    payload = CalibrationMap.from_rows(rows).to_dict()
    payload["settings"][0]["dial"] = 999
    with pytest.raises(MapValidationError, match="inconsistent"):
        CalibrationMap.from_dict(payload)


def test_unsupported_schema_version_is_explicit():
    payload = CalibrationMap.from_rows(measured_rows()).to_dict()
    payload["schema_version"] = SCHEMA_VERSION + 1

    with pytest.raises(UnsupportedSchemaVersion):
        CalibrationMap.from_dict(payload)


@pytest.mark.parametrize("dial,temp", [(-1, 1.0), (5001, 1.0), (1100, 0.0), (1100, -0.1)])
def test_setting_validates_model_ranges(dial, temp):
    with pytest.raises(MapValidationError):
        Setting("temperature", dial, temp)
