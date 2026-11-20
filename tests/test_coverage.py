import json
from types import SimpleNamespace

from levelrl import calibrate, coverage


def test_local_repairs_are_small_and_include_cold_high_end_candidates():
    low = coverage.local_candidates({"dial": 800, "temp": 1.1}, 800)
    high = coverage.local_candidates({"dial": 2500, "temp": 0.55}, 2500)
    assert len(low) == 8
    assert len(high) == 10
    assert {r["temp"] for r in high} >= {0.05, 0.1}


def test_band_audit_requires_disjoint_validation_and_final_months(tmp_path, monkeypatch):
    monkeypatch.setattr(coverage.data, "ROOT", tmp_path)
    (tmp_path / "positions").mkdir()
    (tmp_path / "positions" / "fid_1100.jsonl.gz").write_text("")
    rows = []
    for i in range(coverage.REQUIRED_ROWS):
        validation = i < 6000
        rows.append({"site": f"v{i}" if validation else f"f{i}", "month": "2025-09" if validation else "2026-03"})
    monkeypatch.setattr(coverage.data, "load", lambda name: rows)
    coverage._band_audit.cache_clear()
    result = coverage.band_audit(1100)
    assert result["ready"] is True and result["rows"] == coverage.REQUIRED_ROWS


def test_missing_band_keeps_the_range_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(coverage.data, "ROOT", tmp_path)
    state = {"targets": {}}
    report = coverage.write_readiness(state, tmp_path)
    assert report["complete"] is False and report["complete_targets"] == 0
    assert "missing band" in (tmp_path / "coverage.md").read_text()


def test_every_target_requires_strength_fidelity_and_the_exact_map(tmp_path, monkeypatch):
    monkeypatch.setattr(coverage, "band_audit", lambda T: {"ready": True, "reasons": ["ready"]})
    state = {"targets": {}}
    for T in coverage.TARGETS:
        state["targets"][str(T)] = {
            "selected": {"method": "temperature", "dial": T, "temp": 1.25, "reaches": True},
            "final": {"elo": T + 10, "elo_ci": [T - 20, T + 40], "elo_ci_units": [T - 60, T + 70],
                      "fidelity": {"top1": 55.0, "sce": 1.4},
                      "humanlike": {"n": 10, "blunder_rate": 0.1,
                                    "blunder_types": {"piece drop": 0.5, "tactical oversight": 0.5}, "by_phase": {}},
                      "reference": {"note": "present"}},
            "candidates": [],
        }
    calibrate.write_outputs(state, tmp_path)
    report = coverage.readiness(state, tmp_path)
    assert report["complete"] is True and report["complete_targets"] == 21
    rows = json.loads((tmp_path / "calibration_map.json").read_text())["validated"]
    assert len(rows) == 21 and rows[10]["dial"] == 1600 and rows[10]["temp"] == 1.25


def test_repair_dry_run_does_not_load_the_model_or_play_games(tmp_path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    state = {"games": 100, "targets": {"800": {
        "selected": {"method": "remap", "dial": 650, "temp": 1.0, "reaches": True},
        "final": {"elo": 858.0},
    }}}
    (run / "calibration.partial.json").write_text(json.dumps(state))
    args = SimpleNamespace(run=str(run), out=str(tmp_path / "out"), targets=[800], dry_run=True)
    coverage.repair(args)
    plan = json.loads(capsys.readouterr().out)
    assert plan["targets"] == [800] and plan["candidates"] == {"800": 8}
    assert plan["estimated_new_games"] > 0


def test_repair_remaining_excludes_finished_work_and_counts_reference_games(tmp_path, monkeypatch):
    selected = {"method": "remap", "dial": 650, "temp": 1.0, "reaches": True}
    planned = {800: coverage.local_candidates(selected, 800)}
    monkeypatch.setattr(calibrate, "has_fidelity", lambda T: False)
    state = {"targets": {"800": {"selected": selected, "repair": {
        "candidates": [{"validation_elo": 800}] * len(planned[800]), "final_complete": True}}}}
    assert coverage.repair_remaining(state, [800], planned, tmp_path)["total"] == 0

    monkeypatch.setattr(calibrate, "has_fidelity", lambda T: T == 1100)
    monkeypatch.setattr(coverage.pipeline, "RUN", tmp_path / "empty")
    planned = {1100: []}
    state = {"references": {}, "targets": {"1100": {"repair": {"final_complete": True}}}}
    remaining = coverage.repair_remaining(state, [1100], planned, tmp_path)
    assert remaining["total"] == remaining["reference_games"] > 0


def test_repair_full_run_and_resume_do_not_duplicate_candidates_or_games(monkeypatch, tmp_path):
    class Units:
        lich = lambda self, x: x
        ci = lambda self, x, interval: interval
        closest = lambda self, T, n: ["rung"]

    class Ctx:
        cfg, base, units, elo = None, object(), Units(), {}
        suite = [["a"], ["b"], ["c"]]

        def rated(self, games):
            return {"ladder": 800.0, "ladder_ci": [780.0, 820.0], "games": len(games), "score": 0.5}

    run = tmp_path / "run"
    run.mkdir()
    selected = {"method": "remap", "dial": 650, "temp": 1.0, "reaches": True}
    state = {"games": 100, "references": {}, "targets": {"800": {
        "selected": selected, "final": {"elo": 858.0}, "candidates": []}}}
    (run / "calibration.partial.json").write_text(json.dumps(state))
    monkeypatch.setattr(coverage.pipeline, "Ctx", Ctx)
    monkeypatch.setattr(calibrate, "has_fidelity", lambda T: False)
    plays = []
    monkeypatch.setattr(coverage.gauntlet, "play", lambda *args, **kwargs: plays.append(1) or [
        {"moves": [], "opening": [], "white": True, "score": 0.5}])
    monkeypatch.setattr(coverage.calibrate, "plot_outputs", lambda state, out: None)
    monkeypatch.setattr(coverage.calibrate.arena, "close_pool", lambda: None)
    monkeypatch.setattr(coverage.progress, "tick", lambda *args, **kwargs: None)
    args = SimpleNamespace(run=str(run), out=str(tmp_path / "out"), targets=[800], dry_run=False)

    coverage.repair(args)
    saved = json.loads((run / "calibration.partial.json").read_text())
    assert saved["targets"]["800"]["repair"]["final_complete"] is True
    assert len(saved["targets"]["800"]["repair"]["candidates"]) == 8
    count = len(plays)
    coverage.repair(args)
    resumed = json.loads((run / "calibration.partial.json").read_text())
    assert len(plays) == count
    assert len(resumed["targets"]["800"]["repair"]["candidates"]) == 8


def test_finalize_dry_run_finds_missing_analysis_without_replaying_games(tmp_path, capsys, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    state = {"targets": {"1100": {"selected": {"method": "temperature", "dial": 1100, "temp": 1.25},
                                      "final": {"elo": 1100}}}}
    (run / "calibration.partial.json").write_text(json.dumps(state))
    monkeypatch.setattr(calibrate, "has_fidelity", lambda T: True)
    monkeypatch.setattr(coverage.pipeline, "RUN", tmp_path / "empty")
    args = SimpleNamespace(run=str(run), out=str(tmp_path / "out"), dry_run=True)
    coverage.finalize(args)
    plan = json.loads(capsys.readouterr().out)
    assert plan["targets"] == [1100] and plan["reference_games"] > 0
