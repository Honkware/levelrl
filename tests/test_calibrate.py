import json
from types import SimpleNamespace

from levelrl import calibrate


def test_candidates_include_a_joint_dial_and_temperature_setting():
    rows = calibrate.candidates(1100, [(900, 1000), (1100, 1300), (1300, 1400)],
                              [(0.5, 1400), (1.0, 1300), (1.5, 1000)])
    assert [r["method"] for r in rows] == ["dial as asked", "remap", "temperature", "combined"]
    assert rows[1]["dial"] == 970
    assert rows[2]["temp"] > 1.0
    assert rows[3]["dial"] != 1100 and rows[3]["temp"] != 1.0


def test_selection_prefers_validation_fidelity_inside_the_rating_band():
    rows = [
        {"method": "closest", "validation_elo": 1099, "validation_sce": 1.5},
        {"method": "fidelity", "validation_elo": 1120, "validation_sce": 1.2},
        {"method": "outside", "validation_elo": 1000, "validation_sce": 1.0},
    ]
    assert calibrate.choose(rows, 1100)["method"] == "fidelity"
    assert {r["method"] for r in calibrate.pareto(rows, 1100)} == {"closest", "fidelity", "outside"}


def test_mapping_preserves_measured_settings_and_uses_nearest_only_for_unknown_targets():
    rows = [{"target": 1000, "method": "temperature", "dial": 900, "temp": 1.4, "elo": 990,
             "elo_ci": [950, 1030], "elo_ci_units": [900, 1080]},
            {"target": 1500, "method": "remap", "dial": 1200, "temp": 0.8, "elo": 1510,
             "elo_ci": [1470, 1550], "elo_ci_units": [1400, 1600]},
            {"target": 2000, "method": "combined", "dial": 1500, "temp": 1.0, "elo": 1980,
             "elo_ci": [1940, 2020], "elo_ci_units": [1900, 2080]}]
    fit = calibrate.mapping(rows)
    assert [(r["dial"], r["temp"]) for r in fit] == [(900, 1.4), (1200, 0.8), (1500, 1.0)]
    assert all(r["validated"] and r["strength_in_band"] for r in fit)
    mid = calibrate.setting_at(fit, 1250)
    assert mid["nearest_target"] == 1000 and mid["dial"] == 900 and mid["interpolated"] is False
    assert calibrate.setting_at(fit, 500)["extrapolated"] is True


def test_validation_reference_does_not_read_the_final_fidelity_set(monkeypatch):
    class Ctx:
        cfg = None
        base = object()

        def base_scores(self, T):
            raise AssertionError("final test reference was read during validation")

    monkeypatch.setattr(calibrate.evals, "probe_sets", lambda T, bands=None: {"val_T": [object()]})
    monkeypatch.setattr(calibrate.evals, "score", lambda *args, **kwargs: {"hit": [1, 0], "nll": [1.0, 2.0]})
    monkeypatch.setattr(calibrate.evals, "paired_ci", lambda a, b: [0.0, 0.0])
    calibrate._VALIDATION_REF.clear()
    out = calibrate._fid(Ctx(), 1100, 1000, 1.2, final=False)
    assert out["n"] == 2 and out["top1"] == 50.0


def test_dry_run_covers_the_full_range_and_all_human_bands(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(calibrate.data, "ROOT", tmp_path)
    (tmp_path / "positions").mkdir()
    for band in (1100, 1300, 1500, 1700, 1900):
        (tmp_path / "positions" / f"fid_{band}.jsonl.gz").write_text("")
    args = SimpleNamespace(run=str(tmp_path / "run"), out=str(tmp_path / "out"), start=600, stop=2600, step=100,
                           dry_run=True)
    calibrate.run(args)
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["targets"]) == 21
    assert plan["human_fidelity_bands"] == [1100, 1300, 1500, 1700, 1900]
    assert plan["total_games"] == plan["scan_games"] + plan["validation_games"] + plan["final_games"]


def test_outputs_contain_a_usable_map_and_validation_frontier(tmp_path):
    state = {"targets": {"1100": {
        "selected": {"method": "temperature", "dial": 1100, "temp": 1.25, "reaches": True},
        "final": {"elo": 1078.0, "elo_ci": [1040.0, 1110.0], "elo_ci_units": [1000.0, 1140.0],
                  "fidelity": {"top1": 54.8, "sce": 1.45}},
        "candidates": [{"method": "temperature", "dial": 1100, "temp": 1.25, "validation_elo": 1090.0,
                        "validation_sce": 1.4}],
    }}}
    calibrate.write_outputs(state, tmp_path)
    report = (tmp_path / "calibration.md").read_text()
    mapping = json.loads((tmp_path / "calibration_map.json").read_text())
    assert "1078" in report and "exact measured settings only" in mapping["interpolation"]
    assert mapping["settings"] == [{"target": 1100, "method": "temperature", "dial": 1100, "temp": 1.25,
                                    "nearest_target": 1100, "interpolated": False, "extrapolated": False}]


def test_mocked_full_workflow_is_resumable_and_writes_final_results(monkeypatch, tmp_path):
    class Units:
        lich = lambda self, x: x
        ci = lambda self, x, interval: interval
        closest = lambda self, T, n: ["rung"]

    class Ctx:
        cfg, base, units, elo = None, object(), Units(), {}
        suite = [["a"], ["b"], ["c"]]

        def adaptive(self, model, policy, variants, name, label, seed, n_main, on_progress=None):
            values = {0.7: 1400, 1.0: 1300, 1.3: 1000}
            out = {k: [{"rating": values.get(k, {900: 1000, 1100: 1300, 1300: 1400}.get(k))}] for k in variants}
            if on_progress:
                for _ in variants:
                    on_progress()
            return out

        def rated(self, games):
            rating = games[0]["rating"]
            return {"ladder": rating, "ladder_ci": [rating - 20, rating + 20], "games": len(games), "score": 0.5}

    monkeypatch.setattr(calibrate.pipeline, "Ctx", Ctx)
    monkeypatch.setattr(calibrate, "DIAL_GRID", (900, 1100, 1300))
    monkeypatch.setattr(calibrate, "TEMP_GRID", (0.7, 1.0, 1.3))
    monkeypatch.setattr(calibrate, "VALIDATION_OPENINGS", 1)
    monkeypatch.setattr(calibrate.data, "BANDS", (1100,))
    monkeypatch.setattr(calibrate, "has_fidelity", lambda T: T == 1100)
    monkeypatch.setattr(calibrate.gauntlet, "rate", lambda games, elo: games[0]["rating"])
    plays = []

    def play(cfg, model, policy, *args, **kwargs):
        plays.append(policy)
        rating = 1100 if policy["temp"] != 1.0 else 1300 if policy["self_elo"] == 1100 else policy["self_elo"]
        return [{"rating": rating, "moves": [], "opening": [], "white": True, "score": 0.5}]

    monkeypatch.setattr(calibrate.gauntlet, "play", play)
    monkeypatch.setattr(calibrate, "_fid", lambda *args, **kwargs: {"top1": 55.0, "sce": 1.0})
    hl = {"n": 1, "blunder_rate": 0.0, "blunder_types": {"piece drop": 0.0, "tactical oversight": 0.0},
          "by_phase": {}}
    monkeypatch.setattr(calibrate.evals, "moves_of", lambda games, limit: [])
    monkeypatch.setattr(calibrate.evals, "humanlike", lambda *args, **kwargs: hl)
    monkeypatch.setattr(calibrate, "_reference", lambda ctx, T, state, run_dir=None: {"base": hl, "human": hl})
    monkeypatch.setattr(calibrate, "plot_outputs", lambda state, out: None)
    monkeypatch.setattr(calibrate.arena, "close_pool", lambda: None)
    monkeypatch.setattr(calibrate.progress, "tick", lambda *args, **kwargs: None)
    args = SimpleNamespace(run=str(tmp_path / "run"), out=str(tmp_path / "out"), start=1100, stop=1100, step=100,
                           dry_run=False)

    calibrate.run(args)
    state = json.loads((tmp_path / "run" / "calibration.partial.json").read_text())
    assert state["steps"] == 4
    assert state["targets"]["1100"]["selected"]["method"] == "temperature"
    assert state["targets"]["1100"]["final"]["elo"] == 1100
    assert (tmp_path / "out" / "calibration_map.png").exists() is False
    assert "Sampled-game human-likeness" in (tmp_path / "out" / "calibration.md").read_text()
    count = len(plays)
    calibrate.run(args)
    assert len(plays) == count
