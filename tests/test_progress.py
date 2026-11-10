import json

from levelrl import progress


def test_a_counter_restart_resets_elapsed_events_and_stale_stats(monkeypatch, tmp_path):
    times = iter([100.0, 160.0, 300.0])
    monkeypatch.setattr(progress, "ROOT", tmp_path)
    monkeypatch.setattr(progress, "_jobs", {})
    monkeypatch.setattr(progress, "_last", {})
    monkeypatch.setattr(progress.time, "time", lambda: next(times))

    progress.tick("base", 48, 48, "games")
    progress.tick("base", 200, 200, "games", stats={"score": "50%"})
    progress.tick("base", 0, 200, "games", force=True)

    j = json.loads((tmp_path / "base.json").read_text())
    assert j["start"] == 300.0
    assert j["done"] == 0
    assert j["events"] == []
    assert "stats" not in j


def test_progress_within_one_run_keeps_its_start(monkeypatch, tmp_path):
    times = iter([100.0, 120.0])
    monkeypatch.setattr(progress, "ROOT", tmp_path)
    monkeypatch.setattr(progress, "_jobs", {})
    monkeypatch.setattr(progress, "_last", {})
    monkeypatch.setattr(progress.time, "time", lambda: next(times))

    progress.tick("base", 0, 200, "games", force=True)
    progress.tick("base", 20, 200, "games")

    j = json.loads((tmp_path / "base.json").read_text())
    assert j["start"] == 100.0
    assert j["t"] == 120.0


def test_first_tick_preserves_an_event_that_initialized_the_job(monkeypatch, tmp_path):
    times = iter([100.0, 101.0, 102.0])
    monkeypatch.setattr(progress, "ROOT", tmp_path)
    monkeypatch.setattr(progress, "_jobs", {})
    monkeypatch.setattr(progress, "_last", {})
    monkeypatch.setattr(progress.time, "time", lambda: next(times))

    progress.event("ladder", "first game")
    progress.tick("ladder", 1, 200, "games")

    j = json.loads((tmp_path / "ladder.json").read_text())
    assert j["start"] == 100.0
    assert j["events"] == [[101.0, "first game"]]
