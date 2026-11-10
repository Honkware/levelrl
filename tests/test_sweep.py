import time

from levelrl import pipeline


def test_two_independent_arms_run_concurrently_and_save_in_order(monkeypatch, tmp_path):
    intervals = []

    class Ctx:
        cfg = type("Cfg", (), {"device": "cpu"})()
        base = object()
        train_ops = []
        units = object()

    class Trainer:
        def __init__(self, cfg, base, out, T, goal, start, units, games, seed, name):
            self.name, self.goal, self.out = name, goal, out
            self.st = {"games": games}

        def run(self, openings, on_eval):
            start = time.perf_counter()
            time.sleep(0.05)
            intervals.append((self.name, start, time.perf_counter()))

    monkeypatch.setattr(pipeline, "Trainer", Trainer)
    monkeypatch.setattr(pipeline, "make_eval", lambda ctx, T: lambda tr, tag: {"tag": tag})
    monkeypatch.setattr(pipeline, "GOALS", (0.5, 1.0))
    monkeypatch.setattr(pipeline, "load", lambda name: {"targets": [1500], "curve": [{"dial": 1500, "lichess": 1600.0}]})
    monkeypatch.setattr(pipeline, "done", lambda name: False)
    monkeypatch.setattr(pipeline, "RUN", tmp_path)
    (tmp_path / "games").mkdir()
    (tmp_path / "games" / "base_1500.json").write_text("[]")
    monkeypatch.setattr(pipeline.evals, "moves_of", lambda games, limit: [])
    monkeypatch.setattr(pipeline.evals, "human_moves", lambda T, n: [])
    monkeypatch.setattr(pipeline.evals, "humanlike", lambda moves, on_progress=None: {})
    monkeypatch.setattr(pipeline, "say", lambda text: None)
    saved, ticks = [], []
    monkeypatch.setattr(pipeline, "save", lambda name, obj: saved.append(name))
    monkeypatch.setattr(pipeline.progress, "tick", lambda *args, **kwargs: ticks.append((args, kwargs)))
    monkeypatch.setattr(pipeline, "finish_arm", lambda ctx, T, tr, frac, start: {
        "final": {"elo": 1500, "elo_ci": [1400, 1600], "fidelity": {"top1": 50.0}}, "held": {"in_band": 1.0}})

    result = pipeline.st_sweep(Ctx())

    assert len(intervals) == 2
    first, second = sorted(intervals, key=lambda item: item[1])
    assert second[1] < first[2], intervals
    assert result["arms"] == ["T1500_goal50", "T1500_goal100"]
    assert saved == ["arm_T1500_goal50", "arm_T1500_goal100"]
    comparison = [args for args, _ in ticks if args[0] == "comparison"]
    assert comparison[0][:3] == ("comparison", 0, 0)
    assert comparison[-1][:3] == ("comparison", 0, 0)


def test_evaluation_progress_does_not_replace_the_training_job(monkeypatch, tmp_path):
    class Units:
        def closest(self, T, n):
            return ["rung"]

        def lich(self, value):
            return value

    class Ctx:
        cfg = type("Cfg", (), {"device": "cpu"})()
        units = Units()
        suite = []
        elo = {}
        base_scores = lambda self, T: {}

        def rated(self, games):
            return {"ladder": 1500.0, "ladder_ci": [1400.0, 1600.0]}

    reports = []

    class Trainer:
        name, goal = "T1500_test", 1500.0
        ema = object()
        out = tmp_path
        st = {"log": [], "sign": 1.0, "update": 25, "games": 800, "bias": 0.0}

        def report(self, note, stats=None, force=False):
            reports.append((note, stats, force))

    seen = {}
    monkeypatch.setattr(pipeline.gauntlet, "play", lambda *args, **kwargs: seen.update(name=kwargs["name"]) or [])
    monkeypatch.setattr(pipeline.evals, "probe", lambda *args, **kwargs: ({"band_1500": {"top1": 50.0, "sce": 1.0}}, {}))
    monkeypatch.setattr(pipeline.evals, "threefold", lambda games: 0.0)
    monkeypatch.setattr(pipeline.evals, "book_overlap", lambda games: 0.0)
    monkeypatch.setattr(pipeline, "say", lambda text: None)
    pipeline.make_eval(Ctx(), 1500)(Trainer(), "25")
    assert seen["name"] == "T1500_test:eval"
    assert reports[-1][0] == "checkpoint 25 · Elo 1500 · top-1 50.0%"
    assert reports[-1][1] == {"elo": "1500", "top1": "50.0%"}


def test_sweep_progress_is_monotonic_across_arms(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline.progress, "tick", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(pipeline, "GOALS", (0.5, 1.0))
    tracker = pipeline.SweepProgress([1500])

    class Trainer:
        def __init__(self, name, games):
            self.name, self.st = name, {"games": games}

    tracker.add(Trainer("a", 800))
    tracker.complete("a")
    tracker.add(Trainer("c", 400))
    args, _ = calls[-1]
    assert args[:3] == ("sweep", pipeline.ARM_GAMES + 400, 2 * pipeline.ARM_GAMES)
    tracker.close()


def test_final_checks_report_one_accumulating_total(monkeypatch, tmp_path):
    reports = []

    class Trainer:
        name, out, goal = "T1500_a", tmp_path, 1500.0
        st = {"evals": [{"tag": "25", "elo": 1400}, {"tag": "50", "elo": 1600}],
              "log": [], "recoveries": 0, "sign": 1.0, "bias": 0.0}

        def report(self, note, stats=None, force=False):
            reports.append((note, stats))

    class Units:
        def closest(self, T, n):
            return ["rung"] * n

        def lich(self, value):
            return value

        def ci(self, value, ci):
            return ci

    class Ctx:
        cfg = type("Cfg", (), {"device": "cpu"})()
        units, suite, elo = Units(), [[], []], {}
        base_scores = lambda self, T: {}
        rated = lambda self, games: {"ladder": 1500.0, "ladder_ci": [1400.0, 1600.0]}

    class Model:
        def eval(self):
            return self

        def load_state_dict(self, state):
            pass

        def state_dict(self):
            return {}

    monkeypatch.setattr(pipeline.maia, "load", lambda device: (None, Model()))
    monkeypatch.setattr(pipeline.torch, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr(pipeline.torch, "save", lambda *args: None)
    monkeypatch.setattr(pipeline.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(pipeline.gauntlet, "play", lambda *args, **kwargs: [])
    monkeypatch.setattr(pipeline.gauntlet, "run", lambda *args, **kwargs: [])
    monkeypatch.setattr(pipeline.evals, "probe", lambda *args, **kwargs: ({"fid_T": {"top1": 50.0}}, {}))
    monkeypatch.setattr(pipeline.evals, "moves_of", lambda games, limit: [])
    monkeypatch.setattr(pipeline.evals, "humanlike", lambda moves, **kw: {})

    result = pipeline.finish_arm(Ctx(), 1500, Trainer(), 1.0, 1600.0)
    final = [stats for _, stats in reports if stats.get("final_total")]
    assert [stats["final_before"] for stats in final] == [0, 8, 28]
    assert all(stats["final_total"] == 28 and stats["final_steps"] == 2 for stats in final)
    assert result["band_pick"] is None  # neither eval measured inside the band
    assert final[-1]["final_active"] is False
    assert result["final"]["fidelity"]["top1"] == 50.0


def test_the_setting_that_plays_a_target_is_read_off_a_noisy_curve_made_monotone():
    setting, inside = pipeline.crossing([(900, 1050), (1000, 1120), (1100, 1100), (1200, 1300)], 1100)
    assert inside and abs(setting - 1025) < 1e-9
    assert pipeline.crossing([(900, 1200), (1000, 1300)], 1100) == (900, False)  # the grid never plays that weak
