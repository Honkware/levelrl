from levelrl.train import Trainer, pull


def game(score, judge, plies=4):
    return {"score": score, "plies": plies, "rec": [(None, None, 0, 0.0, 0.0, judge, k) for k in range(0, plies, 2)]}


def trainer(w_out, w_fid, sign):
    t = Trainer.__new__(Trainer)
    t.w_out, t.w_fid, t.st = w_out, w_fid, {"sign": sign, "c": 0.0}
    return t


def test_the_advantage_carries_the_outcome_only_and_the_judge_score_cannot_move_it():
    lucky, plain = [game(1.0, -9.0), game(0.0, -1.0)], [game(1.0, -1.0), game(0.0, -1.0)]
    a, _ = trainer(0.5, 0.5, 1.0).advantages(lucky)
    b, _ = trainer(0.5, 0.5, 1.0).advantages(plain)
    assert [x for _, x in a] == [x for _, x in b] and a[0][1] > 0 > a[-1][1]


def test_the_running_rating_steers_the_pull_toward_the_band():
    from collections import deque
    from levelrl import units
    t = trainer(1.0, 0.0, 1.0)
    t.T, t.units = 1500, units.Units(0.0, 1.0, {"r1500": 1500.0})
    t.elo, t.recent = {"r1500": 1500.0}, deque(maxlen=256)
    t.steer()
    assert t.st["sign"] == 1.0  # too few games, the starting pull stands
    t.recent.extend({"opp": "r1500", "score": 1.0 if i % 4 else 0.0} for i in range(100))  # scoring 75%: too strong
    t.steer()
    assert t.st["sign"] == -1.0 and t.st["est"] > 1650
    t.recent.clear()
    t.recent.extend({"opp": "r1500", "score": float(i % 2)} for i in range(100))  # 50%: on the band
    t.steer()
    assert abs(t.st["sign"]) < 0.05


def test_the_outcome_pull_reverses_above_the_band_and_fades_inside_it():
    assert pull(1500, 1200) == 1.0 and pull(1500, 1800) == -1.0 and abs(pull(1500, 1530) + 0.2) < 1e-9
    grp = [game(1.0, -2.0), game(0.0, -2.0)]
    up, _ = trainer(1.0, 0.0, 1.0).advantages(grp)
    down, _ = trainer(1.0, 0.0, -0.5).advantages(grp)
    assert up[0][1] > 0 > down[0][1] and abs(down[0][1]) < abs(up[0][1])


def test_training_progress_can_be_reported_before_the_first_update(monkeypatch):
    t = trainer(0.5, 0.5, 1.0)
    t.name, t.T, t.games = "smoke", 1500, 96
    t.st.update(games=0, update=0)
    calls = []
    monkeypatch.setattr("levelrl.train.progress.tick", lambda *args, **kwargs: calls.append((args, kwargs)))
    t.report("update 0 · collecting rollouts", force=True)
    args, kwargs = calls[0]
    assert args == ("smoke", 0, 96, "games")
    assert kwargs["force"] is True
    assert kwargs["stats"]["T"] == 1500


def test_latest_evaluation_metrics_survive_later_training_reports(monkeypatch):
    t = trainer(0.5, 0.5, 1.0)
    t.name, t.T, t.games = "smoke", 1500, 3000
    t.st.update(games=800, update=25)
    calls = []
    monkeypatch.setattr("levelrl.train.progress.tick", lambda *args, **kwargs: calls.append(kwargs))
    t.report("checkpoint 25", stats={"elo": "1510", "top1": "52.4%"})
    t.report("update 26", stats={"judge": "0.50"})
    assert calls[-1]["stats"]["elo"] == "1510"
    assert calls[-1]["stats"]["top1"] == "52.4%"
    resumed = trainer(0.5, 0.5, 1.0)
    resumed.name, resumed.T, resumed.games, resumed.st = t.name, t.T, t.games, t.st
    resumed.report("update 27")
    assert calls[-1]["stats"]["elo"] == "1510" and calls[-1]["stats"]["top1"] == "52.4%"
