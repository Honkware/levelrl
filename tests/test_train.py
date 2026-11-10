from levelrl.train import Trainer, pull


def game(score, judge, plies=4):
    return {"score": score, "plies": plies, "rec": [(None, None, 0, 0.0, 0.0, judge, k) for k in range(0, plies, 2)]}


def trainer(sign, goal=1500.0):
    t = Trainer.__new__(Trainer)
    t.goal, t.st = goal, {"sign": sign, "c": 0.0, "nu": 0.0, "bias": 0.0}
    return t


def test_the_advantage_carries_the_outcome_only_and_the_judge_score_cannot_move_it():
    lucky, plain = [game(1.0, -9.0), game(0.0, -1.0)], [game(1.0, -1.0), game(0.0, -1.0)]
    a, _ = trainer(1.0).advantages(lucky)
    b, _ = trainer(1.0).advantages(plain)
    assert [x for _, x in a] == [x for _, x in b] and a[0][1] > 0 > a[-1][1]


def test_a_negative_weight_rewards_losing_and_a_larger_one_pushes_harder():
    assert pull(1500, 1200) == 1.0 and pull(1500, 1800) == -1.0 and abs(pull(1500, 1530) + 0.2) < 1e-9
    grp = [game(1.0, -2.0), game(0.0, -2.0)]
    up, _ = trainer(3.0).advantages(grp)
    down, _ = trainer(-0.5).advantages(grp)
    assert up[0][1] > 0 > down[0][1] and abs(up[0][1]) == 6 * abs(down[0][1])


def steered(scores, goal=1500.0, bias=0.0, rounds=1):
    from collections import deque
    from levelrl import units
    t = trainer(1.0, goal)
    t.st["bias"] = bias
    t.units, t.elo = units.Units(0.0, 1.0, {"r1500": 1500.0}), {"r1500": 1500.0}
    t.recent = deque(({"opp": "r1500", "score": s} for s in scores), maxlen=256)
    for _ in range(rounds):
        t.steer()
    return t.st


def test_the_weight_keeps_growing_while_the_rating_stays_short_of_the_goal():
    # Scoring 25% against a 1500 rung reads about 1310. A fixed weight stalled there in the second run.
    short = [1.0 if i % 4 == 0 else 0.0 for i in range(100)]
    assert steered(short[:10])["sign"] == 1.0  # too few games, the starting weight stands
    once, later = steered(short), steered(short, rounds=20)
    assert once["sign"] > 1.0 and later["sign"] > once["sign"] + 2 and abs(later["nu"] - 20 * 0.15) < 1e-9


def test_the_weight_reverses_above_the_goal_and_rests_on_it():
    strong = [0.0 if i % 4 == 0 else 1.0 for i in range(100)]
    assert steered(strong)["sign"] < -1.0
    even = [float(i % 2) for i in range(100)]
    assert abs(steered(even)["sign"]) < 0.05


def test_the_correction_from_the_last_eval_shifts_the_rating_the_controller_sees():
    even = [float(i % 2) for i in range(100)]
    st = steered(even, bias=60.0)
    assert abs(st["est"] - st["est_raw"] - 60.0) < 1e-9 and st["sign"] < 0


def test_training_progress_can_be_reported_before_the_first_update(monkeypatch):
    t = trainer(1.0)
    t.name, t.T, t.games = "smoke", 1500, 96
    t.st.update(games=0, update=0)
    calls = []
    monkeypatch.setattr("levelrl.train.progress.tick", lambda *args, **kwargs: calls.append((args, kwargs)))
    t.report("update 0 · collecting rollouts", force=True)
    args, kwargs = calls[0]
    assert args == ("smoke", 0, 96, "games")
    assert kwargs["force"] is True
    assert kwargs["stats"]["dial"] == 1500 and kwargs["stats"]["goal"] == "1500"


def test_latest_evaluation_metrics_survive_later_training_reports(monkeypatch):
    t = trainer(1.0)
    t.name, t.T, t.games = "smoke", 1500, 3000
    t.st.update(games=800, update=25)
    calls = []
    monkeypatch.setattr("levelrl.train.progress.tick", lambda *args, **kwargs: calls.append(kwargs))
    t.report("checkpoint 25", stats={"elo": "1510", "top1": "52.4%"})
    t.report("update 26", stats={"judge": "0.50"})
    assert calls[-1]["stats"]["elo"] == "1510"
    assert calls[-1]["stats"]["top1"] == "52.4%"
    resumed = trainer(1.0)
    resumed.name, resumed.T, resumed.games, resumed.st = t.name, t.T, t.games, t.st
    resumed.report("update 27")
    assert calls[-1]["stats"]["elo"] == "1510" and calls[-1]["stats"]["top1"] == "52.4%"


def test_training_stops_once_the_rating_has_held_the_band_for_a_run_of_updates():
    from levelrl.train import HOLD
    t = trainer(1.0, goal=1500.0)
    t.st["est"] = 1530.0
    assert [t.settled() for _ in range(HOLD)] == [False] * (HOLD - 1) + [True]
    t.st["est"] = 1570.0  # one update outside the band starts the count again
    assert not t.settled() and t.st["streak"] == 0
