from levelrl.train import Trainer, pull


def game(score, judge, plies=4):
    return {"score": score, "plies": plies, "rec": [(None, None, 0, 0.0, 0.0, judge, k) for k in range(0, plies, 2)]}


def trainer(w_out, w_fid, sign):
    t = Trainer.__new__(Trainer)
    t.w_out, t.w_fid, t.st = w_out, w_fid, {"sign": sign, "c": 0.0}
    return t


def test_an_even_weighting_gives_the_judge_as_much_say_as_the_outcome():
    # The win has the worse judge score. In raw units the outcome would decide the sign of every advantage.
    grp = [game(1.0, -2.05), game(0.0, -2.00)]
    adv, _ = trainer(0.5, 0.5, 1.0).advantages(grp)
    assert max(abs(a) for _, a in adv) < 0.02
    adv, _ = trainer(0.25, 0.75, 1.0).advantages(grp)
    assert adv[0][1] < 0 < adv[-1][1]


def test_the_outcome_pull_reverses_above_the_band_and_fades_inside_it():
    assert pull(1500, 1200) == 1.0 and pull(1500, 1800) == -1.0 and abs(pull(1500, 1530) + 0.2) < 1e-9
    grp = [game(1.0, -2.0), game(0.0, -2.0)]
    up, _ = trainer(1.0, 0.0, 1.0).advantages(grp)
    down, _ = trainer(1.0, 0.0, -0.5).advantages(grp)
    assert up[0][1] > 0 > down[0][1] and abs(down[0][1]) < abs(up[0][1])
