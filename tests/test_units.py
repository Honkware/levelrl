import random

from levelrl import units


def test_the_affine_fit_recovers_a_compressed_scale_and_prefers_it():
    rng = random.Random(0)
    pts = [(x, 20.0, 300 + 1.6 * x + rng.gauss(0, 20), 45.0) for x in range(600, 2100, 150)]
    f = units.fit(pts, n=500)
    assert f["use"] == "affine"
    assert abs(f["affine"]["b"] - 1.6) < 0.15


def test_rung_choice_between_neighbours_matches_the_target_on_average():
    u = units.Units(0.0, 1.0, {"lo": 1000.0, "hi": 1200.0, "top": 1400.0})
    rng = random.Random(0)
    picks = [u.rung_near(1050.0, rng) for _ in range(4000)]
    assert abs(picks.count("hi") / 4000 - 0.25) < 0.03
    assert u.rung_near(500.0, rng) == "lo" and u.rung_near(9000.0, rng) == "top"


def test_the_interval_with_units_is_wider_than_the_gauntlet_alone_and_widens_away_from_the_anchors():
    rng = random.Random(1)
    pts = [(x, 20.0, 300 + 1.6 * x + rng.gauss(0, 20), 45.0) for x in range(1000, 1900, 100)]
    f = units.fit(pts, n=800)
    m = f[f["use"]]
    u = units.Units(m["a"], m["b"], {}, draws=m["draws"])
    width = lambda x: (lambda c: c[1] - c[0])(u.ci(x, [x - 30, x + 30]))
    assert width(1450) > m["b"] * 60 and width(2400) > width(1450)
