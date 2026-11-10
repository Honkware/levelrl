import random

from levelrl import ladder


def test_the_fit_recovers_known_ratings_from_simulated_games():
    true = {"a": 1100.0, "b": 1300.0, "c": 1500.0, "d": 1800.0}
    rng = random.Random(0)
    games = []
    for x in true:
        for y in true:
            if x != y:
                p = 1 / (1 + 10 ** ((true[y] - true[x]) / 400))
                games += [{"white": x, "black": y, "score": float(rng.random() < p)} for _ in range(400)]
    elo = ladder.fit(games, list(true), pin=("c", 1500.0))
    for k, v in true.items():
        assert abs(elo[k] - v) < 40, (k, elo[k])


def test_a_rung_that_never_scores_still_gets_a_finite_rating():
    games = [{"white": "a", "black": "b", "score": 1.0}] * 20 + [{"white": "b", "black": "a", "score": 0.0}] * 20
    elo = ladder.fit(games, ["a", "b"], pin=("a", 1500.0))
    assert elo["b"] < elo["a"] and elo["b"] > -10_000


def test_close_engines_quits_and_closes_every_rung(monkeypatch):
    class Engine:
        def __init__(self):
            self.calls = []

        def quit(self):
            self.calls.append("quit")

        def close(self):
            self.calls.append("close")

    engines = {"sk0": ladder.Rung("sk0", Engine()), "elo1500": ladder.Rung("elo1500", Engine())}
    calls = [r.eng.calls for r in engines.values()]
    monkeypatch.setattr(ladder, "_engines", engines)
    ladder.close_engines()
    assert calls == [["quit", "close"], ["quit", "close"]]
    assert engines == {}
