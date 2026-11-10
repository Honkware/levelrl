import random

import torch

from levelrl import gauntlet


def test_batched_sampling_only_picks_moves_maia3s_nucleus_keeps():
    gen = torch.Generator().manual_seed(0)
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0, float("-inf")]] * 4000)
    probs = torch.softmax(logits[0] / 0.7, -1)
    sp, si = probs.sort(descending=True)
    keep = sp.cumsum(-1) <= 0.9
    keep[0] = True
    allowed = set(si[keep].tolist())
    got = gauntlet.sample(logits, 0.7, 0.9, gen).tolist()
    assert set(got) == allowed


def test_argmax_at_zero_temperature():
    gen = torch.Generator().manual_seed(0)
    assert gauntlet.sample(torch.tensor([[0.0, 5.0, 1.0]]), 0.0, 1.0, gen).tolist() == [1]


def test_rate_recovers_a_known_rating_against_fixed_rungs():
    elo = {"a": 1200.0, "b": 1500.0, "c": 1800.0}
    rng = random.Random(0)
    games = [{"opp": o, "score": float(rng.random() < 1 / (1 + 10 ** ((e - 1450) / 400)))}
             for o, e in elo.items() for _ in range(2000)]
    assert abs(gauntlet.rate(games, elo) - 1450) < 25
    rated, ci = gauntlet.rate_ci(games[::20], elo, n=100)
    assert abs(rated - 1450) < 60
    assert ci[0] < rated < ci[1]


def test_play_reports_zero_of_total_before_the_first_arena_step(monkeypatch):
    class EmptyArena:
        live = []

        def __init__(self, *args, **kwargs):
            pass

        def done(self):
            return True

    calls = []
    monkeypatch.setattr(gauntlet.arena, "Arena", EmptyArena)
    monkeypatch.setattr(gauntlet.progress, "tick", lambda *args, **kwargs: calls.append((args, kwargs)))
    cfg = type("Cfg", (), {"device": "cpu"})()
    policy = {"history": True, "temp": 1.0, "top_p": 1.0, "self_elo": 1500, "oppo_elo": 1500}
    assert gauntlet.play(cfg, object(), policy, ["elo1500"], [[]]) == []
    args, kwargs = calls[0]
    assert args == ("gauntlet", 0, 2, "games")
    assert kwargs["force"] is True
    assert "starting" in kwargs["note"]
