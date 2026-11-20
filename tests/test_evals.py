import torch

from levelrl import evals, maia


def test_score_reuses_cached_features_without_reencoding_rows(monkeypatch):
    cfg = type("Cfg", (), {"device": "cpu", "history": 4, "include_time_info": False})()
    rows = [
        {"moves": [], "move": "e2e4", "elo": 1500, "oppo_elo": 1500},
        {"moves": ["e2e4"], "move": "e7e5", "elo": 1500, "oppo_elo": 1500},
    ]

    def fake_run(model, x, mask, se, oe, dev):
        return torch.zeros((len(x), len(maia.MOVES))), None

    monkeypatch.setattr(evals, "_features", {})
    monkeypatch.setattr(evals.maia, "run", fake_run)
    first = evals.score(cfg, object(), rows, bsz=1)
    assert len(first["hit"]) == len(first["nll"]) == 2
    assert len(evals._features) == 1

    def no_reencode(cfg, moves):
        raise AssertionError("features were not cached")

    monkeypatch.setattr(evals.maia, "replay_features", no_reencode)
    assert evals.score(cfg, object(), rows, bsz=1) == first


def test_paired_ci_is_deterministic_and_preserves_a_constant_difference():
    assert evals.paired_ci([1.0] * 100, [0.0] * 100) == [1.0, 1.0]
    a = [0.0, 1.0, 2.0, 3.0]
    b = [0.0, 0.5, 1.0, 1.5]
    assert evals.paired_ci(a, b, n=200) == evals.paired_ci(a, b, n=200)


def test_humanlike_streams_position_progress_without_changing_its_results(monkeypatch):
    class Pool:
        def imap(self, fn, moves, chunksize):
            assert chunksize == 16
            return iter({"phase": "opening", "loss": 0, "kind": None} for _ in moves)

    monkeypatch.setattr(evals.arena, "pool", lambda: Pool())
    seen = []
    result = evals.humanlike([object()] * 65, on_progress=seen.append)
    assert seen == [64, 65]
    assert result["n"] == 65 and result["by_phase"]["opening"]["n"] == 65


def test_probe_sets_include_custom_human_bands(monkeypatch):
    rows = [{"move": "e2e4"}] * (evals.BAND_N + evals.VAL_N + evals.FID_N)
    monkeypatch.setattr(evals, "rows", lambda name: rows)
    sets = evals.probe_sets(700, bands=(700,))
    assert "val_T" in sets and "fid_T" in sets and "band_700" in sets
    assert len(sets["val_T"]) == evals.VAL_N and len(sets["fid_T"]) == evals.FID_N
