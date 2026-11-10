import json

from levelrl import report


def test_the_summary_is_written_from_run_files_and_lists_every_arm(tmp_path):
    run, res = tmp_path / "runs" / "pipeline", tmp_path / "results"
    run.mkdir(parents=True)
    hl = {"n": 100, "blunder_rate": 0.05, "blunder_types": {"piece drop": 0.4, "tactical oversight": 0.6},
          "by_phase": {"middlegame": {"n": 60, "mistake_rate": 0.1, "blunder_rate": 0.05, "mean_cp_loss": 40.0}}}
    fit = {"a": -759.0, "b": 1.5, "a_ci": [-900, -600], "b_ci": [1.3, 1.7], "resid": [10.0]}
    pr = {"calib": {"top1": 50.0, "sce": 2.0}, "band_1500": {"top1": 51.0, "sce": 1.9}}
    ev = lambda tag, u, elo: {"tag": tag, "update": u, "elo": elo, "elo_ci": [elo - 40, elo + 40], "probe": pr,
                              "entropy": 2.0, "kl_base": 0.01, "threefold": 0.02, "book": 0.5}
    arm = {"T": 1500, "goal": 1500.0, "start": 1650.0, "frac": 1.0, "weight": -2.5, "bias": 12.0, "games": 1600, "updates": 50, "stopped_early": 50,
           "held": {"updates": 30, "mean": 1512.0, "sd": 21.0, "in_band": 0.9}, "evals": [ev("25", 25, 1600.0), ev("final", 90, 1520.0)],
           "final": {"tag": "final", "elo": 1520.0, "elo_ci": [1480.0, 1560.0], "elo_ci_units": [1440.0, 1600.0],
                     "ladder": {"ladder": 1550.0},
                     "fidelity": {"top1": 51.2, "sce": 1.9, "top1_vs_base_ci": [-0.5, 0.3], "sce_vs_base_ci": [-0.01, 0.01]}},
           "sensitivity": {"1300": {"elo": 1400.0}, "1500": {"elo": 1520.0}, "1700": {"elo": 1650.0}},
           "humanlike": hl, "recoveries": 0}
    files = {"anchors": {"snapshot": "s.json", "fit": {"use": "affine", "affine": fit},
                         "dials": [{"dial": 600, "ladder": 1050.0, "lichess": 890}]},
             "base": {"curve": [{"dial": 1500, "ladder": 1600.0, "lichess": 1650.0, "lichess_ci": [1600.0, 1700.0], "games": 248}],
                      "targets": [1500]},
             "sweep": {"arms": ["a"], "humanlike_reference": {"1500": {"base": hl, "human": hl}}},
             "baselines": {"1500": {"methods": [{"method": "remap", "dial": 1380, "temp": 1.0, "reaches": True, "elo": 1495.0,
                                                 "elo_ci": [1460.0, 1530.0], "humanlike": hl,
                                                 "fidelity": {"top1": 51.0, "sce": 1.9, "top1_vs_base_ci": [-0.6, -0.1],
                                                              "sce_vs_base_ci": [0.001, 0.004]}}]}},
             "arm_a": arm}
    for k, v in files.items():
        (run / f"{k}.json").write_text(json.dumps(v))
    assert report.write(run, res) == {"arms": 1, "catalog": 0}
    text = (res / "summary.md").read_text()
    assert "| 100% of the way, 1500 | 1550 | 1520 | 1480 to 1560 | 1440 to 1600 | +20 | 90% | 21 | -2.50 | 1,600 (stopped early) |" in text
    assert "| remap | 1380 | 1.0 | 1495 | 1460 to 1530 | -5 | 51.0 | -0.6 to -0.1 | +0.001 to +0.004 |" in text
    assert "no training: remap" in text
    assert "humans at 1500" in text and "-130" in text and "5.0 ± 4.3" in text
    assert json.loads((res / "paretos.json").read_text())["1500"]["a"]["distance_to_goal"] == 20.0
