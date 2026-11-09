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
    arm = {"T": 1500, "w_out": 0.5, "w_fid": 0.5, "evals": [ev("25", 25, 1600.0), ev("final", 90, 1520.0)],
           "final": {"tag": "final", "elo": 1520.0, "elo_ci": [1480.0, 1560.0],
                     "fidelity": {"top1": 51.2, "sce": 1.9, "top1_vs_base_ci": [-0.5, 0.3], "sce_vs_base_ci": [-0.01, 0.01]}},
           "sensitivity": {"1300": {"elo": 1400.0}, "1500": {"elo": 1520.0}, "1700": {"elo": 1650.0}},
           "humanlike": hl, "recoveries": 0}
    files = {"anchors": {"snapshot": "s.json", "fit": {"use": "affine", "affine": fit},
                         "dials": [{"dial": 600, "ladder": 1050.0, "lichess": 890}]},
             "base": {"curve": [{"dial": 1500, "lichess": 1650.0, "lichess_ci": [1600.0, 1700.0], "games": 248}],
                      "targets": [1500]},
             "sweep": {"arms": ["a"], "humanlike_reference": {"1500": {"base": hl, "human": hl}}},
             "arm_a": arm}
    for k, v in files.items():
        (run / f"{k}.json").write_text(json.dumps(v))
    assert report.write(run, res) == {"arms": 1, "catalog": 0}
    text = (res / "summary.md").read_text()
    assert "| 0.5/0.5 | final | 1520 |" in text and "humans at 1500" in text and "+250" in text
    assert json.loads((res / "paretos.json").read_text())["1500"]["a"]["distance_to_T"] == 20.0
