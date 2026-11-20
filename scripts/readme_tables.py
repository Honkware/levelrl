from __future__ import annotations

import json
import sys
from pathlib import Path

# Tables in the README, rebuilt from results/runN. The confirmation gauntlet size is not stored in the result
# files: finish_arm played suite[:200] once with both colours in run 1 (commit d45b763) and twice in run 2
# (commit 9f93035). Move-match is on the last 10,000 positions of fid_T (evals.FID_N) in both runs.
CONFIRM = {"run1": 400, "run2": 800}
FID_N = 10_000
ROOT = Path(__file__).resolve().parent.parent / "results"


def _load(run: str, name: str):
    return json.loads((ROOT / run / f"{name}.json").read_text())


def _arm(name: str) -> str:
    out, fid = name.split("_out")[1].split("_fid")
    return {"1.0": "outcome only", "0.0": "fidelity only"}.get(out, f"outcome {out}, fidelity {fid}")


def table(run: str) -> list[str]:
    base = {c["dial"]: c for c in _load(run, "base_curve")["curve"]}
    paretos = _load(run, "paretos")
    kl = any(a.get("kl_base") is not None for t in paretos.values() for a in t.values())
    L = ["| target | player | games | Elo | 95% CI | off target | top-1 % | top-1 vs base, pp |" + (" KL |" if kl else ""),
         "|---|---|---|---|---|---|---|---|" + ("---|" if kl else "")]
    for T in sorted(paretos, key=int):
        b, t = base[int(T)], int(T)
        L.append(f"| {t} | dial {t} as asked | {b['games']} | {b['lichess']:.0f} | {b['lichess_ci'][0]:.0f} to "
                 f"{b['lichess_ci'][1]:.0f} | {b['lichess'] - t:+.0f} | | |" + (" |" if kl else ""))
        for name, a in sorted(paretos[T].items(), key=lambda x: (-x[1]["w_out"], x[0])):
            d = a["fidelity_top1_vs_base_ci"]
            L.append(f"| {t} | {_arm(name)}, checkpoint {a['final_tag']} | {CONFIRM[run]} | {a['elo']:.0f} | "
                     f"{a['elo_ci'][0]:.0f} to {a['elo_ci'][1]:.0f} | {a['distance_to_T']:+.0f} | "
                     f"{a['fidelity_top1']:.1f} | {d[0]:+.1f} to {d[1]:+.1f} |"
                     + (f" {a['kl_base']:.3f} |" if kl else ""))
    return L


def main():
    runs = sys.argv[1:] or sorted(p.name for p in ROOT.glob("run*") if (p / "paretos.json").exists())
    for run in runs:
        print(f"{run}: Elo is Lichess blitz from results/{run}/base_curve.json and results/{run}/paretos.json, "
              f"top-1 on {FID_N:,} held-out human positions at the target band\n")
        print("\n".join(table(run)), end="\n\n")


if __name__ == "__main__":
    main()
