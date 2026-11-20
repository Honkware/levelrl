from __future__ import annotations

import sys
from pathlib import Path

from maia_calibrate import load_map

# Resolves requested ratings to the dial and temperature measured for them in results/calibration.
MAP = Path(__file__).resolve().parent.parent / "results" / "calibration" / "calibration_map.json"


def main():
    cal = load_map(MAP)
    measured = {m.target: m for m in cal.measurements}
    for target in map(int, sys.argv[1:] or ("1100", "1500", "1950", "2600")):
        r = cal.lookup(target)
        m = measured[r.nearest_target]
        how = "measured" if r.validated else f"nearest measured target is {r.nearest_target}"
        band = "inside" if m.strength_in_band else "outside"
        print(f"{target}: dial {r.setting.dial}, temperature {r.setting.temp} ({r.setting.method}, {how}), "
              f"played {m.measured_elo:.0f} [{m.elo_ci[0]:.0f}, {m.elo_ci[1]:.0f}], {band} {r.nearest_target} +-50")


if __name__ == "__main__":
    main()
