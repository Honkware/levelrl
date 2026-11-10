from __future__ import annotations

import json
import random
from pathlib import Path

# Ladder Elo to Lichess blitz Elo, fitted on the maia3-79m bots: each dial's ladder Elo (from a gauntlet played
# the way the bots sample) against its Lichess rating in one snapshot. Each point is weighted by its ladder
# standard error and its Lichess RD, with RD floored at 150 from 2200 up since those accounts are provisional.
# The spec asks for a constant offset. The first ladder came out compressed at 50 ms a move (nominal 1320-2500 spans
# ~730 points), so a slope is fitted too, and the slope is used when its 95% interval excludes 1.

PROVISIONAL_FROM, PROVISIONAL_SD = 2200, 150.0


def wls(pts, slope: bool):
    w = [1 / (se ** 2 + sd ** 2) for _, se, _, sd in pts]
    sw = sum(w)
    mx = sum(wi * p[0] for wi, p in zip(w, pts)) / sw
    my = sum(wi * p[2] for wi, p in zip(w, pts)) / sw
    if not slope:
        return my - mx, 1.0
    sxx = sum(wi * (p[0] - mx) ** 2 for wi, p in zip(w, pts))
    b = sum(wi * (p[0] - mx) * (p[2] - my) for wi, p in zip(w, pts)) / sxx
    return my - b * mx, b


def fit(pts, n: int = 4000, seed: int = 0) -> dict:
    # pts: (ladder elo, its standard error, lichess rating, lichess sd). Parametric bootstrap over both errors.
    rng = random.Random(seed)
    out = {}
    for name, slope in (("offset", False), ("affine", True)):
        a, b = wls(pts, slope)
        draws = []
        for _ in range(n):
            q = [(rng.gauss(x, se), se, rng.gauss(y, sd), sd) for x, se, y, sd in pts]
            draws.append(wls(q, slope))
        A = sorted(d[0] for d in draws)
        B = sorted(d[1] for d in draws)
        out[name] = {"a": a, "b": b, "a_ci": [A[int(.025 * n)], A[int(.975 * n)]],
                     "b_ci": [B[int(.025 * n)], B[int(.975 * n)]],
                     "resid": [p[2] - (a + b * p[0]) for p in pts]}
    lo, hi = out["affine"]["b_ci"]
    out["use"] = "affine" if not lo <= 1.0 <= hi else "offset"
    return out


class Units:
    def __init__(self, a: float, b: float, rungs: dict[str, float], refused=()):
        self.a, self.b = a, b
        self.rungs = sorted(((e, n) for n, e in rungs.items() if n not in refused))

    @classmethod
    def load(cls, path: Path, calib: Path):
        f = json.loads(path.read_text())
        c = json.loads(calib.read_text())
        m = f["fit"][f["fit"]["use"]]
        return cls(m["a"], m["b"], c["elo"], c["refused"])

    def lich(self, x: float) -> float:
        return self.a + self.b * x

    def ladder(self, lich: float) -> float:
        return (lich - self.a) / self.b

    def closest(self, lich: float, k: int = 3) -> list[str]:
        t = self.ladder(lich)
        return [n for _, n in sorted(self.rungs, key=lambda r: abs(r[0] - t))[:k]]

    def rung_near(self, lich: float, rng: random.Random) -> str:
        # Between two rungs, pick each in proportion to how close the target is, so the expected strength matches.
        t = self.ladder(lich)
        if t <= self.rungs[0][0]:
            return self.rungs[0][1]
        if t >= self.rungs[-1][0]:
            return self.rungs[-1][1]
        for (e0, n0), (e1, n1) in zip(self.rungs, self.rungs[1:]):
            if e0 <= t <= e1:
                return n1 if rng.random() < (t - e0) / (e1 - e0) else n0
        return self.rungs[-1][1]
