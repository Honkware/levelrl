from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from levelrl import data, maia

# Gate before any RL: the base model has to reproduce the paper. Evaluated the way the paper describes it
# (history on, each mover's own rating and the opponent's real rating, argmax), against Figure 5's cells within
# 200 points of the diagonal. The figure has no intervals and was read off an image, so the pass rule allows
# SLACK points on top of our own bootstrap interval. Written down before the first run.
N = 10_000
SLACK = 1.0
REF = json.loads((data.ROOT / "parity_reference.json").read_text())["accuracy"]


def cell(elo: int) -> str:
    return str(min(2800, max(600, elo // 200 * 200)))


def hits(cfg, model, rows: list[dict], bsz: int = 512) -> list[int]:
    out = []
    for i in range(0, len(rows), bsz):
        chunk = rows[i:i + bsz]
        boards = [data.board(r) for r in chunk]
        logits, _ = maia.forward(cfg, model, boards, [r["elo"] for r in chunk], [r["oppo_elo"] for r in chunk],
                                 history=True)
        out += [int(maia.to_move(b, int(j)).uci() == r["move"]) for b, j, r in zip(boards, logits.argmax(-1), chunk)]
    return out


def ci(x: list[int], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(x, k=len(x))) / len(x) for _ in range(n))
    return 100 * means[int(0.025 * n)], 100 * means[int(0.975 * n)]


def run(device: str = "mps") -> dict:
    cfg, model = maia.load(device)
    res = {"rule": f"bootstrap 95% CI of ours, widened by {SLACK} pt, must cover the Figure 5 value", "bands": {}}
    for b in data.BANDS:
        rows = [r for r in data.load(f"fid_{b}") if cell(r["oppo_elo"]) in REF[cell(r["elo"])]
                and abs(int(cell(r["oppo_elo"])) - int(cell(r["elo"]))) <= 200][:N]
        h = hits(cfg, model, rows)
        by = defaultdict(list)
        for r, x in zip(rows, h):
            by[cell(r["oppo_elo"])].append(x)
        ref = sum(REF[cell(b)][c] * len(v) for c, v in by.items()) / len(rows)
        lo, hi = ci(h)
        acc = 100 * sum(h) / len(h)
        res["bands"][b] = {"n": len(rows), "acc": acc, "ci": [lo, hi], "ref": ref,
                           "cells": {c: {"n": len(v), "acc": 100 * sum(v) / len(v), "ref": REF[cell(b)][c]}
                                     for c, v in sorted(by.items())},
                           "pass": lo - SLACK <= ref <= hi + SLACK}
        print(b, json.dumps({k: v for k, v in res["bands"][b].items() if k != "cells"}), flush=True)
    res["pass"] = all(v["pass"] for v in res["bands"].values())
    Path("runs").mkdir(exist_ok=True)
    Path("runs/parity.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    print("pass" if run()["pass"] else "FAIL")
