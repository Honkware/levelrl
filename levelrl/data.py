from __future__ import annotations

import argparse
import gzip
import io
import json
import random
import re
import subprocess
from collections import Counter
from pathlib import Path

import chess
import chess.pgn

# Maia-3-79M's model card: "Lichess monthly game dumps, January 2023 - July 2025". Nothing from that window
# or before may reach an eval or a reward.
CUTOFF = "2025-07"
MONTHS = ("2025-09", "2026-03", "2026-09")
BANDS = (1100, 1300, 1500, 1700, 1900)
HALF = 50
MIN_GAMES = 50
SKIP_PLIES = 10
MIN_CLOCK = 30.0
N_CALIB = 20_000
N_FID = 50_000

ROOT = Path(__file__).resolve().parent.parent / "data"
URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{}.pgn.zst"
HDR = re.compile(r'^\[(\w+) "(.*)"\]$')
CLK = re.compile(r"\[%clk (\d+):(\d+):(\d+(?:\.\d+)?)\]")


def check_month(month: str) -> None:
    if month <= CUTOFF:
        raise SystemExit(f"leakage guard: {month} is inside Maia-3's training window (ends {CUTOFF})")


def raw(month: str) -> Path:
    check_month(month)
    p = ROOT / "raw" / f"lichess_{month}.pgn.zst"
    if not p.exists() or p.with_suffix(".part").exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.with_suffix(".part").touch()
        subprocess.run(["curl", "-sfL", "-C", "-", "-o", str(p), URL.format(month)], check=True)
        p.with_suffix(".part").unlink()
    return p


def human(h: dict) -> bool:
    return "Rated" in h.get("Event", "") and "BOT" not in (h.get("WhiteTitle", ""), h.get("BlackTitle", ""))


def headers(month: str):
    # Headers only, via grep: a Python pass over ~200 GB of decompressed text per month takes hours.
    keys = "Event|White|Black|WhiteTitle|BlackTitle"
    cmd = f"zstd -dc '{raw(month)}' | grep -E '^\\[({keys}) '"
    with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, text=True, bufsize=1 << 20) as proc:
        h = {}
        for line in proc.stdout:
            if not (mt := HDR.match(line.strip())):
                continue  # a quote inside a player name breaks the pattern; a few lines per month
            k, v = mt.groups()
            if k == "Event" and h:
                yield h
                h = {}
            h[k] = v
        if h:
            yield h


def count_players(months: tuple[str, ...]) -> Counter:
    p = ROOT / f"counts_{'_'.join(months)}.json.gz"
    if p.exists():
        return Counter(json.loads(gzip.decompress(p.read_bytes())))
    n = Counter()
    for m in months:
        for h in headers(m):
            if human(h):
                n[h["White"]] += 1
                n[h["Black"]] += 1
    p.write_bytes(gzip.compress(json.dumps(n).encode()))
    return n


def games(month: str):
    proc = subprocess.Popen(["zstd", "-dc", str(raw(month))], stdout=subprocess.PIPE)
    text = io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace")
    try:
        while (g := chess.pgn.read_game(text)) is not None:
            yield g
    finally:
        proc.kill()


def clock(node) -> float | None:
    m = CLK.search(node.comment or "")
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else None


def pick(g: chess.pgn.Game, rng: random.Random) -> dict | None:
    # One position per game, so rows are not correlated within a game. Plies before SKIP_PLIES are book, and
    # nothing after either clock first drops under MIN_CLOCK.
    nodes, clocks = list(g.mainline()), [None, None]
    usable = []
    for i, node in enumerate(nodes):
        c = clock(node)
        if c is not None:
            clocks[i % 2] = c
        if any(x is not None and x < MIN_CLOCK for x in clocks):
            break
        if i >= SKIP_PLIES:
            usable.append(i)
    if not usable:
        return None
    i = rng.choice(usable)
    h, white = g.headers, i % 2 == 0
    return {
        "site": h["Site"], "moves": [n.move.uci() for n in nodes[:i]], "move": nodes[i].move.uci(),
        "elo": int(h["WhiteElo" if white else "BlackElo"]), "oppo_elo": int(h["BlackElo" if white else "WhiteElo"]),
        "tc": h.get("TimeControl", "-"),
    }


def band(elo: int) -> int | None:
    return next((b for b in BANDS if abs(elo - b) <= HALF), None)


def build(months: tuple[str, ...] = MONTHS, seed: int = 0) -> None:
    for m in months:
        check_month(m)
    counts = count_players(months)
    rng = random.Random(seed)
    per = {b: -(-N_FID // len(months)) for b in BANDS}
    calib_per = -(-N_CALIB // len(months))
    calib, fid = [], {b: [] for b in BANDS}
    for m in months:
        got, cal = Counter(), 0
        for g in games(m):
            h = g.headers
            if not human(h) or not h.get("WhiteElo", "?").isdigit() or not h.get("BlackElo", "?").isdigit():
                continue
            if min(counts[h["White"]], counts[h["Black"]]) < MIN_GAMES:
                continue
            row = pick(g, rng)
            if row is None:
                continue
            row["month"] = m
            b = band(row["elo"])
            if cal < calib_per:
                calib.append(row)
                cal += 1
            elif b is not None and got[b] < per[b]:
                fid[b].append(row)
                got[b] += 1
            if cal >= calib_per and all(got[b] >= per[b] for b in BANDS):
                break
    out = ROOT / "positions"
    out.mkdir(parents=True, exist_ok=True)
    write(out / "calib.jsonl.gz", calib)
    for b, rows in fid.items():
        write(out / f"fid_{b}.jsonl.gz", rows)
    (out / "manifest.json").write_text(json.dumps({
        "cutoff": CUTOFF, "months": months, "seed": seed, "bands": BANDS, "half": HALF, "min_games": MIN_GAMES,
        "skip_plies": SKIP_PLIES, "min_clock": MIN_CLOCK, "calib": len(calib),
        "fid": {b: len(r) for b, r in fid.items()},
    }, indent=1))


def write(p: Path, rows: list[dict]) -> None:
    with gzip.open(p, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load(name: str) -> list[dict]:
    rows = [json.loads(line) for line in gzip.open(ROOT / "positions" / f"{name}.jsonl.gz", "rt")]
    for r in rows:
        check_month(r["month"])
    return rows


def board(row: dict) -> chess.Board:
    b = chess.Board()
    for mv in row["moves"]:
        b.push_uci(mv)
    return b


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", default=list(MONTHS))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(tuple(a.months), a.seed)
