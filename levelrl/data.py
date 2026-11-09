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
KEEP = 0.01

ROOT = Path(__file__).resolve().parent.parent / "data"
URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{}.pgn.zst"
HDR = re.compile(r'^\[(\w+) "(.*)"\]$')
CLK = re.compile(r"\[%clk (\d+):(\d+):(\d+(?:\.\d+)?)\]")


def check_month(month: str) -> None:
    if month <= CUTOFF:
        raise SystemExit(f"leakage guard: {month} is inside Maia-3's training window (ends {CUTOFF})")


def source(month: str) -> str:
    # A local copy if there is one, else the dump streamed from Lichess. Home links here manage 2-3 MB/s,
    # so a month is about three hours either way, and streaming needs no disk.
    check_month(month)
    p = ROOT / "raw" / f"lichess_db_standard_rated_{month}.pgn.zst"
    return f"zstd -dc '{p}'" if p.exists() else f"curl -sfL '{URL.format(month)}' | zstd -dc"


def human(h: dict) -> bool:
    return "Rated" in h.get("Event", "") and "BOT" not in (h.get("WhiteTitle", ""), h.get("BlackTitle", ""))


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
        "white": h["White"], "black": h["Black"], "tc": h.get("TimeControl", "-"),
    }


def scan(month: str, keep: float = KEEP, seed: int = 0) -> Path:
    # One pass per month: every human game counts toward its players' totals, and a random `keep` share is
    # parsed into a candidate position. The 50-game rule needs the whole window counted, so it is applied in
    # build(), after every month is in.
    out = ROOT / "scan" / month
    if (out / "done").exists():
        return out
    rng, counts, rows, buf, h = random.Random(f"{seed}-{month}"), Counter(), [], [], {}

    seen = [0]

    def flush():
        seen[0] += 1
        if seen[0] % 2_000_000 == 0:
            print(f"{month}: {seen[0] // 1_000_000}M games, {len(rows)} rows", flush=True)
        if not h or not human(h):
            return
        counts[h["White"]] += 1
        counts[h["Black"]] += 1
        if h.get("WhiteElo", "?").isdigit() and h.get("BlackElo", "?").isdigit() and rng.random() < keep:
            if (g := chess.pgn.read_game(io.StringIO("".join(buf)))) and (row := pick(g, rng)):
                rows.append(row | {"month": month})

    proc = subprocess.Popen(["bash", "-c", f"set -o pipefail; {source(month)}"], stdout=subprocess.PIPE)
    for line in io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace"):
        if line.startswith("[Event ") and buf:
            flush()
            buf, h = [], {}
        buf.append(line)
        if line[0] == "[" and (mt := HDR.match(line.rstrip())):
            h[mt[1]] = mt[2]
    flush()
    # A dropped stream would undercount every player and quietly bias the 50-game rule.
    assert proc.wait() == 0, f"{month}: stream failed ({proc.returncode})"
    out.mkdir(parents=True, exist_ok=True)
    (out / "counts.json.gz").write_bytes(gzip.compress(json.dumps(counts).encode()))
    write(out / "rows.jsonl.gz", rows)
    (out / "done").write_text(json.dumps({"games": sum(counts.values()) // 2, "rows": len(rows), "keep": keep}))
    return out


def blitz(tc: str) -> bool:
    # Lichess's own speed rule (base + 40 x increment, 180-479 s). Maia-3 trained and was tested on blitz only.
    base, _, inc = tc.partition("+")
    return base.isdigit() and inc.isdigit() and 180 <= int(base) + 40 * int(inc) < 480


def band(elo: int) -> int | None:
    return next((b for b in BANDS if abs(elo - b) <= HALF), None)


def build(months: tuple[str, ...] = MONTHS, seed: int = 0) -> None:
    for m in months:
        check_month(m)
    counts, by_month = Counter(), {}
    for m in months:
        d = scan(m, seed=seed)
        counts.update(json.loads(gzip.decompress((d / "counts.json.gz").read_bytes())))
        by_month[m] = [r for line in gzip.open(d / "rows.jsonl.gz", "rt") if blitz((r := json.loads(line))["tc"])]
    rng = random.Random(seed)
    calib, fid = [], {b: [] for b in BANDS}
    # Equal shares per month, so both years carry the same weight.
    n_cal, n_fid = -(-N_CALIB // len(months)), -(-N_FID // len(months))
    for m, rows in by_month.items():
        rng.shuffle(rows)
        cal = rows[:n_cal]
        used = {r["site"] for r in cal}
        calib += cal
        for b in BANDS:
            fid[b] += [r for r in rows[n_cal:] if band(r["elo"]) == b and r["site"] not in used
                       and min(counts[r["white"]], counts[r["black"]]) >= MIN_GAMES][:n_fid]
    out = ROOT / "positions"
    out.mkdir(parents=True, exist_ok=True)
    write(out / "calib.jsonl.gz", calib)
    for b, rows in fid.items():
        write(out / f"fid_{b}.jsonl.gz", rows)
    (out / "manifest.json").write_text(json.dumps({
        "cutoff": CUTOFF, "months": months, "seed": seed, "bands": BANDS, "half": HALF, "min_games": MIN_GAMES,
        "skip_plies": SKIP_PLIES, "min_clock": MIN_CLOCK, "keep": KEEP, "speed": "blitz", "calib": len(calib),
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
    ap.add_argument("--scan", help="scan one month only")
    a = ap.parse_args()
    if a.scan:
        scan(a.scan, seed=a.seed)
    else:
        build(tuple(a.months), a.seed)
