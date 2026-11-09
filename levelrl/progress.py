from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path

# Long jobs drop their state here so a run can be checked from outside: {done, total} plus a line on what the job is
# doing, a few recent events and some stats. One game at a time also streams its position to board.json.
# Writes are throttled, so calling these every step is fine.
ROOT = Path(__file__).resolve().parent.parent / "runs" / "progress"
_jobs: dict[str, dict] = {}
_last: dict[str, float] = {}
_board = {"t": 0.0, "game": None}


def _write(path: Path, obj: dict) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def tick(name: str, done: int, total: int | None, unit: str = "", force: bool = False, note: str | None = None,
         stats: dict | None = None) -> None:
    now = time.time()
    j = _jobs.setdefault(name, {"name": name, "start": now, "events": []})
    j.update(done=done, total=total, unit=unit, t=now)
    if note is not None:
        j["note"] = note
    if stats is not None:
        j["stats"] = stats
    if force or done == total or now - _last.get(name, 0) >= 1:
        _last[name] = now
        _write(ROOT / f"{name}.json", j)


def event(name: str, text: str, keep: int = 6) -> None:
    j = _jobs.setdefault(name, {"name": name, "start": time.time(), "events": [], "done": 0, "total": None})
    j["events"] = (j["events"] + [[time.time(), text]])[-keep:]


def board(job: str, game, b, white: str, black: str, result: str | None = None) -> None:
    # Many processes may call this. The first to find board.json stale claims it and keeps it until its game ends,
    # so the view follows one game instead of flickering between all of them.
    now = time.time()
    if result is None and now - _board["t"] < 0.4:
        return
    _board["t"] = now
    p = ROOT / "board.json"
    try:
        cur = json.loads(p.read_text())
    except (OSError, ValueError):
        cur = None
    mine = cur is not None and cur.get("pid") == os.getpid() and cur.get("game") == game
    if cur is not None and not mine and now - cur["t"] < (6 if cur.get("result") else 3):
        return
    last = san = None
    if b.move_stack:
        mv = b.pop()
        last, san = mv.uci(), b.san(mv)
        b.push(mv)
    _write(p, {"pid": os.getpid(), "game": game, "job": job, "fen": b.fen(), "last": last, "san": san,
               "white": white, "black": black, "ply": b.ply(), "result": result, "t": now})
