from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Long jobs drop {done, total} here so a run can be checked from outside. At most one write a second.
ROOT = Path(__file__).resolve().parent.parent / "runs" / "progress"
_start: dict[str, float] = {}
_last: dict[str, float] = {}


def tick(name: str, done: int, total: int | None, unit: str = "", force: bool = False) -> None:
    now = time.time()
    _start.setdefault(name, now)
    if not force and done != total and now - _last.get(name, 0) < 1:
        return
    _last[name] = now
    ROOT.mkdir(parents=True, exist_ok=True)
    tmp = ROOT / f".{name}.tmp"
    tmp.write_text(json.dumps({"name": name, "done": done, "total": total, "unit": unit,
                               "start": _start[name], "t": now}))
    os.replace(tmp, ROOT / f"{name}.json")
