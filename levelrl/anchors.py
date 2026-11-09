from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# The maia3-79m_<dial> bots run the same checkpoint we post-train and play humans only, so their Lichess
# ratings pin our ladder to Lichess units with no games of our own. Read-only: no account, no token.
DIALS = (600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600)
ROOT = Path(__file__).resolve().parent.parent / "data" / "anchors"
UA = {"User-Agent": "levelrl (research; read-only)"}


def get(url: str, accept: str = "application/json") -> bytes:
    req = urllib.request.Request(url, headers={**UA, "Accept": accept})
    for attempt in range(5):
        try:
            return urllib.request.urlopen(req, timeout=120).read()
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(60 * (attempt + 1))  # Lichess asks for a full minute after a 429
    raise RuntimeError(f"rate limited: {url}")


def snapshot() -> Path:
    # Never mix snapshots: these bots are still playing, so each file is one moment, named by its time.
    at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = {"at": at, "bots": {}}
    for d in DIALS:
        u = json.loads(get(f"https://lichess.org/api/user/maia3-79m_{d}"))
        out["bots"][d] = {k: {f: v.get(f) for f in ("rating", "rd", "prog", "games", "prov")}
                          for k, v in u.get("perfs", {}).items() if k in ("bullet", "blitz", "rapid", "classical")}
        time.sleep(1)
    ROOT.mkdir(parents=True, exist_ok=True)
    p = ROOT / f"lichess_maia3_calibration_{at}.json"
    p.write_text(json.dumps(out, indent=1))
    return p


def games(dial: int, n: int = 200) -> Path:
    p = ROOT / "games" / f"maia3-79m_{dial}.ndjson"
    if p.exists():
        return p
    url = (f"https://lichess.org/api/games/user/maia3-79m_{dial}?max={n}&perfType=blitz&rated=true"
           f"&moves=true&clocks=false&evals=false&opening=false")
    body = get(url, accept="application/x-ndjson")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(body)
    return p


def bot_positions(dial: int) -> list[tuple]:
    # (board before the bot's move, the move it played, the human's rating) for every bot move in the export.
    import chess
    out = []
    for line in games(dial).open():
        g = json.loads(line)
        if g.get("variant") != "standard" or "initialFen" in g:
            continue
        side = "white" if g["players"]["white"].get("user", {}).get("id") == f"maia3-79m_{dial}" else "black"
        human = g["players"]["black" if side == "white" else "white"].get("rating")
        b = chess.Board()
        for i, san in enumerate(g["moves"].split()):
            mv = b.parse_san(san)
            if (i % 2 == 0) == (side == "white") and human:
                out.append((b.copy(), mv, human))
            b.push(mv)
    return out


def nucleus_need(p, idx):
    # maia3's nucleus keeps a move while the cumulative mass up to and including it is <= top_p, and always keeps
    # the top move. So this is the smallest top_p whose nucleus still contains the move at idx.
    import torch
    pm = p.gather(1, idx[:, None])
    above = (p * (p > pm)).sum(-1)
    return torch.where(above == 0, torch.zeros_like(above), above + pm[:, 0])


def infer(device: str = "mps", bsz: int = 512) -> dict:
    # Which settings make the bots' own moves most likely: history on or off, the opponent Elo set to the dial or
    # to the human's real rating, and the sampling temperature. top-p is judged by how often a played move falls
    # outside the nucleus, since a move outside it has zero probability under that setting.
    import torch
    from levelrl import maia
    cfg, model = maia.load(device)
    temps = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
    res = {}
    for d in DIALS:
        pos = bot_positions(d)
        if not pos:
            continue
        row = {"moves": len(pos)}
        for history in (False, True):
            for oppo in ("dial", "human"):
                lp, top1, ranks = {t: 0.0 for t in temps}, 0, []
                for i in range(0, len(pos), bsz):
                    chunk = pos[i:i + bsz]
                    boards = [b for b, _, _ in chunk]
                    oe = [d if oppo == "dial" else h for _, _, h in chunk]
                    logits, _ = maia.forward(cfg, model, boards, [d] * len(chunk), oe, history=history)
                    idx = torch.tensor([maia.MOVE_IDX[mv.uci() if b.turn else maia.mirror_move(mv.uci())]
                                        for b, mv, _ in chunk], device=logits.device)
                    top1 += int((logits.argmax(-1) == idx).sum())
                    p = torch.softmax(logits, -1)
                    ranks += nucleus_need(p, idx).tolist()
                    for t in temps:
                        lp[t] += float(torch.log_softmax(logits / t, -1).gather(1, idx[:, None]).sum())
                n = len(pos)
                key = f"history={'on' if history else 'off'},oppo={oppo}"
                row[key] = {"top1": top1 / n, "best_t": max(temps, key=lambda t: lp[t]),
                            "nll": {str(t): -lp[t] / n for t in temps},
                            "outside_top_p": {str(q): sum(r > q for r in ranks) / n for q in (0.8, 0.9, 0.95)}}
        res[d] = row
        print(d, json.dumps(row)[:300], flush=True)
    (ROOT / "bot_settings.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["snapshot", "games", "infer"])
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    if a.cmd == "snapshot":
        print(snapshot())
    elif a.cmd == "infer":
        infer()
    else:
        for d in DIALS:
            p = games(d, a.n)
            print(d, sum(1 for _ in p.open()), "games")
