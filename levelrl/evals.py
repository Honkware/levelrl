from __future__ import annotations

import gzip
import random

import chess
import chess.engine
import numpy as np
import torch

from levelrl import arena, data, maia, progress

# Measurements that never feed a reward: move-match probes on held-out human positions, the human-likeness
# analysis (blunder types and per-phase error rates judged by Stockfish), and opening-book overlap.

PROBE_N, BAND_N, VAL_N, FID_N = 2000, 1000, 5000, 10_000
BLUNDER, MISTAKE = 200, 100
NODES = 100_000
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
_rows: dict = {}
_features: dict = {}


def rows(name: str) -> list[dict]:
    if name not in _rows:
        _rows[name] = data.load(name)
    return _rows[name]


def probe_sets(T: int | None = None) -> dict[str, list[dict]]:
    # Fixed slices, so every checkpoint and the base model see the same positions. The 10k fidelity set comes from
    # the far end of the band file and never overlaps the per-band probe.
    s = {"calib": rows("calib")[:PROBE_N]}
    for b in data.BANDS:
        s[f"band_{b}"] = rows(f"fid_{b}")[:BAND_N]
    if T is not None and T in data.BANDS:
        # val_T is for choosing between checkpoints, fid_T is only ever reported. The band files hold one position
        # a game in month order, so the two slices share no game and come from different months.
        s["val_T"] = rows(f"fid_{T}")[BAND_N:BAND_N + VAL_N]
        s["fid_T"] = rows(f"fid_{T}")[-FID_N:]
    return s


def features(cfg, rs: list[dict], bsz: int):
    key = (str(cfg.device), cfg.history, bsz, tuple(id(r) for r in rs))
    if key not in _features:
        batches = []
        for i in range(0, len(rs), bsz):
            if len(rs) > 2 * bsz:  # a 10k set takes a while the first time, and nothing else is reporting then
                progress.tick("positions", i, len(rs), "positions", note="encoding held-out human positions, once a run")
            ch = rs[i:i + bsz]
            encoded = [maia.replay_features(cfg, r["moves"]) for r in ch]
            boards = [b for b, _, _ in encoded]
            x = torch.stack([v for _, v, _ in encoded]).to(cfg.device)
            m = torch.stack([mask for _, _, mask in encoded]).to(cfg.device)
            idx = torch.tensor([maia.MOVE_IDX[r["move"] if b.turn else maia.mirror_move(r["move"])]
                                for b, r in zip(boards, ch)], device=cfg.device)
            batches.append((x, m, idx))
        if len(rs) > 2 * bsz:
            progress.tick("positions", len(rs), len(rs), "positions", force=True, note="held-out human positions encoded")
        _features[key] = (rs, batches)
    return _features[key][1]


@torch.no_grad()
def score(cfg, model, rs: list[dict], self_elo: int | None = None, bsz: int = 1024, temp: float = 1.0) -> dict:
    # Per position: did argmax match the human move, and the human move's negative log-likelihood. Ratings are
    # the row's own unless self_elo pins the dial. Features are fixed, so retain them on the device across arms.
    hit, nll = [], []
    for i, (x, m, idx) in enumerate(features(cfg, rs, bsz)):
        ch = rs[i * bsz:(i + 1) * bsz]
        se = [self_elo or r["elo"] for r in ch]
        oe = [self_elo or r["oppo_elo"] for r in ch]
        logits, _ = maia.run(model, x, m, se, oe, cfg.device)
        lp = torch.log_softmax(logits / temp, -1)
        hit += (logits.argmax(-1) == idx).int().tolist()
        nll += (-lp.gather(1, idx[:, None])[:, 0]).tolist()
    return {"hit": hit, "nll": nll}


def summary(s: dict) -> dict:
    n = len(s["hit"])
    return {"n": n, "top1": 100 * sum(s["hit"]) / n, "sce": sum(s["nll"]) / n}


def paired_ci(a: list[float], b: list[float], n: int = 2000, seed: int = 0) -> list[float]:
    # 95% bootstrap interval of mean(a) - mean(b) over the same positions.
    rng = np.random.default_rng(seed)
    d = np.asarray([x - y for x, y in zip(a, b)])
    k = len(d)
    chunks = []
    for i in range(0, n, 100):
        idx = rng.integers(0, k, size=(min(100, n - i), k))
        chunks.append(d[idx].mean(axis=1))
    v = np.sort(np.concatenate(chunks))
    return [float(v[int(.025 * n)]), float(v[int(.975 * n)])]


def probe(cfg, model, base_scores: dict | None, T: int | None, full: bool = False) -> tuple[dict, dict]:
    sets = probe_sets(T)
    if not full:
        sets.pop("fid_T", None)
    raw, out = {}, {}
    for k, rs in sets.items():
        raw[k] = score(cfg, model, rs, self_elo=T if k in ("fid_T", "val_T") else None)
        out[k] = summary(raw[k])
        if base_scores and k in base_scores:
            out[k]["top1_vs_base_ci"] = [100 * x for x in paired_ci(raw[k]["hit"], base_scores[k]["hit"])]
            out[k]["sce_vs_base_ci"] = paired_ci(raw[k]["nll"], base_scores[k]["nll"])
    return out, raw


# --- human-likeness ---------------------------------------------------------------------------------------

_an = None


def _phase(b: chess.Board, ply: int) -> str:
    if ply < 20:
        return "opening"
    mat = sum(VAL[p.piece_type] for p in b.piece_map().values() if p.piece_type != chess.PAWN)
    return "endgame" if mat <= 26 else "middlegame"


def _cp(score: chess.engine.PovScore, turn: bool) -> int:
    return max(-1000, min(1000, score.pov(turn).score(mate_score=1000)))


def _analyse(args):
    # One move: centipawns lost against the engine's best, and, for a blunder, whether the refutation simply
    # takes a piece the move left hanging or needed a real tactic.
    global _an
    fen, uci, ply = args
    if _an is None:
        _an = chess.engine.SimpleEngine.popen_uci("stockfish")
        _an.configure({"Threads": 1, "Hash": 16})
    b = chess.Board(fen)
    me = b.turn
    best = _cp(_an.analyse(b, chess.engine.Limit(nodes=NODES))["score"], me)
    mv = chess.Move.from_uci(uci)
    phase = _phase(b, ply)
    b.push(mv)
    if b.is_game_over():
        after = _cp(chess.engine.PovScore(chess.engine.Mate(0), not me), me) if b.is_checkmate() else 0
        return {"loss": max(0, best - after), "phase": phase, "kind": None}
    info = _an.analyse(b, chess.engine.Limit(nodes=NODES))
    after = _cp(info["score"], me)
    loss = max(0, best - after)
    kind = None
    if loss >= BLUNDER:
        kind = "tactical oversight"
        pv = info.get("pv") or []
        if pv and b.is_capture(pv[0]):
            sq = pv[0].to_square
            victim = b.piece_at(sq)
            attacker = b.piece_at(pv[0].from_square)
            if victim and (not b.is_attacked_by(me, sq) or (attacker and VAL[attacker.piece_type] < VAL[victim.piece_type])):
                kind = "piece drop"
    return {"loss": loss, "phase": phase, "kind": kind}


def moves_of(games: list[dict], limit: int = 20_000) -> list[tuple]:
    # (fen before, the move, ply) for every move the evaluated side made after the opening suite.
    out = []
    for g in games:
        b = chess.Board()
        for i, u in enumerate(g["moves"]):
            if i >= len(g["opening"]) and (b.turn == chess.WHITE) == g["white"]:
                out.append((b.fen(), u, i))
            b.push_uci(u)
    random.Random(0).shuffle(out)
    return out[:limit]


def human_moves(T: int, n: int = 3000) -> list[tuple]:
    out = []
    for r in rows(f"fid_{T}")[:n]:
        b = data.board(r)
        out.append((b.fen(), r["move"], len(r["moves"])))
    return out


def humanlike(moves: list[tuple], on_progress=None, job: str | None = None, what: str = "") -> dict:
    # job names a progress feed for the analysis, so its minutes are not silent.
    if job and on_progress is None:
        on_progress = lambda n: progress.tick(job, n, len(moves), "positions", force=n == len(moves),
                                              note=f"{what} · Stockfish mistake analysis")
        on_progress(0)
    res = []
    for r in arena.pool().imap(_analyse, moves, chunksize=16):
        res.append(r)
        if on_progress and (len(res) % 64 == 0 or len(res) == len(moves)):
            on_progress(len(res))
    n = len(res)
    by = {}
    for ph in ("opening", "middlegame", "endgame"):
        rs = [r for r in res if r["phase"] == ph]
        if rs:
            by[ph] = {"n": len(rs), "mistake_rate": sum(r["loss"] >= MISTAKE for r in rs) / len(rs),
                      "blunder_rate": sum(r["loss"] >= BLUNDER for r in rs) / len(rs),
                      "mean_cp_loss": sum(r["loss"] for r in rs) / len(rs)}
    bl = [r for r in res if r["kind"]]
    return {"n": n, "blunder_rate": len(bl) / max(1, n),
            "blunder_types": {k: sum(r["kind"] == k for r in bl) / max(1, len(bl))
                              for k in ("piece drop", "tactical oversight")},
            "by_phase": by}


# --- opening book ----------------------------------------------------------------------------------------

_book = None


def book() -> set:
    global _book
    if _book is None:
        _book = set(gzip.open(data.ROOT / "book.txt.gz", "rt").read().splitlines())
    return _book


def book_overlap(games: list[dict], lo: int = 9, hi: int = 12) -> float:
    # Share of the evaluated side's moves at plies lo..hi whose line so far is one humans played at least 3 times
    # in the scanned months.
    bk, hit, n = book(), 0, 0
    for g in games:
        for k in range(lo, min(hi, len(g["moves"])) + 1):
            if ((k - 1) % 2 == 0) == g["white"]:
                n += 1
                hit += " ".join(g["moves"][:k]) in bk
    return hit / max(1, n)


def threefold(games: list[dict]) -> float:
    return sum(g["end"] == "threefold repetition" for g in games) / max(1, len(games))
