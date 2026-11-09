from __future__ import annotations

import gzip
import random

import chess
import chess.engine
import torch

from levelrl import arena, data, maia

# Measurements that never feed a reward: move-match probes on held-out human positions, the human-likeness
# analysis (blunder types and per-phase error rates judged by Stockfish), and opening-book overlap.

PROBE_N, BAND_N, FID_N = 2000, 1000, 10_000
BLUNDER, MISTAKE = 200, 100
NODES = 100_000
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
_rows: dict = {}


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
        s["fid_T"] = rows(f"fid_{T}")[-FID_N:]
    return s


@torch.no_grad()
def score(cfg, model, rs: list[dict], self_elo: int | None = None, bsz: int = 1024) -> dict:
    # Per position: did argmax match the human move, and the human move's negative log-likelihood. Ratings are
    # the row's own unless self_elo pins the dial.
    hit, nll = [], []
    for i in range(0, len(rs), bsz):
        ch = rs[i:i + bsz]
        boards = [data.board(r) for r in ch]
        x = torch.stack([maia.tokens(cfg, b, True) for b in boards])
        m = torch.stack([maia.legal(b) for b in boards])
        se = [self_elo or r["elo"] for r in ch]
        oe = [self_elo or r["oppo_elo"] for r in ch]
        logits, _ = maia.run(model, x, m, se, oe, cfg.device)
        lp = torch.log_softmax(logits, -1)
        idx = torch.tensor([maia.MOVE_IDX[r["move"] if b.turn else maia.mirror_move(r["move"])]
                            for b, r in zip(boards, ch)], device=lp.device)
        hit += (logits.argmax(-1) == idx).int().tolist()
        nll += (-lp.gather(1, idx[:, None])[:, 0]).tolist()
    return {"hit": hit, "nll": nll}


def summary(s: dict) -> dict:
    n = len(s["hit"])
    return {"n": n, "top1": 100 * sum(s["hit"]) / n, "sce": sum(s["nll"]) / n}


def paired_ci(a: list[float], b: list[float], n: int = 2000, seed: int = 0) -> list[float]:
    # 95% bootstrap interval of mean(a) - mean(b) over the same positions.
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    k = len(d)
    v = sorted(sum(d[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return [v[int(.025 * n)], v[int(.975 * n)]]


def probe(cfg, model, base_scores: dict | None, T: int | None, full: bool = False) -> tuple[dict, dict]:
    sets = probe_sets(T)
    if not full:
        sets.pop("fid_T", None)
    raw, out = {}, {}
    for k, rs in sets.items():
        raw[k] = score(cfg, model, rs, self_elo=T if k == "fid_T" else None)
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


def humanlike(moves: list[tuple]) -> dict:
    res = arena.pool().map(_analyse, moves, chunksize=16)
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
