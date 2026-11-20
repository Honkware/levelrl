from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import chess

from .errors import ValidationError
from .positions import ProbePosition

_PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


@dataclass(frozen=True, slots=True)
class EngineEvaluation:
    score: int
    pv: tuple[str, ...] = ()


@runtime_checkable
class Engine(Protocol):
    def evaluate(self, board: chess.Board, *, nodes: int) -> EngineEvaluation:
        """Return centipawns from the side-to-move's perspective and an optional principal variation."""


@dataclass(frozen=True, slots=True)
class MoveAnalysis:
    loss: int
    phase: str
    kind: str | None = None


def game_phase(board: chess.Board, ply: int) -> str:
    if ply < 20:
        return "opening"
    material = sum(_PIECE_VALUES[piece.piece_type] for piece in board.piece_map().values()
                   if piece.piece_type != chess.PAWN)
    return "endgame" if material <= 26 else "middlegame"


def _clipped(score: int, limit: int) -> int:
    return max(-limit, min(limit, score))


def analyse_move(
    engine: Engine,
    position: ProbePosition,
    *,
    ply: int = 0,
    nodes: int = 100_000,
    mate_score: int = 1000,
    clip: int = 1000,
    blunder: int = 200,
) -> MoveAnalysis:
    if nodes <= 0 or mate_score <= 0 or clip <= 0:
        raise ValidationError("nodes, mate_score, and clip must be positive")
    board = position.board()
    mover = board.turn
    phase = game_phase(board, ply)
    best = _clipped(engine.evaluate(board, nodes=nodes).score, clip)
    board.push_uci(position.move)
    if board.is_game_over():
        after = mate_score if board.is_checkmate() else 0
        return MoveAnalysis(max(0, best - after), phase)
    evaluation = engine.evaluate(board, nodes=nodes)
    after = _clipped(-evaluation.score, clip)
    loss = max(0, best - after)
    kind = None
    if loss >= blunder:
        kind = "tactical oversight"
        if evaluation.pv:
            try:
                reply = chess.Move.from_uci(evaluation.pv[0])
            except ValueError as error:
                raise ValidationError("engine returned an invalid principal-variation move",
                                      move=evaluation.pv[0]) from error
            if reply in board.legal_moves and board.is_capture(reply):
                victim = board.piece_at(reply.to_square)
                attacker = board.piece_at(reply.from_square)
                if victim and (not board.is_attacked_by(mover, reply.to_square)
                               or (attacker and _PIECE_VALUES[attacker.piece_type] < _PIECE_VALUES[victim.piece_type])):
                    kind = "piece drop"
    return MoveAnalysis(loss, phase, kind)


def human_likeness(
    analyses: Sequence[MoveAnalysis],
    *,
    mistake: int = 100,
    blunder: int = 200,
) -> dict:
    if mistake < 0 or blunder < 0:
        raise ValidationError("loss thresholds must be non-negative", mistake=mistake, blunder=blunder)
    count = len(analyses)
    by_phase = {}
    for phase in ("opening", "middlegame", "endgame"):
        rows = [row for row in analyses if row.phase == phase]
        if rows:
            by_phase[phase] = {
                "n": len(rows),
                "mistake_rate": sum(row.loss >= mistake for row in rows) / len(rows),
                "blunder_rate": sum(row.loss >= blunder for row in rows) / len(rows),
                "mean_cp_loss": sum(row.loss for row in rows) / len(rows),
            }
    blunders = [row for row in analyses if row.kind is not None]
    return {
        "n": count,
        "blunder_rate": len(blunders) / max(1, count),
        "blunder_types": {
            kind: sum(row.kind == kind for row in blunders) / max(1, len(blunders))
            for kind in ("piece drop", "tactical oversight")
        },
        "by_phase": by_phase,
    }
