from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import chess

from .errors import ChessEvalError, DataError, InvalidPositionError, MissingDataError, ValidationError
from .manifest import _validate_month, check_month


@dataclass(frozen=True, slots=True)
class ProbePosition:
    fen: str
    move: str
    game_id: str | None = None
    month: str | None = None
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            board = chess.Board(self.fen)
        except (TypeError, ValueError) as error:
            raise InvalidPositionError("invalid FEN", fen=self.fen, detail=str(error)) from error
        if not board.is_valid():
            raise InvalidPositionError("FEN does not describe a legal chess position", fen=self.fen)
        try:
            move = chess.Move.from_uci(self.move)
        except (TypeError, ValueError) as error:
            raise InvalidPositionError("invalid UCI move", move=self.move, fen=self.fen) from error
        if move not in board.legal_moves:
            raise InvalidPositionError("move is not legal in this position", move=self.move, fen=self.fen)
        if self.game_id is not None and not isinstance(self.game_id, str):
            raise DataError("game_id must be a string", game_id=self.game_id)
        if self.month is not None:
            _validate_month(self.month, "month")
        if not isinstance(self.context, Mapping):
            raise DataError("probe context must be a mapping")
        object.__setattr__(self, "context", MappingProxyType(dict(self.context)))

    def board(self) -> chess.Board:
        return chess.Board(self.fen)

    @classmethod
    def from_moves(
        cls,
        moves: Sequence[str],
        move: str,
        *,
        fen: str = chess.STARTING_FEN,
        game_id: str | None = None,
        month: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> ProbePosition:
        try:
            board = chess.Board(fen)
            for index, uci in enumerate(moves):
                board.push_uci(uci)
        except (TypeError, ValueError) as error:
            raise InvalidPositionError("invalid move history", index=index if "index" in locals() else None,
                                       detail=str(error)) from error
        return cls(board.fen(), move, game_id, month, context or {})

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> ProbePosition:
        if not isinstance(row, Mapping):
            raise DataError("probe row must be a JSON object")
        if "move" not in row:
            raise MissingDataError("probe row is missing the human move", field="move")
        if "fen" not in row and "moves" not in row:
            raise MissingDataError("probe row needs either fen or moves", fields=["fen", "moves"])
        if "fen" in row and "moves" in row:
            raise DataError("probe row must contain only one of fen or moves")
        if not isinstance(row.get("context", {}), Mapping):
            raise DataError("probe context must be a JSON object")
        if "moves" in row and not isinstance(row["moves"], list):
            raise DataError("probe moves must be a JSON array")
        reserved = {"fen", "moves", "move", "game_id", "site", "month", "context"}
        context = dict(row.get("context", {}))
        context.update({key: value for key, value in row.items() if key not in reserved})
        game_id = row.get("game_id", row.get("site"))
        if "fen" in row:
            return cls(row["fen"], row["move"], game_id, row.get("month"), context)
        return cls.from_moves(row["moves"], row["move"], game_id=game_id, month=row.get("month"), context=context)

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {"fen": self.fen, "move": self.move, "context": dict(self.context)}
        if self.game_id is not None:
            row["game_id"] = self.game_id
        if self.month is not None:
            row["month"] = self.month
        return row


@runtime_checkable
class MovePredictor(Protocol):
    def predict(self, position: ProbePosition) -> Mapping[str, float]:
        """Return a normalized probability for every legal UCI move."""


def load_probe_set(path: str | Path, *, cutoff: str | None = None) -> tuple[ProbePosition, ...]:
    source = Path(path)
    if not source.is_file():
        raise MissingDataError("probe-set file does not exist", path=str(source))
    opener = gzip.open if source.suffix == ".gz" else open
    positions = []
    try:
        with opener(source, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    position = ProbePosition.from_row(row)
                    if cutoff is not None:
                        if position.month is None:
                            raise MissingDataError("probe row is missing its month", line=line_number)
                        check_month(position.month, cutoff)
                    positions.append(position)
                except json.JSONDecodeError as error:
                    raise DataError("invalid JSON in probe set", path=str(source), line=line_number,
                                    detail=str(error)) from error
                except ChessEvalError as error:
                    error.context.setdefault("path", str(source))
                    error.context.setdefault("line", line_number)
                    raise
    except (OSError, UnicodeError) as error:
        raise DataError("could not read probe set", path=str(source), detail=str(error)) from error
    if not positions:
        raise MissingDataError("probe set is empty", path=str(source))
    return tuple(positions)


def split_probe_set(
    positions: Sequence[ProbePosition],
    *,
    probe_size: int = 1000,
    validation_size: int = 0,
    fidelity_size: int = 0,
) -> dict[str, tuple[ProbePosition, ...]]:
    sizes = {"probe": probe_size, "validation": validation_size, "fidelity": fidelity_size}
    if any(not isinstance(size, int) or isinstance(size, bool) for size in sizes.values()):
        raise ValidationError("split sizes must be integers", **sizes)
    if probe_size <= 0 or validation_size < 0 or fidelity_size < 0:
        raise ValidationError("split sizes must be positive for probe and non-negative otherwise", **sizes)
    required = sum(sizes.values())
    if len(positions) < required:
        raise MissingDataError("not enough positions for disjoint split", required=required, available=len(positions))
    result = {"probe": tuple(positions[:probe_size])}
    if validation_size:
        result["validation"] = tuple(positions[probe_size:probe_size + validation_size])
    if fidelity_size:
        result["fidelity"] = tuple(positions[-fidelity_size:])
    return result

