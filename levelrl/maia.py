from __future__ import annotations

from collections import deque

import chess
import torch

from maia3.dataset import PIECE_MAP, get_historical_tokens
from maia3.model_registry import resolve_checkpoint_path
from maia3.uci import load_model, parse_args
from maia3.utils import get_all_possible_moves, mirror_move

from levelrl import native

MOVES = get_all_possible_moves()
MOVE_IDX = {m: i for i, m in enumerate(MOVES)}
REVISION = "a107d6ceb7b298cb04ae1da4edffe2939858b894"


def _uci_key(uci: str) -> int:
    promotion = chess.PIECE_SYMBOLS.index(uci[4]) if len(uci) == 5 else 0
    return (chess.parse_square(uci[:2]) << 10) | (chess.parse_square(uci[2:4]) << 4) | promotion


def _move_key(move: chess.Move) -> int:
    return (move.from_square << 10) | (move.to_square << 4) | (move.promotion or 0)


_WHITE_IDX = {_uci_key(m): i for m, i in MOVE_IDX.items()}
_BLACK_IDX = {_uci_key(mirror_move(m)): i for m, i in MOVE_IDX.items()}


def load(device: str = "cpu", ckpt: str | None = None) -> tuple:
    cfg = parse_args(["--model", "maia3-79m", "--device", device, "--revision", REVISION]
                     + (["--checkpoint-path", ckpt] if ckpt else []))
    if cfg.checkpoint_path is None:
        cfg.checkpoint_path = resolve_checkpoint_path(cfg.model_spec, revision=REVISION)
    return cfg, load_model(cfg)


def encode_board(board: chess.Board) -> torch.Tensor:
    out = torch.zeros((64, 12), dtype=torch.float)
    mirror = board.turn == chess.BLACK
    for square, piece in board.piece_map().items():
        color = piece.color
        if mirror:
            square = chess.square_mirror(square)
            color = not color
        token = PIECE_MAP[piece.piece_type] + (6 if color == chess.BLACK else 0)
        out[square, token - 1] = 1
    return out


def replay_features(cfg, moves: list[str]) -> tuple[chess.Board, torch.Tensor, torch.Tensor]:
    b = chess.Board()
    start = max(0, len(moves) - cfg.history + 1)
    for mv in moves[:start]:
        b.push_uci(mv)
    h = deque(maxlen=cfg.history)
    h.append(encode_board(b))
    for mv in moves[start:]:
        b.push_uci(mv)
        h.append(encode_board(b))
    if native.available():
        current, mask = native.features(b)
        h[-1] = current
    else:
        mask = legal(b)
    x = get_historical_tokens(h, cfg, base=0.0, inc=0.0, clk_left_before=0.0, clk_ponder=0.0)
    return b, x, mask


def replay(cfg, moves: list[str]) -> tuple[chess.Board, torch.Tensor]:
    b, x, _ = replay_features(cfg, moves)
    return b, x


def tokens(cfg, board: chess.Board, history: bool) -> torch.Tensor:
    # Same history the UCI engine builds: the replayed game when history is on, else the current board
    # alone, which get_historical_tokens pads by repeating.
    h = deque(maxlen=cfg.history)
    if history:
        replay = board.copy()
        moves = []
        while replay.move_stack and len(moves) < cfg.history - 1:
            moves.append(replay.pop())
        h.append(encode_board(replay))
        for mv in reversed(moves):
            replay.push(mv)
            h.append(encode_board(replay))
    else:
        h.append(encode_board(board))
    return get_historical_tokens(h, cfg, base=0.0, inc=0.0, clk_left_before=0.0, clk_ponder=0.0)


def to_move(board: chess.Board, idx: int) -> chess.Move:
    uci = MOVES[idx]
    return chess.Move.from_uci(uci if board.turn == chess.WHITE else mirror_move(uci))


@torch.no_grad()
def forward(cfg, model, boards: list[chess.Board], self_elo: list[int], oppo_elo: list[int],
            history: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    # Masked move logits (side to move's frame, illegal = -inf) and WDL logits as [loss, draw, win].
    dev = cfg.device
    x = torch.stack([tokens(cfg, b, history) for b in boards]).to(dev)
    mask = torch.stack([legal(b) for b in boards]).to(dev)
    se = torch.tensor(self_elo, dtype=torch.long, device=dev)
    oe = torch.tensor(oppo_elo, dtype=torch.long, device=dev)
    logits, value, _ = model(x, se, oe)
    return logits.float().masked_fill(~mask, float("-inf")), value.float()


class Hist:
    # Incremental history tokens for one game. Profiling on the 4090 box put full replay encoding at 0.65 ms a
    # position against 0.25 ms for the forward, so each position is tokenized once, as it is reached.
    def __init__(self, cfg, board: chess.Board):
        self.cfg, self.h = cfg, deque(maxlen=cfg.history)
        self.use_native = native.available()
        self.native_board = native.NativeBoard(board) if self.use_native else None
        replay, moves = board.copy(), []
        while replay.move_stack and len(moves) < cfg.history - 1:
            moves.append(replay.pop())
        self.h.append(encode_board(replay))
        for mv in reversed(moves):
            replay.push(mv)
            self.h.append(encode_board(replay))
        if self.use_native:
            current, self.mask = self.native_board.features()
            self.h[-1] = current.clone()
        else:
            self.mask = legal(board)

    def push(self, board: chess.Board) -> None:
        if self.use_native:
            current, self.mask = self.native_board.push_features(board.move_stack[-1].uci())
            self.h.append(current.clone())
        else:
            self.h.append(encode_board(board))
            self.mask = legal(board)

    def add(self, tok, mask=None) -> None:
        # A position encoded elsewhere (an arena worker), as numpy arrays.
        self.h.append(torch.from_numpy(tok))
        if mask is not None:
            self.mask = torch.from_numpy(mask)

    def legal_mask(self) -> torch.Tensor:
        return self.mask

    def tokens(self) -> torch.Tensor:
        return get_historical_tokens(self.h, self.cfg, base=0.0, inc=0.0, clk_left_before=0.0, clk_ponder=0.0)


def legal(board: chess.Board) -> torch.Tensor:
    out = torch.zeros(len(MOVES), dtype=torch.bool)
    indices = _WHITE_IDX if board.turn == chess.WHITE else _BLACK_IDX
    for move in board.legal_moves:
        i = indices.get(_move_key(move))
        if i is not None:
            out[i] = True
    return out


def run(model, x: torch.Tensor, mask: torch.Tensor, se, oe, dev) -> tuple[torch.Tensor, torch.Tensor]:
    # Forward on prebuilt tokens and masks, bf16 on cuda. Returns masked fp32 move logits and value logits.
    se = torch.as_tensor(se, dtype=torch.long, device=dev)
    oe = torch.as_tensor(oe, dtype=torch.long, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(dev).startswith("cuda")):
        logits, value, _ = model(x.to(dev, non_blocking=True), se, oe)
    return logits.float().masked_fill(~mask.to(dev, non_blocking=True), float("-inf")), value.float()
