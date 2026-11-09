from __future__ import annotations

from collections import deque

import chess
import torch

from maia3.dataset import get_historical_tokens, get_legal_moves_mask, tokenize_board
from maia3.model_registry import resolve_checkpoint_path
from maia3.uci import load_model, parse_args
from maia3.utils import get_all_possible_moves, mirror_move

MOVES = get_all_possible_moves()
MOVE_IDX = {m: i for i, m in enumerate(MOVES)}
REVISION = "a107d6ceb7b298cb04ae1da4edffe2939858b894"


def load(device: str = "cpu", ckpt: str | None = None) -> tuple:
    cfg = parse_args(["--model", "maia3-79m", "--device", device, "--revision", REVISION]
                     + (["--checkpoint-path", ckpt] if ckpt else []))
    if cfg.checkpoint_path is None:
        cfg.checkpoint_path = resolve_checkpoint_path(cfg.model_spec, revision=REVISION)
    return cfg, load_model(cfg)


def tokens(cfg, board: chess.Board, history: bool) -> torch.Tensor:
    # Same history the UCI engine builds: the replayed game when history is on, else the current board
    # alone, which get_historical_tokens pads by repeating.
    h = deque(maxlen=cfg.history)
    if history:
        replay = board.copy()
        moves = []
        while replay.move_stack and len(moves) < cfg.history - 1:
            moves.append(replay.pop())
        h.append(tokenize_board(replay))
        for mv in reversed(moves):
            replay.push(mv)
            h.append(tokenize_board(replay))
    else:
        h.append(tokenize_board(board))
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
    mask = torch.stack([get_legal_moves_mask(b, MOVE_IDX) for b in boards]).to(dev)
    se = torch.tensor(self_elo, dtype=torch.long, device=dev)
    oe = torch.tensor(oppo_elo, dtype=torch.long, device=dev)
    logits, value, _ = model(x, se, oe)
    return logits.float().masked_fill(~mask, float("-inf")), value.float()
