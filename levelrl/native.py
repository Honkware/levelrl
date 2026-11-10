from __future__ import annotations

import ctypes
import json
import os
import platform
from pathlib import Path

import chess
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent / "native"
LIBRARY = ROOT / "target" / "release" / ("liblevelrl_native.dylib" if platform.system() == "Darwin" else "liblevelrl_native.so")
_lib = None
_available = None


def load():
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(str(LIBRARY))
        lib.levelrl_features.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_ubyte)]
        lib.levelrl_features.restype = ctypes.c_int
        _lib = lib
    return _lib


def available() -> bool:
    global _available
    if _available is None:
        try:
            load()
            _available = True
        except OSError:
            _available = False
        runs = ROOT.parent / "runs"
        runs.mkdir(exist_ok=True)
        tmp = runs / f".backend.{os.getpid()}.tmp"
        tmp.write_text(json.dumps({"features": "rust" if _available else "python"}))
        tmp.replace(runs / "backend.json")
    return _available


def features(board: chess.Board) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = np.zeros(64 * 12, dtype=np.float32)
    mask = np.zeros(4352, dtype=np.uint8)
    fen = board.fen(en_passant="fen").encode()
    status = load().levelrl_features(fen, tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                     mask.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))
    if status:
        raise ValueError(f"native feature extraction failed with status {status}: {board.fen()}")
    return torch.from_numpy(tokens.reshape(64, 12)), torch.from_numpy(mask).bool()
