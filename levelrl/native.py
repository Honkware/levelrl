from __future__ import annotations

import ctypes
import json
import os
import platform
import threading
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
        lib.levelrl_board_create.argtypes = [ctypes.c_char_p]
        lib.levelrl_board_create.restype = ctypes.c_void_p
        lib.levelrl_board_free.argtypes = [ctypes.c_void_p]
        lib.levelrl_board_push.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.levelrl_board_push.restype = ctypes.c_int
        lib.levelrl_board_features.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_ubyte)]
        lib.levelrl_board_features.restype = ctypes.c_int
        lib.levelrl_board_move_features.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
                                                   ctypes.POINTER(ctypes.c_ubyte)]
        lib.levelrl_board_move_features.restype = ctypes.c_int
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
        tmp = runs / f".backend.{os.getpid()}.{threading.get_ident()}.tmp"
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


class NativeBoard:
    def __init__(self, board: chess.Board):
        self.handle = load().levelrl_board_create(board.fen(en_passant="fen").encode())
        if not self.handle:
            raise ValueError(f"could not create native board: {board.fen()}")
        self.tokens = np.zeros(64 * 12, dtype=np.float32)
        self.mask = np.zeros(4352, dtype=np.uint8)

    def push_uci(self, uci: str):
        status = load().levelrl_board_push(self.handle, uci.encode())
        if status:
            raise ValueError(f"native move failed with status {status}: {uci}")

    def push_features(self, uci: str) -> tuple[torch.Tensor, torch.Tensor]:
        status = load().levelrl_board_move_features(self.handle, uci.encode(),
                                                   self.tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                   self.mask.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))
        if status:
            raise ValueError(f"native move/feature extraction failed with status {status}: {uci}")
        return torch.from_numpy(self.tokens.reshape(64, 12)), torch.from_numpy(self.mask).bool()

    def features(self) -> tuple[torch.Tensor, torch.Tensor]:
        status = load().levelrl_board_features(self.handle, self.tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                               self.mask.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))
        if status:
            raise ValueError(f"native feature extraction failed with status {status}")
        return torch.from_numpy(self.tokens.reshape(64, 12)), torch.from_numpy(self.mask).bool()

    def close(self):
        if self.handle:
            load().levelrl_board_free(self.handle)
            self.handle = None

    def __del__(self):
        self.close()
