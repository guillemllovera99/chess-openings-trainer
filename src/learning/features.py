from __future__ import annotations
from typing import Tuple
import numpy as np
import chess

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

def board_to_planes(board: chess.Board) -> np.ndarray:
    """
    Encode board to planes: 12 piece planes + side-to-move + 4 castling + ep file (8)
    Total channels = 12 + 1 + 4 + 8 = 25 (we'll keep 18 by omitting ep file detail for now)
    Here we implement: 12 piece planes + side-to-move + 4 castling + empty (padding) = 18
    Shape: (18, 8, 8), dtype=float32
    """
    planes = np.zeros((18, 8, 8), dtype=np.float32)
    idx = 0
    for color in [chess.WHITE, chess.BLACK]:
        for pt in PIECE_TYPES:
            bb = board.pieces(pt, color)
            for sq in bb:
                r, c = divmod(sq, 8)
                planes[idx, 7 - r, c] = 1.0
            idx += 1
    # side to move
    planes[idx] = (1.0 if board.turn == chess.WHITE else 0.0)
    idx += 1
    # castling rights: WK, WQ, BK, BQ
    planes[idx] = (1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0); idx += 1
    planes[idx] = (1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0); idx += 1
    planes[idx] = (1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0); idx += 1
    planes[idx] = (1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0); idx += 1
    # pad remaining channels to 18 with zeros automatically (they are already zero)
    return planes

def targets_from_engine(cp: float | None, multipv_cps: list[float] | None = None) -> Tuple[float, int]:
    """
    Build supervision targets from engine eval.
    value: tanh(cp/800) in [-1,1]; None -> 0
    policy_index: naive "best to-square" index 0..63 (fallback 0)
    """
    import math
    v = 0.0 if cp is None else math.tanh((cp/100.0)/8.0)
    # We cannot map full move space cheaply here; use a dummy index (e.g., center preference)
    # Upstream code can override with actual move to-square index if desired.
    pol = 27  # e4 square index as placeholder
    return float(v), int(pol)
