from typing import Tuple, Optional
import math
import chess

# Minimal alpha-beta fallback for when a UCI engine is not available.
# It's intentionally simple (material + mobility) and not meant to replace Stockfish.

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

def evaluate(board: chess.Board) -> int:
    if board.is_checkmate():
        return -100000 if board.turn else 100000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    score = 0
    for pt, val in PIECE_VALUES.items():
        score += len(board.pieces(pt, chess.WHITE)) * val
        score -= len(board.pieces(pt, chess.BLACK)) * val
    # light mobility term
    moves = board.legal_moves.count()
    score += moves if board.turn else -moves
    return score

def _search(board: chess.Board, depth: int, alpha: int, beta: int) -> int:
    if depth == 0 or board.is_game_over():
        return evaluate(board)
    best = -math.inf
    for move in board.legal_moves:
        board.push(move)
        val = -_search(board, depth - 1, -beta, -alpha)
        board.pop()
        if val > best:
            best = val
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return int(best)

def best_move(board: chess.Board, depth: int = 3) -> Tuple[chess.Move, int]:
    best_mv: Optional[chess.Move] = None
    best_val = -math.inf
    for move in board.legal_moves:
        board.push(move)
        val = -_search(board, depth - 1, -math.inf, math.inf)
        board.pop()
        if val > best_val:
            best_val = val
            best_mv = move
    return best_mv or chess.Move.null(), int(best_val)
