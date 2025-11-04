from __future__ import annotations
from datetime import datetime
from pathlib import Path
import chess
import chess.pgn

def save_game(board: chess.Board, path: str | None = None):
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "Openings Trainer Session"
    game.headers["Date"] = datetime.utcnow().strftime("%Y.%m.%d")
    out = str(path or f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pgn")
    with open(out, "w", encoding="utf-8") as f:
        print(game, file=f)
    return out
