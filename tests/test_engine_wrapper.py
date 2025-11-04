import chess
from src.engine.engine_wrapper import EngineWrapper

def test_engine_returns_lines():
    b = chess.Board()
    with EngineWrapper("stockfish", depth=8, multipv=2) as eng:
        lines = eng.analyse(b)
        assert len(lines) >= 1
        assert lines[0].move in b.legal_moves
