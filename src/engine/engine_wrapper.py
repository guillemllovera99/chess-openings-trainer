import contextlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

import chess
import chess.engine

@dataclass
class EngineLine:
    move: chess.Move
    cp: Optional[float]           # centipawns from side-to-move perspective
    mate: Optional[int]           # mate in N (positive = mate for side-to-move)
    pv_san: str                   # principal variation in SAN

class EngineWrapper:
    """
    Thin wrapper around a UCI engine (e.g., Stockfish) to fetch MultiPV
    candidate lines with evaluations and SAN-converted PVs.
    """
    def __init__(self, engine_path: str, depth: int = 16, multipv: int = 3):
        self.engine_path = engine_path
        self.depth = depth
        self.multipv = max(1, multipv)
        self._proc: Optional[chess.engine.SimpleEngine] = None

    def __enter__(self):
        self._proc = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        try:
            # set a few safe defaults; ignore if unsupported
            self._proc.configure({"Threads": 2})
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._proc:
            with contextlib.suppress(Exception):
                self._proc.quit()
        self._proc = None

    def analyse(self, board: chess.Board) -> List[EngineLine]:
        """
        Returns a list of EngineLine for the top multipv moves.
        If engine returns fewer lines, we return what we have.
        """
        assert self._proc is not None, "Engine not started. Use 'with EngineWrapper(...):'"
        limit = chess.engine.Limit(depth=self.depth)
        info_list = self._proc.analyse(board, limit=limit, multipv=self.multipv)

        # python-chess returns either a dict (single) or list of dicts (multi)
        if isinstance(info_list, dict):
            info_list = [info_list]

        lines: List[EngineLine] = []
        for info in info_list:
            pv = info.get("pv", [])
            if not pv:
                continue
            top_move = pv[0]
            # Convert score to cp/mate from the perspective of the side to move
            score = info.get("score")
            cp_val: Optional[float] = None
            mate_val: Optional[int] = None
            if score is not None:
                pov = score.pov(board.turn)
                if pov.is_mate():
                    mate_val = pov.mate()
                    cp_val = None
                else:
                    cp = pov.score(mate_score=100000)
                    if cp is not None:
                        cp_val = float(cp)

            # Build SAN PV string
            tmp = board.copy(stack=False)
            pv_sans = []
            for m in pv[:12]:  # limit PV length
                if m not in tmp.legal_moves:
                    break
                pv_sans.append(tmp.san(m))
                tmp.push(m)
            lines.append(EngineLine(move=top_move, cp=cp_val, mate=mate_val, pv_san=" ".join(pv_sans)))
        # Sort best first by cp (higher better for side to move) and then by mate
        def key_fn(el: EngineLine):
            if el.mate is not None:
                # Prefer faster mates for side to move (smaller positive number is better)
                # We invert sign so that smaller positive mate is sorted first when descending.
                return (float('inf'), -abs(el.mate))
            return (el.cp if el.cp is not None else float('-inf'), 0)
        lines.sort(key=key_fn, reverse=True)
        return lines
