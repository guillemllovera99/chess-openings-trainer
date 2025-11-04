from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import chess
from . import pgn_loader
from .embedded_openings import OPENINGS as EMBEDDED

@dataclass
class OpeningMatch:
    eco: str
    name: str
    matched_moves: int

class OpeningExplorer:
    """
    Opening identification via longest-prefix match on SAN sequences.
    Loads (a) embedded Python dict and (b) optional JSON/PGN books.
    """
    def __init__(self, eco_json_path: Optional[str] = "data/openings/eco_small.json"):
        self.trie: Dict[Tuple[str, ...], Tuple[str, str]] = dict(EMBEDDED)
        self._maybe_load_json(eco_json_path)

    def _maybe_load_json(self, path: Optional[str]):
        if not path:
            return
        p = Path(path)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                for seq, (eco, name) in data.items():
                    key = tuple(seq.split())
                    self.trie.setdefault(key, (eco, name))
            except Exception:
                pass

    def load_eco_pgn(self, eco_pgn: str):
        for key, (eco, name) in pgn_loader.iter_opening_tuples(eco_pgn):
            self.trie.setdefault(key, (eco, name))

    def identify(self, board: chess.Board) -> Optional[OpeningMatch]:
        san_moves: List[str] = []
        tmp = chess.Board()
        for mv in board.move_stack:
            san_moves.append(tmp.san(mv))
            tmp.push(mv)

        best: Optional[Tuple[str, str, int]] = None
        for key, (eco, name) in self.trie.items():
            L = len(key)
            if tuple(san_moves[:L]) == key:
                if best is None or L > best[2]:
                    best = (eco, name, L)
        if best:
            eco, name, L = best
            return OpeningMatch(eco=eco, name=name, matched_moves=L)
        return None
