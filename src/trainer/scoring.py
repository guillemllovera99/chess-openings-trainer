from __future__ import annotations
from typing import Optional

def cp_loss(best_cp_before: Optional[float], user_cp_after: Optional[float]) -> Optional[float]:
    """
    Centipawn loss estimate = eval(best line before move) - eval(after user's move).
    Both values are from the POV of the side-to-move before the move.
    """
    if best_cp_before is None or user_cp_after is None:
        return None
    return best_cp_before - user_cp_after

def verdict(is_best: bool, loss: Optional[float]) -> str:
    if is_best:
        return "✓ Correct"
    if loss is None:
        return "…"
    if loss < 30:
        return "≈ Acceptable"
    if loss < 80:
        return "△ Inaccuracy"
    if loss < 200:
        return "⨯ Mistake"
    return "⨂ Blunder"
