from __future__ import annotations
from typing import Optional, Tuple

BAR_WIDTH = 40

def _score_to_winprob(cp: Optional[float], mate: Optional[int]) -> float:
    """
    Convert engine evaluation to a rough win probability for the side to move.
    - If mate is present: clamp to 0 or 1 depending on sign.
    - If cp is present: logistic mapping similar to Elo win prob.
    Returns p in [0,1] for side to move.
    """
    if mate is not None:
        # Positive mate means side-to-move mates in N
        return 1.0 if mate > 0 else 0.0
    if cp is None:
        return 0.5
    # cp in centipawns; scale controls steepness. 600 is a common heuristic.
    import math
    p = 1.0 / (1.0 + 10 ** (-(cp / 400.0)))
    return max(0.0, min(1.0, p))

def render_eval_bar(cp: Optional[float], mate: Optional[int]) -> Tuple[str, str]:
    """
    Render a text bar with width BAR_WIDTH and a numeric label.
    Returns (bar, label). Label is like '+0.65' or '#3' (mate).
    """
    if mate is not None:
        label = f"#{abs(mate)}" if mate != 0 else "#"
        p = _score_to_winprob(None, mate)
    else:
        label = f"{(cp or 0.0)/100.0:+.2f}"
        p = _score_to_winprob(cp, None)

    filled = int(round(p * BAR_WIDTH))
    empty = BAR_WIDTH - filled
    bar = f"White  {'█'*filled}{'░'*empty}  Black"
    return bar, label
