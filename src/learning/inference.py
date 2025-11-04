from __future__ import annotations
import numpy as np
import chess

def choose_square_from_policy(board: chess.Board, policy_logits: np.ndarray) -> int:
    """
    Naive helper: picks a 'to-square' (0..63) with softmax sampling.
    This is not a full move policy; it's a lightweight hinting signal.
    """
    import math
    z = policy_logits - np.max(policy_logits)
    p = np.exp(z) / np.sum(np.exp(z))
    return int(np.random.choice(np.arange(64), p=p))
