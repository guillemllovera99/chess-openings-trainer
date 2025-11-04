import chess
from src.learning.features import board_to_planes

def test_planes_shape():
    import numpy as np
    b = chess.Board()
    x = board_to_planes(b)
    assert x.shape == (18,8,8)
    assert x.dtype == np.float32
