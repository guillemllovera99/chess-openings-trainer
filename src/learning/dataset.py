from __future__ import annotations
from typing import Optional, Tuple
import os
import numpy as np

class NPZReplayBuffer:
    """
    Simple on-disk circular buffer using .npz files.
    Stores (X: float32 [N, C, 8, 8], val: float32 [N,1], pol: int64 [N])
    """
    def __init__(self, path: str = "data/cache/replay_buffer.npz", capacity: int = 50000):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.capacity = capacity
        if not os.path.exists(self.path):
            np.savez_compressed(self.path,
                                X=np.zeros((0, 18, 8, 8), dtype=np.float32),
                                V=np.zeros((0, 1), dtype=np.float32),
                                P=np.zeros((0,), dtype=np.int64))

    def append(self, x: np.ndarray, v: float, p: int):
        data = np.load(self.path)
        X = data["X"]; V = data["V"]; P = data["P"]
        X_new = np.concatenate([X, x[None, ...]], axis=0)
        V_new = np.concatenate([V, np.array([[v]], dtype=np.float32)], axis=0)
        P_new = np.concatenate([P, np.array([p], dtype=np.int64)], axis=0)
        # enforce capacity
        if X_new.shape[0] > self.capacity:
            off = X_new.shape[0] - self.capacity
            X_new = X_new[off:]; V_new = V_new[off:]; P_new = P_new[off:]
        np.savez_compressed(self.path, X=X_new, V=V_new, P=P_new)

    def sample(self, batch_size: int = 64) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        data = np.load(self.path)
        X = data["X"]; V = data["V"]; P = data["P"]
        n = X.shape[0]
        if n == 0:
            return None
        idx = np.random.choice(n, size=min(batch_size, n), replace=False)
        return X[idx], V[idx], P[idx]
