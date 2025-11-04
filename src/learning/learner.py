from __future__ import annotations
from typing import Optional
import os
import torch
import torch.nn as nn
import torch.optim as optim

from models.nn.value_policy_net import ValuePolicyNet
from src.learning.dataset import NPZReplayBuffer

class OnlineLearner:
    """
    Minimal online learner: after moves, append (x, v, p) and do a few optimizer steps.
    """
    def __init__(self,
                 model_path: str = "models/nn/checkpoints/latest.ckpt",
                 opt_path: str = "models/nn/optimizer_state/latest.opt",
                 lr: float = 1e-3,
                 device: str = "cpu"):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(opt_path), exist_ok=True)
        self.model_path = model_path
        self.opt_path = opt_path
        self.device = torch.device(device)
        self.model = ValuePolicyNet(in_channels=18).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # try load
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            except Exception:
                pass
        if os.path.exists(self.opt_path):
            try:
                self.optimizer.load_state_dict(torch.load(self.opt_path, map_location=self.device))
            except Exception:
                pass
        self.buffer = NPZReplayBuffer()

    def step_after_move(self, x_np, value_target: float, policy_index: int, updates: int = 1, batch_size: int = 64):
        # append sample
        self.buffer.append(x_np, value_target, policy_index)
        # train
        self.model.train()
        loss_total = 0.0
        for _ in range(updates):
            batch = self.buffer.sample(batch_size=batch_size)
            if batch is None:
                break
            X, V, P = batch
            X_t = torch.from_numpy(X).to(self.device)
            V_t = torch.from_numpy(V).to(self.device)
            P_t = torch.from_numpy(P).to(self.device)
            v_pred, p_logits = self.model(X_t)
            v_loss = torch.mean((v_pred - V_t)**2)
            p_loss = torch.nn.functional.cross_entropy(p_logits, P_t)
            loss = v_loss + 0.1 * p_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
        # save checkpoints
        torch.save(self.model.state_dict(), self.model_path)
        torch.save(self.optimizer.state_dict(), self.opt_path)
        return loss_total

    def predict_policy_value(self, x_np):
        self.model.eval()
        with torch.no_grad():
            import numpy as np
            X_t = torch.from_numpy(x_np[None, ...]).to(self.device)
            v, p = self.model(X_t)
            return float(v.item()), p.cpu().numpy()[0]
