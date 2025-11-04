import torch
import torch.nn as nn
import torch.nn.functional as F

class ValuePolicyNet(nn.Module):
    """
    Tiny CNN over 8x8 planes.
    Inputs: (B, C, 8, 8)
    Outputs:
      - value: (B, 1) in [-1, 1]
      - policy_logits: (B, 64) naive square-based logits (used as a soft hint)
    """
    def __init__(self, in_channels: int = 18):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.head_val = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        self.head_pol = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 128),
            nn.ReLU(),
            nn.Linear(128, 64)  # naive policy over to-squares
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        v = self.head_val(x)
        p = self.head_pol(x)
        return v, p
