"""The PyTorch model and its quantisation to the integer network (see context.md)."""

import numpy as np
import torch
from torch import nn

from .features import EMPTY, HIDDEN, INPUTS, flip
from .vknn import QA, QB, QNet

LIMIT = 1.98  # 1.98 * 255 = 504.9 <= 505 and 1.98 * 64 = 126.7 <= 127


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ft = nn.EmbeddingBag(INPUTS + 1, HIDDEN, mode="sum", padding_idx=EMPTY)
        self.ft_bias = nn.Parameter(torch.zeros(HIDDEN))
        self.out = nn.Linear(2 * HIDDEN, 1)
        with torch.no_grad():
            self.ft.weight.normal_(0.0, 0.05)
            self.ft.weight[EMPTY].zero_()
        self.register_buffer("flipped", torch.tensor([flip(i) for i in range(INPUTS + 1)], dtype=torch.long))

    def forward(self, features: torch.Tensor, stm: torch.Tensor) -> torch.Tensor:
        """Score for the side to move, in units of 400 centipawns."""
        white = self.ft(features) + self.ft_bias
        black = self.ft(self.flipped[features]) + self.ft_bias
        black_to_move = (stm.reshape(-1, 1) != 0)
        us = torch.where(black_to_move, black, white)
        them = torch.where(black_to_move, white, black)
        hidden = torch.cat([us.clamp(0.0, 1.0) ** 2, them.clamp(0.0, 1.0) ** 2], dim=1)
        return self.out(hidden).squeeze(1)

    @torch.no_grad()
    def clip_(self) -> None:
        for p in (self.ft.weight, self.ft_bias, self.out.weight):
            p.clamp_(-LIMIT, LIMIT)


def _round(values: torch.Tensor, scale: float) -> np.ndarray:
    return np.rint(values.detach().cpu().double().numpy() * scale)


def quantise(net: Net) -> QNet:
    """Float weights to the integer network: features by QA, output weights by QB, output bias by QA * QB."""
    ft_w = _round(net.ft.weight[:INPUTS], QA).astype(np.int16)
    ft_b = _round(net.ft_bias, QA).astype(np.int16)
    out_w = _round(net.out.weight[0], QB).astype(np.int16)
    out_b = int(_round(net.out.bias[0], QA * QB))
    return QNet(ft_w, ft_b, out_w, out_b)
