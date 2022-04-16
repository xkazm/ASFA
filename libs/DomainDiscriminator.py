# Domain Discriminator
from typing import List, Dict
import torch.nn as nn
import torch


class DomainDiscriminator(nn.Module):
    """
    Domain Discriminator Module - 3 layers MLP
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int):
        super(DomainDiscriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
