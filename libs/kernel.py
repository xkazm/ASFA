# kernel function used in DAN
# code from https://github.com/thuml/Transfer-Learning-Library
from typing import Optional
import torch
import torch.nn as nn

__all__ = ['GaussianKernel']


class GaussianKernel(nn.Module):
    """
    Gaussian Kernel Matrix

    Parameters:
        - **sigma** Optional(float): bandwidth
        - **track_running_stats** Optional(bool): whether to track the running mean of sigma^2
        - **alpha** Optional(float): magnitude of track_running_stats of sigma^2
    """

    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        super(GaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)

        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())

        return torch.exp(-l2_distance_square / (2 * self.sigma_square))
