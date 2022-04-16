# loss for JAN
# code from https://github.com/thuml/Transfer-Learning-Library
from typing import Optional, Sequence
import torch
import torch.nn as nn

import sys

sys.path.append('.')
from .dan import _update_index_matrix

__all__ = ['JointMultipleKernelMaximumMeanDiscrepancy']


class JointMultipleKernelMaximumMeanDiscrepancy(nn.Module):
    """
    The Joint Multiple Kernel Maximum Mean Discrepancy (JMMD) used in JAN
    """

    def __init__(self, kernels: Sequence[Sequence[nn.Module]], linear: Optional[bool] = True):
        super(JointMultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        batch_size = int(z_s[0].size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s[0].device)

        kernel_matrix = torch.ones_like(self.index_matrix)
        for layer_z_s, layer_z_t, layer_kernels in zip(z_s, z_t, self.kernels):
            layer_features = torch.cat([layer_z_s, layer_z_t], dim=0)
            kernel_matrix *= sum(
                [kernel(layer_features) for kernel in layer_kernels])  # Add up the matrix of each kernel

        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)
        return loss
