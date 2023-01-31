# DANN Loss
# code from https://github.com/thuml/Transfer-Learning-Library

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

sys.path.append('.')

from .grl import WarmStartGradientReverseLayer


class DomainAdversarialLoss(nn.Module):
    def __init__(self,
                 domain_discriminator: nn.Module,
                 grl: nn.Module,
                 reduction: Optional[str] = 'mean'):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = grl
        self.domain_discriminator = domain_discriminator
        self.bce = lambda input, target: \
            F.binary_cross_entropy(input, target, reduction=reduction)

    def forward(self,
                fs: torch.Tensor,
                ft: torch.Tensor,
                iter_num: int) -> torch.Tensor:
        f = self.grl(torch.cat((fs, ft), dim=0), iter_num)
        d = self.domain_discriminator(f)
        ds, dt = d.chunk(2, dim=0)
        d_label_s = torch.ones((fs.size(0), 1)).to(fs.device)
        d_label_t = torch.zeros((ft.size(0), 1)).to(ft.device)

        return 0.5 * (self.bce(ds, d_label_s) + self.bce(dt, d_label_t))
