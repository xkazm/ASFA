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

    def w_forward(self,
                  fs: torch.Tensor,
                  ft: torch.Tensor,
                  iter_num: int) -> Tuple[torch.Tensor, float]:
        f = self.grl(torch.cat((fs, ft), dim=0), iter_num)
        d = self.domain_discriminator(f)
        ds, dt = d.chunk(2, dim=0)
        d_label_s = torch.ones((fs.size(0), 1)).to(fs.device)
        d_label_t = torch.zeros((ft.size(0), 1)).to(ft.device)

        ls = nn.Softmax(dim=1)(ds).argmax(dim=1)
        lt = nn.Softmax(dim=1)(dt).argmax(dim=1)

        d_error_s = len(torch.nonzero(ls.ne(0)).view(-1))
        d_error_t = len(torch.nonzero(lt.ne(1)).view(-1))

        # print("d_error_t", d_error_t)

        da = (d_error_s + d_error_t) / (fs.size(0) + ft.size(0))
        da = 2 * (1 - 2 * da)
        # print('dam', da)

        return 0.5 * (self.bce(ds, d_label_s) + self.bce(dt, d_label_t)), da
