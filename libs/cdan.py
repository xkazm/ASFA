# CDAN Loss
# code from https://github.com/thuml/Transfer-Learning-Library

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

sys.path.append('.')

from .grl import WarmStartGradientReverseLayer
from .loss import Entropy


class MultiLinearMap(nn.Module):
    """
    Multi linear mapping

    Shape:
        - f: (batchsize, F)
        - g: (batchsize, C)
        - Outputs: (batchsize, F*C)
    """

    def __init__(self):
        super(MultiLinearMap, self).__init__()

    def forward(self,
                f: torch.Tensor,
                g: torch.Tensor) -> torch.Tensor:
        batchsize = f.size(0)
        output = torch.bmm(g.unsqueeze(2), f.unsqueeze(1))
        return output.view(batchsize, -1)


class RandomizedMultiLinearMap(nn.Module):
    """
    Random Multi Linear Mapping

    Shape:
        - f: (batchsize, F)
        - g: (batchsize, C)
        - Outputs: (batchsize, output_dim)
    """

    def __init__(self,
                 features_dim: int,
                 num_classes: int,
                 output_dim: int):
        super(RandomizedMultiLinearMap, self).__init__()
        self.Rf = torch.randn(features_dim, output_dim)
        self.Rg = torch.randn(num_classes, output_dim)
        self.output_dim = output_dim

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        f = torch.mm(f, self.Rf.to(f.device))
        g = torch.mm(g, self.Rg.to(g.device))
        output = torch.mul(f, g) / np.sqrt(float(self.output_dim))
        return output


class ConditionalDomainAdversarialLoss(nn.Module):
    def __init__(self,
                 domain_discriminator: nn.Module,
                 grl: nn.Module,
                 feature_dim: int,
                 n_class: int,
                 randomized_dim: int,
                 entropy_conditioning: Optional[bool] = False,
                 reduction: Optional[str] = 'mean'):
        super(ConditionalDomainAdversarialLoss, self).__init__()
        self.domain_discriminator = domain_discriminator
        self.entropy_conditioning = entropy_conditioning
        self.grl = grl

        self.map = RandomizedMultiLinearMap(feature_dim, n_class, randomized_dim)

        self.bce = lambda input, target, weight: F.binary_cross_entropy(input, target, weight=weight,
                                                                        reduction=reduction) if self.entropy_conditioning \
            else F.binary_cross_entropy(input, target, reduction=reduction)

    def forward(self,
                fs: torch.Tensor,
                gs: torch.Tensor,
                ft: torch.Tensor,
                gt: torch.Tensor,
                iter_num: int) -> torch.Tensor:
        f = torch.cat((fs, ft), dim=0)
        g = torch.cat((gs, gt), dim=0)
        g = F.softmax(g, dim=1).detach()
        h = self.grl(self.map(f, g), iter_num)
        d = self.domain_discriminator(h)
        d_label = torch.cat((
            torch.ones((gs.size(0), 1)).to(gs.device),
            torch.zeros((gt.size(0), 1)).to(gt.device),
        ))
        weight = 1.0 + torch.exp(-Entropy(g))
        batch_size = f.size(0)
        weight = weight / torch.sum(weight) * batch_size
        return self.bce(d, d_label, weight.view_as(d))

    def w_forward(self,
                  fs: torch.Tensor,
                  gs: torch.Tensor,
                  ft: torch.Tensor,
                  gt: torch.Tensor,
                  iter_num: int,
                  ys: torch.Tensor,
                  yt: torch.Tensor) -> torch.Tensor:
        f = torch.cat((fs, ft), dim=0)
        g = torch.cat((gs, gt), dim=0)
        g = F.softmax(g, dim=1).detach()
        h = self.grl(self.map(f, g), iter_num)
        d = self.domain_discriminator(h)
        d_label = torch.cat((
            torch.ones((gs.size(0), 1)).to(gs.device),
            torch.zeros((gt.size(0), 1)).to(gt.device),
        ))
        weight = 1.0 + torch.exp(-Entropy(g))
        batch_size = f.size(0)
        weight = weight / torch.sum(weight) * batch_size

        y_set = torch.unique(ys)
        da = torch.zeros(len(y_set))
        k = 0
        y = torch.cat((ys, yt), dim=0).view(-1)
        l = (d > 0.5).long()
        for label in y_set:
            index = torch.nonzero(y.eq(label)).view(-1)
            dak = torch.sum(torch.eq(l[index], d_label[index]))
            da[k] = 2 * (1 - 2 * dak / len(index))
            k += 1
        daf = da.mean()

        return self.bce(d, d_label, weight.view_as(d)), daf
