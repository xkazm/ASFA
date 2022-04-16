# MLP model for used

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from sklearn.metrics import accuracy_score, cohen_kappa_score

import sys

sys.path.append('.')
from .utils import temperature_normalization


class MLPBase(nn.Module):
    """
    The MLP based feature extraction module

    Parameters:
        - **input_dim** (int): num of input features
        - **fea_dim** Optional(int): num of output features
    """

    def __init__(self,
                 input_dim: int,
                 fea_dim: Optional[int] = None):
        super(MLPBase, self).__init__()
        # layer1
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ELU()
        )
        # layer2
        if fea_dim is None:
            fea_dim = input_dim
        self.layer2 = nn.Sequential(
            nn.Linear(input_dim, fea_dim),
            nn.BatchNorm1d(fea_dim),
            nn.ELU()
        )

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        return self.layer2(x)


class feat_bottleneck(nn.Module):
    """
    The bottleneck module

    Parameters:
        - **input_dim** (int): the dim of input feature
        - **bottleneck_dim** (int): the dim of bottleneck module output
        - **t** Optional(float): the temperature factor
    """

    def __init__(self,
                 input_dim: int,
                 bottleneck_dim: int,
                 t: Optional[float] = 0.1):
        super(feat_bottleneck, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ELU()
        )
        self.t = t

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        return temperature_normalization(x, self.t)


class feat_classifier(nn.Module):
    """
    The classifier module

    Parameters:
        - **input_dim** (int): the dim of input feature
        - **n_class** (int): the num of task classes
    """

    def __init__(self,
                 input_dim: int,
                 n_class: int):
        super(feat_classifier, self).__init__()
        self.fc = weight_norm(nn.Linear(input_dim, n_class), name="weight")

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def cal_acc(x: torch.Tensor,
            y: torch.Tensor,
            netF: nn.Module,
            netB: nn.Module,
            netC: nn.Module) -> Tuple[float, float]:
    """
    calculate the accuracy
    :param x: input
    :param y: ground truth label
    :param netF: feature extraction module
    :param netB: bottleneck module
    :param netC: classifier module
    :return: accuracy
    """
    pred = netC(netB(netF(x)))
    pred = nn.Softmax(dim=1)(pred).cpu()
    return accuracy_score(y.cpu(), pred.argmax(dim=1)), cohen_kappa_score(y.cpu(), pred.argmax(dim=1))
