# EEGNet Module
from typing import Optional, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score
import math


class EEGNetBase(nn.Module):
    """
    EEGNet module for feature extraction

    Parameters:
        - **channels** (int): the number of channels of input signals
        - **timepoints** (int): the number of time sample points of input signals
        - **kernelLength** (int): the kernel length of the time filter conv kernel
        - **numFilters** (int): the number of kernels
        - **bottleneck_dim** (int): the dimension of features
        - **dropoutRate** Optional(float): the dropout rate
    """

    def __init__(self,
                 channels: int,
                 timepoints: int,
                 kernelSize: int,
                 numFilters: int,
                 bottleneck_dim: int,
                 dropoutRate: Optional[float] = 0.5):
        super(EEGNetBase, self).__init__()

        self.channels = channels
        self.timepoints = timepoints
        self.kernelSize = kernelSize
        self.numFilters = numFilters
        self.bottleneck_dim = bottleneck_dim
        self.dropoutRate = dropoutRate

        self.block1 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            nn.ZeroPad2d((math.ceil(self.kernelSize / 2) - 1, self.kernelSize - math.ceil(self.kernelSize / 2), 0, 0)),
            nn.Conv2d(
                in_channels=1,
                out_channels=self.numFilters,
                kernel_size=(1, self.kernelSize),
                bias=False
            ),
            nn.BatchNorm2d(self.numFilters)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.numFilters,
                out_channels=self.numFilters,
                kernel_size=(self.channels, 1),
                groups=self.numFilters,
                bias=False
            ),
            nn.BatchNorm2d(self.numFilters),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.dropoutRate)
        )

        self.block3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(
                in_channels=self.numFilters,
                out_channels=self.numFilters * 2,
                kernel_size=(1, 16),
                groups=self.numFilters,
                bias=False
            ),
            nn.Conv2d(
                in_channels=self.numFilters * 2,
                out_channels=self.numFilters * 2,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(self.numFilters * 2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropoutRate)
        )

        self.bottleneck = nn.Linear((self.numFilters * 2 * int(int(self.timepoints / 4) / 8)), self.bottleneck_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        return x


class feat_classifier(nn.Module):
    """
    Classifier Module for EEGNet
    """

    def __init__(self,
                 input_dim: int,
                 n_class: int):
        super(feat_classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ELU(),
            nn.Linear(input_dim, n_class)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def cal_acc(x: torch.Tensor,
            y: torch.Tensor,
            netF: nn.Module,
            netC: nn.Module) -> Tuple[float, float]:
    """
    calculate the accuracy
    :param x: input
    :param y: ground truth label
    :param netF: feature extraction module
    :param netC: classifier module
    :return: accuracy
    """
    pred = netC(netF(x))
    pred = nn.Softmax(dim=1)(pred).cpu()
    return accuracy_score(y.cpu(), pred.argmax(dim=1)), cohen_kappa_score(y.cpu(), pred.argmax(dim=1))
