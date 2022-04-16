# Loss function used in this project

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def Entropy(input: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy
    :param input: the softmax output
    :return: entropy
    """
    entropy = -input * torch.log(input + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy


class LabelSmooth(nn.Module):
    """
    Label smooth cross entropy loss

    Parameters:
        - **num_class** (int): num of classes
        - **alpha** Optional(float): the smooth factor
        - **device** Optional(str): the used device "cuda" or "cpu"
    """

    def __init__(self,
                 num_class: int,
                 alpha: Optional[float] = 0.1,
                 device: Optional[str] = "cuda"):
        super(LabelSmooth, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.device = device

    def forward(self,
                inputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).to(self.device).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.alpha) * targets + self.alpha / self.num_class
        loss = (-targets * log_probs).sum(dim=1)
        return loss.mean()


class InformationMaximization(nn.Module):
    """
    Entropy Minimization and Diversity Maximization

    Parameters:
        - **gent** (bool): whether to consider the diversity
    """

    def __init__(self,
                 gent: bool):
        super(InformationMaximization, self).__init__()
        self.gent = gent

    def forward(self,
                output: torch.Tensor) -> torch.Tensor:
        softmax_out = nn.Softmax(dim=1)(output)
        entropy_loss = torch.mean(Entropy(softmax_out))
        if self.gent:
            msoftmax = softmax_out.mean(dim=0)
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-8))
            entropy_loss -= gentropy_loss
        return entropy_loss


class WeightedInformationMaximization(nn.Module):
    """
    Sample weighted information maximization
    """

    def __init__(self,
                 gent: bool,
                 t: Optional[float] = 2.0):
        super(WeightedInformationMaximization, self).__init__()
        self.gent = gent
        self.t = t

    def forward(self,
                output: torch.Tensor) -> torch.Tensor:
        n_sample, n_class = output.shape
        softmax_out = nn.Softmax(dim=1)(output / self.t)
        entropy_weight = Entropy(softmax_out).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (n_sample * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)
        entropy_weight = entropy_weight.view(-1)
        entropy_loss = torch.mean(Entropy(softmax_out) * entropy_weight)
        if self.gent:
            msoftmax = softmax_out.mean(dim=0)
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-8))
            entropy_loss -= gentropy_loss
        return entropy_loss


class SKLDivLoss(nn.Module):
    """
    Symmetric KL Divergence Loss for BAIT
    """

    def __init__(self):
        super(SKLDivLoss, self).__init__()

    def forward(self,
                out1: torch.Tensor,
                out2: torch.Tensor) -> torch.Tensor:
        out2_t = out2.clone()
        out2_t = out2_t.detach()
        out1_t = out1.clone()
        out1_t = out1_t.detach()
        return (F.kl_div(out1.log(), out2_t, reduction='none') +
                F.kl_div(out2.log(), out1_t, reduction='none')) / 2


class ClassifierDiscrepancyLoss(nn.Module):
    """
    The discrepancy loss between two classifier for BAIT

    Parameters:
        - **balance** Optional(bool): whether to consider the diversity
    """

    def __init__(self,
                 balance: Optional[bool] = True):
        super(ClassifierDiscrepancyLoss, self).__init__()
        self.balance = balance

    def forward(self,
                out1: torch.Tensor,
                out2: torch.Tensor) -> torch.Tensor:
        loss = torch.mean((-out1 * torch.log(out2 + 1e-5)).sum(dim=1) + (-out2 * torch.log(out1 + 1e-5)).sum(dim=1))
        if self.balance:
            mout1 = torch.mean(out1, dim=0)
            loss += torch.sum(mout1 * torch.log(mout1 + 1e-5))
            mout2 = torch.mean(out2, dim=0)
            loss += torch.sum(mout2 * torch.log(mout2 + 1e-5))
        return loss


class ClassConfusionLoss(nn.Module):
    """
    The class confusion loss

    Parameters:
        - **t** Optional(float): the temperature factor used in MCC
    """

    def __init__(self,
                 t: Optional[float] = 2.0):
        super(ClassConfusionLoss, self).__init__()
        self.t = t

    def forward(self,
                output: torch.Tensor) -> torch.Tensor:
        n_sample, n_class = output.shape
        softmax_out = nn.Softmax(dim=1)(output / self.t)
        entropy_weight = Entropy(softmax_out).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (n_sample * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)
        class_confusion_matrix = torch.mm((softmax_out * entropy_weight).transpose(1, 0), softmax_out)
        class_confusion_matrix = class_confusion_matrix / torch.sum(class_confusion_matrix, dim=1)
        mcc_loss = (torch.sum(class_confusion_matrix) - torch.trace(class_confusion_matrix)) / n_class
        return mcc_loss


class TEntropyLoss(nn.Module):
    """
    The Tsallis Entropy for Uncertainty Reduction

    Parameters:
        - **t** Optional(float): the temperature factor used in TEntropyLoss
        - **order** Optional(float): the order of loss function
    """

    def __init__(self,
                 t: Optional[float] = 2.0,
                 order: Optional[float] = 2.0):
        super(TEntropyLoss, self).__init__()
        self.t = t
        self.order = order

    def forward(self,
                output: torch.Tensor) -> torch.Tensor:
        n_sample, n_class = output.shape
        softmax_out = nn.Softmax(dim=1)(output / self.t)
        entropy_weight = Entropy(softmax_out).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (n_sample * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)
        entropy_weight = entropy_weight.repeat(1, n_class)
        tentropy = torch.pow(softmax_out, self.order) * entropy_weight
        # weight_softmax_out=softmax_out*entropy_weight
        tentropy = tentropy.sum(dim=0) / softmax_out.sum(dim=0)
        loss = -torch.sum(tentropy) / (n_class * (self.order - 1.0))
        return loss


class ConsistencyLoss(nn.Module):
    """
    The consistency loss for auxiliary decoders

    Parameters:
        - **k** Optional(int): num of auxiliary decoders
    """

    def __init__(self,
                 k: Optional[int] = 2):
        super(ConsistencyLoss, self).__init__()
        self.k = k

    def forward(self,
                output: torch.Tensor,
                output_aux: List[torch.Tensor]) -> torch.Tensor:
        loss = 0
        n_class = output.shape[1]
        for i in range(self.k):
            loss_aug = torch.mean(torch.sum((output_aux[i] - output) ** 2, dim=1))
            loss += loss_aug
        return loss / (self.k * n_class)
