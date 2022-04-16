# utils function used in the project

from typing import Optional
import os
import argparse

import random
import numpy as np
import torch
import torch.nn as nn
import scipy


def temperature_normalization(x: torch.Tensor,
                              t: Optional[float] = 0.1) -> torch.Tensor:
    """
    L2 Normalization for bottleneck module output with temperature factor
    :param x: feature tensor
    :param t: temperature factor
    :return: t-normalized feature
    """
    n_feature = x.shape[1]
    norm = torch.norm(x, p=2, dim=1).view(-1, 1).repeat(1, n_feature)
    return x / (norm * t)


def init_weights(model: nn.Module):
    """
    Network Parameters Initialization Function
    :param model: the model to initialize
    :return: None
    """
    classname = model.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight, 1.0, 0.02)
        nn.init.zeros_(model.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(model.weight)
        nn.init.zeros_(model.bias)
    elif classname.find('Conv2d') != -1:
        nn.init.kaiming_uniform_(model.weight)


def seed(seed: Optional[int] = 0):
    """
    fix all the random seed
    :param seed: random seed
    :return: None
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def print_args(args: argparse.ArgumentParser):
    """
    print the hyperparameters
    :param args: hyperparameters
    :return: None
    """
    s = "=========================================================\n"
    for arg, concent in args.__dict__.items():
        s += "{}:{}\n".format(arg, concent)
    return s


def _matrix_operator(Ci, operator):
    """matrix equivalent of an operator."""
    eigvals, eigvects = scipy.linalg.eigh(Ci, check_finite=False)
    eigvals = np.diag(operator(eigvals))
    Out = np.dot(np.dot(eigvects, eigvals), eigvects.T)
    return Out


def invsqrtm(Ci):
    """Return the inverse matrix square root of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{-1/2} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :returns: the inverse matrix square root

    """
    isqrt = lambda x: 1. / np.sqrt(x)
    return _matrix_operator(Ci, isqrt)
