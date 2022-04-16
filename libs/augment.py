# Data augmentation for MI-based EEG signal and perturbation for features

from typing import Optional

from pyriemann.estimation import Covariances
import numpy as np
import torch
from torch.autograd import Variable

import sys

sys.path.append('.')
from .dataLoad import MIDataLoader


def augment(data: MIDataLoader,
            subject: int,
            args,
            p: float,
            lamb: float) -> torch.Tensor:
    """
    augment the trials by randomly weakening some channels for the exact subject
    :param data: original data
    :param subject: the augmented subject
    :param args: hyper parameters
    :param p: weaken probability
    :param lamb: weaken low bound
    :return: augmented trials' tangent space feature
    """
    x = data.data[str(subject)]['x']
    n_trials, n_channels, n_timepoints = x.shape
    x_aug = np.zeros((n_trials, n_channels, n_timepoints))

    # Generate the random for fix the weakened channels
    prng = np.random.RandomState(args.seed)
    args.seed = args.seed + 1  # avoid the same random seed during the training process
    random_all = prng.random((n_trials, n_channels))
    for i in range(n_trials):
        random_trial = random_all[i, :].reshape(-1)
        indx = np.where(random_trial < p)[0]
        # Generate the weakening factor
        mask = np.ones((n_channels, n_timepoints))
        prng = np.random.RandomState(args.seed)
        args.seed = args.seed + 1
        mask_c = prng.random(len(indx))
        mask[indx, :] = (mask_c * (1.0 - lamb) + lamb).reshape(-1, 1).repeat(n_timepoints, axis=1)
        x_aug[i, :, :] = mask * x[i, :, :]

    # Get the feature
    cov = Covariances().transform(x_aug)
    tan = data.data[str(subject)]['tanTrans'].transform(cov)

    return Variable(torch.from_numpy(tan).type(torch.FloatTensor))


def random_dropout(x: torch.Tensor,
                   args) -> torch.Tensor:
    """
    perturbation on feature level by randomly dropout
    :param x: the input feature tensor
    :param args: hyper parameter
    :return: perturbed feature tensor
    """
    m, n = x.shape
    prng = np.random.RandomState(args.seed)
    args.seed = args.seed + 1
    random_all = prng.random((m, n))
    mask = torch.from_numpy(random_all).to(args.device)
    zero = torch.zeros(1).type(torch.FloatTensor).to(
        args.device)  # get 0,due to bug, can not directly use 0. in next line
    augment_x = torch.where(mask >= args.pf, x, zero[0])
    return augment_x / (1 - args.pf)


def random_noise(x: torch.Tensor,
                 args) -> torch.Tensor:
    """
    perturbation on feature level by adding random noise
    :param x: the input feature tensor
    :param args: hyper parameters
    :return: perturbed feature tensor
    """
    m, n = x.shape
    std = torch.std(x)
    prng = np.random.RandomState(args.seed)
    args.seed = args.seed + 1
    noise = (prng.random((m, n)) - 0.5) * 2.0  # noise~(-1,1)
    noise = torch.from_numpy(noise).type(torch.FloatTensor).to(args.device)
    noise = noise * std * args.alpha
    return x + noise


def random_dropmin(x: torch.Tensor,
                   args) -> torch.Tensor:
    """
    perturbation on feature level by drop the last several small features
    :param x: the input feature tensor
    :param args: hyper parameters
    :return: perturbed feature tensor
    """
    m, n = x.shape
    interval = (args.high - args.low) / n
    gmask = np.linspace(start=1.0, stop=interval, num=50)
    gmask = torch.from_numpy(gmask).type(torch.FloatTensor).to(args.device)
    prng = np.random.RandomState(args.seed)
    args.seed = args.seed + 1
    mask = prng.random(m)
    mask = 1.0 - ((args.high - args.low) * mask + args.low)
    mask = torch.from_numpy(mask).to(args.device)
    indx = torch.argsort(input=x, dim=1, descending=True)
    x_sort = torch.sort(input=x, dim=1, descending=True)[0]
    augment_x = torch.zeros_like(x).to(args.device)
    for i in range(m):
        temp_indx = indx[i, :int(mask[i] * n)]
        augment_x[i, temp_indx] = x_sort[i, :int(mask[i] * n)] / gmask[:int(mask[i] * n)]
    return augment_x


def loadAugment(data: MIDataLoader,
                subject: int,
                args) -> torch.Tensor:
    """
    augment the trials by randomly weakening some channels for all subjects without the exact subject
    :param data: original data
    :param subject: the augmented subject
    :param args: hyper parameters
    :return: augmented trials' tangent space feature
    """
    x_aug = None
    for i in range(data.nSubjects):
        if i != subject:
            temp = augment(data, i, args, args.p, args.lamb)
            if x_aug is None:
                x_aug = temp
            else:
                x_aug = torch.cat((x_aug, temp), dim=0)
    return Variable(x_aug.type(torch.FloatTensor))
