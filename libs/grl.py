# Gradient Reversal Layer Module
# code from https://github.com/thuml/Transfer-Learning-Library

from typing import Optional, Any, Tuple

import numpy as np
import torch.nn as nn
from torch.autograd import Function
import torch


class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any,
                input: torch.Tensor,
                coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any,
                 grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class WarmStartGradientReverseLayer(nn.Module):
    """
    Gradient Reverse Layer with warm start

    Parameters:
        - **alpha** Optional(float): a hyperparameter for weight the ratio
        - **low** Optional(float): initial value of the coefficient
        - **high** Optional(float): final value of the coefficient
        - **max_iters** Optional(float): num of maximum iterations
    """

    def __init__(self,
                 alpha: Optional[float] = 1.0,
                 low: Optional[float] = 0.0,
                 high: Optional[float] = 1.0,
                 max_iters: Optional[int] = 1000.):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.low = low
        self.high = high
        self.max_iters = max_iters

    def forward(self,
                input: torch.Tensor,
                iter_num: int) -> torch.Tensor:
        coeff = np.float(
            2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * iter_num / self.max_iters))
            - (self.high - self.low) + self.low
        )
        # print('coeff:',coeff)
        return GradientReverseFunction.apply(input, coeff)
