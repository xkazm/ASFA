# DeepConvNet

from typing import Optional, Tuple
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1., **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data,
                                        p=2,
                                        dim=0,
                                        maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=1., **kwargs):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data,
                                        p=2,
                                        dim=0,
                                        maxnorm=self.max_norm)
        return super(LinearWithConstraint, self).forward(x)


def CalculateOutSize(blocks, channels, samples):
    '''
    Calculate the output based on input size.
    model is from nn.Module and inputSize is a array.
    '''
    x = torch.rand(1, 1, channels, samples)
    for block in blocks:
        block.eval()
        x = block(x)
    shape = x.shape[-2] * x.shape[-1]
    return shape


class DeepConvNetBase(nn.Module):
    def __init__(self,
                 n_classes: int,
                 Chans: int,
                 Samples: int,
                 dropoutRate: Optional[float] = 0.5):
        super(DeepConvNetBase, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate

        self.block1 = nn.Sequential(
            Conv2dWithConstraint(in_channels=1,
                                 out_channels=25,
                                 kernel_size=(1, 5),
                                 max_norm=2.),
            Conv2dWithConstraint(in_channels=25,
                                 out_channels=25,
                                 kernel_size=(self.Chans, 1),
                                 max_norm=2.),
            nn.BatchNorm2d(num_features=25),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.block2 = nn.Sequential(
            Conv2dWithConstraint(in_channels=25,
                                 out_channels=50,
                                 kernel_size=(1, 5),
                                 max_norm=2.),
            nn.BatchNorm2d(num_features=50),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.block3 = nn.Sequential(
            Conv2dWithConstraint(in_channels=50,
                                 out_channels=100,
                                 kernel_size=(1, 5),
                                 max_norm=2.),
            nn.BatchNorm2d(num_features=100),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.classifier_block = nn.Sequential(
            LinearWithConstraint(
                in_features=100 *
                            CalculateOutSize([self.block1, self.block2, self.block3],
                                             self.Chans, self.Samples),
                out_features=self.n_classes,
                bias=True,
                max_norm=0.5))

        # print('computed:', CalculateOutSize([self.block1, self.block2, self.block3], self.Chans, self.Samples))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output = self.block3(output)
        output = output.reshape(output.size(0), -1)
        # print('real:', output.shape[1])
        output = self.classifier_block(output)
        return output


class Activation(nn.Module):
    def __init__(self, type):
        super(Activation, self).__init__()
        self.type = type

    def forward(self, input):
        if self.type == 'square':
            output = input * input
        elif self.type == 'log':
            output = torch.log(torch.clamp(input, min=1e-6))
        else:
            raise Exception('Invalid type !')

        return output


def cal_acc(x: torch.Tensor,
            y: torch.Tensor,
            netF: nn.Module) -> Tuple[float, float]:
    """
    calculate the accuracy
    :param x: input
    :param y: ground truth label
    :param netF: feature extraction module
    :param netC: classifier module
    :return: accuracy
    """
    pred = netF(x)
    pred = nn.Softmax(dim=1)(pred).cpu()
    return accuracy_score(y.cpu(), pred.argmax(dim=1)), cohen_kappa_score(y.cpu(), pred.argmax(dim=1))
