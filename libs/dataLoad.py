# Load the data and get Tangent Space Feature

from typing import Optional
import os

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
import scipy.io as scio
import scipy.signal as signal


class MIDataLoader(object):
    """
    Load MI-BCI datasets
    Obtain preprocessed EEG trials and tangent space features

    Parameters:
        - **root** (str): Root directory of the dataset
        - **label_dict** Optional(dict): Label dictionary of the dataset
    """

    def __init__(self,
                 root: str,
                 label_dict: Optional[dict] = None,
                 fs: Optional[int] = 250,
                 fd: Optional[int] = 128,
                 downsample: Optional[bool] = False):
        self.root = root
        self.filename_list = sorted(os.listdir(self.root))
        self.nSubjects = len(self.filename_list)

        self.fs = fs
        self.fd = fd
        self.downsample = downsample

        self.label_dict = label_dict

        self.data = {}  # the data will be returned in a dict

    def load_data(self):
        """
        Load the EEG trials and get the tangent space feature
        :return: self.data, which has one keys, str(s), for indexing users
                 example: data['1']['x'] denotes the second subject's signals
                          data['1']['y'] denotes the second subject's labels
                          data['1']['tan'] denotes the second subject's tangent space feature
                          data['1']['tanTrans'] denotes the second subject's tangent space transformer
        """
        for s in range(self.nSubjects):
            print("load data subject:", s)
            data = scio.loadmat(self.root + self.filename_list[s])
            if 'x' in data.keys():
                x = np.array(data['x'])
            else:
                x = np.array(data['X'])

            # ref_x = np.mean(x, axis=1)
            # ref_x = np.expand_dims(ref_x, axis=1).repeat(x.shape[1], axis=1)
            # x = x - ref_x

            # common average reference会破坏最终的效果
            yC = np.array(data['y'])

            # Load the label dict and get the numerical labels
            if self.label_dict is not None:
                y = np.array([self.label_dict[yC[j]] for j in range(len(yC))]).reshape(-1)
            else:
                y = yC.reshape(-1)

            n_trials, n_channels, n_timepoints = x.shape

            if self.downsample:
                secs = n_timepoints / self.fs
                samps = int(secs * self.fd)
                x = signal.resample(x, num=samps, axis=2)

            # Get the tangent space feature
            cov = Covariances().transform(x) + 1e-8 * np.eye(n_channels).reshape(1, n_channels, n_channels).repeat(
                n_trials, axis=0)
            tanTrans = TangentSpace().fit(cov)
            tan = tanTrans.transform(cov)

            # Save the data in the dict
            temp = {'x': x,
                    'y': y,
                    'tan': tan,
                    'tanTrans': tanTrans}
            self.data[str(s)] = temp


def loadData(root: str,
             label_dict: Optional[dict] = None,
             fs: Optional[int] = 250,
             fd: Optional[int] = 128,
             downsample: Optional[bool] = False) -> MIDataLoader:
    """
    Load the data before main function
    :param root: the fileroot of the dataset
    :param label_dict: the label dictionary of the dataset
    :return: data
    """
    data = MIDataLoader(
        root=root,
        label_dict=label_dict,
        fs=fs,
        fd=fd,
        downsample=downsample
    )
    data.load_data()
    return data
