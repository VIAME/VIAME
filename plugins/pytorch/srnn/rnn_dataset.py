# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import pickle

import numpy as np
import torch.utils.data as data
import torch

from .models import RnnType

from .g_config import get_config
from .storage import DataStorage
from .utilities import DividedDataset, rnn_type_to_feature

g_config = get_config()


def make_dataset(detection_file):
    with open(detection_file, 'rb') as f:
        return DividedDataset(pickle.load(f), 1, 0)


def feature_loader(f_blob, f_dim=g_config.A_F_num):
    f_data = np.frombuffer(f_blob.read(), dtype=np.float32)
    array = np.reshape(f_data, (f_dim,))

    return array


def compute_list_and_target(seq_w_target, rnn_type, get_blob, padding=None):
    """Get features from a list of detections

    Arguments:
    - seq_w_target: List of DetectionIds, with the last being the
      target
    - rnn_type: An RnnType, indicating the features to get
    - get_blob: Bound DataStorage.blob
    - padding: A string indicating the padding method, one of:
      - "left": Pad by duplicating the first state
      - "right" (default): Pad by duplicating the last state

    Returns a pair of:
    - f_list: 2D ndarray of features, padded to length
      g_config.timeStep as indicated by the padding parameter
    - target_f: 1D ndarray holding the target's features

    """
    if padding is None:
        padding = 'right'
    if padding == 'left':
        pad_left = True
    elif padding == 'right':
        pad_left = False
    else:
        raise ValueError("Invalid padding method")

    feature = rnn_type_to_feature(rnn_type)

    def loader(f_det, f_dim):
        return feature_loader(get_blob(f_det, feature), f_dim)
    if rnn_type is RnnType.Motion:  # motion data
        f_dim = g_config.M_F_num

        f_list = []
        for i_idx, f_det in enumerate(seq_w_target):
            if i_idx == 0:
                f_list.append(np.array([0, 0], dtype=np.float32))
                pre_center = loader(f_det, f_dim)
            else:
                cur_center = loader(f_det, f_dim)
                f_list.append(cur_center - pre_center)
                # update
                pre_center = cur_center
    else:
        if rnn_type is RnnType.Appearance:
            f_dim = g_config.A_F_num
        elif rnn_type is RnnType.Interaction:
            f_dim = g_config.I_F_num
        elif rnn_type is RnnType.BBox:
            f_dim = g_config.B_F_num
        else:
            raise ValueError("Unknown RnnType {!r}".format(rnn_type))
        f_list = [loader(f_det, f_dim) for f_det in seq_w_target]
    target_f = f_list.pop()
    if pad_left:
        f_list = [f_list[0]] * (g_config.timeStep - len(f_list)) + f_list
    else:
        f_list = f_list + [f_list[-1]] * (g_config.timeStep - len(f_list))
    return np.stack(f_list), target_f


class RNNDataLoader(data.Dataset):
    """A Siamese data loader where the images are arranged in this way in a file: ::
        assume time sequence length = 6

        tf_1 tf_2 tf_3 tf_4 tf_5 tf_6 pos_target_f neg_target_f

        tf:             track features (500)
        pos_target_f:   positive target features (500)
        neg_target_f:   negative target features (500)

        tf_1 ... tf_6 and pos_target_f's label=1: same class
        tf_1 ... tf_6 and neg_target_f's label=0: diff class

    Args:
        train_or_test_file (string): file name, which contains all data for training / testing

     Attributes:
        TODO: add the description about the
    """

    def __init__(self, data_root, train_or_test_file, rnn_type, padding=None):
        self.get_blob = DataStorage(data_root).blob
        self.data = make_dataset(train_or_test_file)
        self.rnn_type = rnn_type
        self.padding = padding

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            sequence feature list, target feature, label where target is 1: same class; 0: diff class.
        """
        temp_t, label = self.data[index]
        seq_f_list, target_f = compute_list_and_target(
            temp_t, self.rnn_type, self.get_blob, self.padding,
        )
        return (torch.from_numpy(seq_f_list),
                torch.from_numpy(target_f.copy()),
                torch.LongTensor([label]))

    def __len__(self):
        return len(self.data)
