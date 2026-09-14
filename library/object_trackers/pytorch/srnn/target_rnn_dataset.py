# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import pickle

import numpy as np
from PIL import Image
import torch
import torch.utils.data as data

from .models import RnnType

from .g_config import get_config
from .rnn_dataset import compute_list_and_target
from .storage import DataStorage
from .utilities import DividedDataset, rnn_type_to_feature

g_config = get_config()


def make_dataset(detection_file):
    r"""
    generate the data set with following format:
    [((*features, target), label)]

    features has time-step features (e.g. 6)
    target has one feature
    label = 1: from same track
    label = 0: from different track

    :param app_label_file:
    :param rnn_type_list:
    :return:
    """
    with open(detection_file, 'rb') as f:
        return DividedDataset(pickle.load(f), 1, 0)


class TargetRNNDataLoader(data.Dataset):
    """Target RNN data loader where the images are arranged in this way in a file: ::
        assume time sequence length = 6

        tf_1 tf_2 tf_3 tf_4 tf_5 tf_6 pos_target_f neg_target_f

        tf:             track features (500)
        pos_target_f:   positive target features (500)
        neg_target_f:   negative target features (500)

        tf_1 ... tf_6 and pos_target_f's label=1: same class
        tf_1 ... tf_6 and neg_target_f's label=0: diff class

        Here, we use appearance feature files (e.g., app_mot_train_set.txt, app_mot_test_set.txt). The motion and
        interaction input are obtained by replacing the appearance feature name as follow:

        appearance feature name:       app_feature.bin
        motion feature name:        bc_app_feature.bin
        interaction feature name: grid_app_feature.bin

    """

    def __init__(self, data_root, train_or_test_file, rnn_type_list,
                 padding=None):
        self.data = make_dataset(train_or_test_file)
        self.get_blob = DataStorage(data_root).blob
        self.rnn_type_list = rnn_type_list
        self.padding = padding

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image1, image2, target) where target is 1: same class; -1: diff class.
        """
        temp_t, label = self.data[index]

        app_f_list = np.empty([1, g_config.A_F_num])
        app_target_f = np.empty([1, g_config.A_F_num])
        motion_f_list = np.empty([1, g_config.M_F_num])
        motion_target_f = np.empty([1, g_config.M_F_num])
        interaction_f_list = np.empty([1, g_config.I_F_num])
        interaction_target_f = np.empty([1, g_config.I_F_num])
        bbar_f_list = np.empty([1, g_config.B_F_num])
        bbar_target_f = np.empty([1, g_config.B_F_num])

        for rnn_type in self.rnn_type_list:
            lt = compute_list_and_target(temp_t, rnn_type, self.get_blob,
                                         self.padding)
            lt = lt[0], lt[1].copy()
            if rnn_type is RnnType.Motion:
                motion_f_list, motion_target_f = lt
            elif rnn_type is RnnType.Appearance:
                app_f_list, app_target_f = lt
            elif rnn_type is RnnType.Interaction:
                interaction_f_list, interaction_target_f = lt
            elif rnn_type is RnnType.BBox:
                bbar_f_list, bbar_target_f = lt

        return (torch.from_numpy(app_f_list), torch.from_numpy(app_target_f),
                torch.from_numpy(motion_f_list), torch.from_numpy(motion_target_f),
                torch.from_numpy(interaction_f_list), torch.from_numpy(interaction_target_f),
                torch.from_numpy(bbar_f_list), torch.from_numpy(bbar_target_f),
                torch.LongTensor([label]))

    def __len__(self):
        return len(self.data)
