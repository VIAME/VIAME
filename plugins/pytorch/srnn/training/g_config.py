# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from torch import optim

from ..models import get_config as _get_model_config


class Config(_get_model_config().__class__):
    # general
    displayInterval = 100
    vali_displayInterval = 100
    train_BatchSize = 64
    vali_BatchSize = 64

    optimizer = optim.Adam

    # Siamese CNN
    maxIterations = 10
    margin = 1.0

    # lstm training settings
    # (other lstm settings are inherited)
    maxRNNIterations = 50
    lstm_init_lr = 0.002
    lstm_lr_step = 5


def get_config():
    return Config()
