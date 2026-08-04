# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import os

from torch import optim

from .models import get_config as _get_model_config


def _epochs(name, default):
    """An epoch count, overridable from the environment.

    Each stage runs as its own process, so the environment is what reaches
    all of them; a config value would have to be threaded through every
    stage's command line. Used to cut a full run down to something that can
    be exercised end to end in minutes, and to trim epochs on a real one.
    """
    value = os.environ.get(name)

    if value and value.strip().isdigit() and int(value) > 0:
        return int(value)

    return default


class Config(_get_model_config().__class__):
    # general
    displayInterval = 100
    vali_displayInterval = 100
    train_BatchSize = 64
    vali_BatchSize = 64

    optimizer = optim.Adam

    # Siamese CNN
    maxIterations = _epochs('VIAME_SRNN_SIAMESE_EPOCHS', 10)
    margin = 1.0

    # lstm training settings
    # (other lstm settings are inherited)
    maxRNNIterations = _epochs('VIAME_SRNN_LSTM_EPOCHS', 50)
    lstm_init_lr = 0.002
    lstm_lr_step = 5


def get_config():
    return Config()
