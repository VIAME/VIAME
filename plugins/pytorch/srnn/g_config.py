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

    # The scheduler multiplies the rate by 0.1 every lstm_lr_step epochs. At
    # the old step of 5 the rate reached 2e-10 by epoch 35 -- zero for any
    # practical purpose -- so of a 50 epoch budget, thirty were spent applying
    # updates too small to change a weight. Three steps across the budget
    # leaves every phase long enough to converge at its rate.
    lstm_lr_step = 15

    # Stop a training whose validation loss has not improved for this many
    # epochs, rather than running out the epoch budget on a model that has
    # already been chosen. The per epoch validation pass this reads already
    # runs; the best epoch is still selected from the record afterwards, so
    # stopping early never changes which weights ship, only how long the tail
    # costs. None disables it.
    early_stop_patience = _epochs('VIAME_SRNN_PATIENCE', 8)


def get_config():
    return Config()
