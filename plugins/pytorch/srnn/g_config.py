# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import os

from torch import optim

from .models import env_rate as _rate, get_config as _get_model_config


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

    # No weight decay on the Siamese stage by default. Its loss is
    # contrastive against a fixed margin of 1.0, and decay pulls the whole
    # embedding towards the origin, which shrinks every pairwise distance
    # against a margin that does not move with it -- a change to what the
    # stage optimizes, not just how hard it is regularized. The blanket
    # parameter group would also decay resnet50's BatchNorm scales and
    # biases, which is its own known pathology. The knob is here so the
    # stage can be tried with decay, but it takes a deliberate setting.
    siamese_weight_decay = _rate('VIAME_SRNN_SIAMESE_WEIGHT_DECAY', 0.0)

    # lstm training settings
    # (other lstm settings are inherited)
    maxRNNIterations = _epochs('VIAME_SRNN_LSTM_EPOCHS', 50)
    lstm_init_lr = 0.002

    # Adam with the conventional 1e-4, applied to every LSTM stage including
    # the combined one. Small enough to leave the heads that already
    # generalize alone -- motion is two inputs and 0.947 validation, and
    # decay of this size will not move it -- while giving the appearance
    # head's much larger weight matrices something pulling back against a
    # training loss it can drive to 0.02. Dropout is the larger lever here;
    # this is the one that keeps working after dropout is turned off.
    lstm_weight_decay = _rate('VIAME_SRNN_LSTM_WEIGHT_DECAY', 1e-4)

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
