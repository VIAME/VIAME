# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import os

import torch
from torchvision import models
from torch import nn
from enum import Enum


def env_rate(name, default):
    """A rate in [0, 1), overridable from the environment.

    The sibling of g_config's _epochs, and here rather than there because
    g_config imports this module and not the other way around. Same reason
    for reading the environment: each stage runs as its own process, so the
    environment is the only thing that reaches all of them without threading
    a value through every stage's command line. _epochs itself parses
    positive integers only, which a dropout or decay rate never is.

    Anything unparseable or outside the range falls back to the default. A
    rate is read once per process at import, long before a log line would be
    read, so raising here would only turn a typo in a job script into a stage
    that dies an hour in.
    """
    value = os.environ.get(name)

    if value:
        try:
            rate = float(value)
        except ValueError:
            return default

        if 0.0 <= rate < 1.0:
            return rate

    return default


def get_config():
    class Config():
        # lstm settings
        H = 128
        K = 100
        A_F_num = 500  # appearance CNN output #Dim of feature
        I_F_num = 49
        M_F_num = 2
        B_F_num = 2
        timeStep = 6

        # Dropout rates, one per head, because the heads differ by more than
        # two orders of magnitude in input width and overfit in proportion.
        # On FishTrack23 the appearance head reached 0.99 train accuracy
        # against 0.585 validation -- chance, for a binary same/different
        # decision -- while motion reached 0.947 validation on two inputs.
        # A single rate would either leave appearance unregularized or drown
        # motion, so appearance gets the heaviest and the narrow heads the
        # lightest. These live on the model config rather than the training
        # one because the modules below read them at construction, and the
        # training config inherits this class, so both see the same values.
        app_dropout = env_rate('VIAME_SRNN_APP_DROPOUT', 0.3)
        interaction_dropout = env_rate('VIAME_SRNN_INTERACTION_DROPOUT', 0.2)
        lstm_dropout = env_rate('VIAME_SRNN_LSTM_DROPOUT', 0.1)
        target_dropout = env_rate('VIAME_SRNN_TARGET_DROPOUT', 0.2)
    return Config()

g_config = get_config()


class RnnType(Enum):
    Appearance = 1
    Motion = 2
    Interaction = 3
    BBox = 4
    Target_RNN_AIM = 5
    Target_RNN_AI = 6
    Target_RNN_AIM_V= 7
    Target_RNN_AIMB = 8


# Siamese network
# ==================================================================
class Siamese(nn.Module):
    def __init__(self, pretrained=False):
        super(Siamese, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        self.num_fcin = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.num_fcin, 500)

    def forward(self, input1):
        output1 = self.resnet(input1)

        return output1


# LSTMs
# ==================================================================
def _or_default(rate, default):
    """The caller's rate, or the configured one where it gave none.

    Zero is a rate a caller can ask for -- it turns dropout off for one
    model without touching the others -- so None, not falsiness, is what
    means "unspecified" here.
    """
    return default if rate is None else rate


class BaseLSTM(nn.Module):
    def __init__(self, f_in_num, normalized, dropout=None, input_dropout=None):
        super(BaseLSTM, self).__init__()

        self.normalized = normalized
        if normalized:
            self.bn = nn.BatchNorm1d(f_in_num)
        self.target_fc = nn.Linear(f_in_num, g_config.H)
        self.lstm = nn.LSTM(
            input_size=f_in_num,
            hidden_size=g_config.H,
            num_layers=1,
            batch_first=True
        )
        self.fc1 = nn.Linear(g_config.H * 2, g_config.K)
        self.fc2 = nn.Linear(g_config.K, 2)
        self.relu = nn.ReLU()

        # nn.LSTM takes a dropout argument of its own, but it only applies
        # between stacked layers and these are all num_layers=1, where torch
        # ignores it and warns. Regularizing the recurrence itself would mean
        # a variational/zoneout scheme and new weights; these two modules
        # instead sit either side of it and hold no parameters at all, so
        # every existing snapshot still loads strict into the new class.
        self.input_drop = nn.Dropout(_or_default(input_dropout, 0.0))
        self.head_drop = nn.Dropout(_or_default(dropout, g_config.lstm_dropout))

    # TODO: we may need to add hidden status from previous
    def forward(self, track_input, target_input):
        if self.normalized:
            # Put all input into one batch and normalize it
            all_input = torch.cat((track_input, target_input.unsqueeze(1)), 1)
            all_input_norm = self.bn(all_input.transpose(1, 2)).transpose(1, 2)
            track_input, target_input = all_input_norm[:, :-1], all_input_norm[:, -1]

        # After normalization, so that the batch statistics are still those of
        # the real features, and to both sides of the comparison, since a
        # decision made from a track feature the target never drops is exactly
        # the asymmetry that lets a wide embedding memorize a pairing.
        track_input = self.input_drop(track_input)
        target_input = self.input_drop(target_input)

        target_out = self.target_fc(target_input)
        r_out, (h_t, c_t) = self.lstm(track_input, None)

        outs = []
        relu_outs = []
        for i in range(g_config.timeStep):
            h_t = r_out[:, i, :]
            combined_out = torch.cat((h_t, target_out), 1)
            fc1_output = self.fc1(combined_out)
            relu_output = self.relu(fc1_output)
            relu_outs.append(relu_output)
            # Dropped on the way into the classifier only. The undropped
            # activation is the second return value, which is what TargetLSTM
            # consumes as its input feature; that stage does its own dropout
            # on the concatenation, and doing it here as well would leave the
            # combined model unable to tell a dropped unit from a dead one.
            outs.append(self.fc2(self.head_drop(relu_output)))

        return torch.stack(outs, dim=1), torch.stack(relu_outs, dim=1)


class AppearanceLSTM(BaseLSTM):
    # 500 dimensions out of the Siamese CNN, and the only head whose input is
    # both wide and dense enough for dropping raw features to remove
    # redundancy rather than signal, so this is the one head that drops its
    # input as well as its classifier.
    def __init__(self, normalized, dropout=None, input_dropout=None):
        super(AppearanceLSTM, self).__init__(
            g_config.A_F_num, normalized,
            _or_default(dropout, g_config.app_dropout),
            _or_default(input_dropout, g_config.app_dropout),
        )


class InteractionLSTM(BaseLSTM):
    # 49 dimensions, a 7x7 occupancy grid around the detection. Wide enough to
    # overfit -- 0.733 validation on FishTrack23, between appearance and
    # motion -- but sparse, so its few non-zero cells are the signal and
    # zeroing them at random destroys it. Classifier dropout only.
    def __init__(self, normalized, dropout=None, input_dropout=None):
        super(InteractionLSTM, self).__init__(
            g_config.I_F_num, normalized,
            _or_default(dropout, g_config.interaction_dropout), input_dropout,
        )


class MotionLSTM(BaseLSTM):
    # Two dimensions of velocity. There is no redundancy to drop in a pair of
    # numbers, and this head already generalizes (0.947 validation), so it
    # takes the lightest rate and nothing on the input.
    def __init__(self, normalized, dropout=None, input_dropout=None):
        super(MotionLSTM, self).__init__(
            g_config.M_F_num, normalized, dropout, input_dropout)


class BBoxLSTM(BaseLSTM):
    # Two dimensions of box aspect/area ratio; same reasoning as motion.
    def __init__(self, normalized, dropout=None, input_dropout=None):
        super(BBoxLSTM, self).__init__(
            g_config.B_F_num, normalized, dropout, input_dropout)


# Target LSTM
# ==================================================================
class TargetLSTM(nn.Module):
    def __init__(self, app_model='', motion_model='', interaction_model='', bbox_model='',
                 model_list=(RnnType.Appearance, RnnType.Motion, RnnType.Interaction),
                 normalized=False, use_gpu_flag=True, dropout=None):
        super(TargetLSTM, self).__init__()

        self.model_list = model_list

        def load_model(make_model, model_path):
            """Call make_model and move the resulting model to GPU if use_gpu_flag
            is true and initialize it from model_path if truthy.

            """
            model = make_model(normalized=normalized)
            if use_gpu_flag:
                model = model.cuda()
            if model_path:
                snapshot = torch.load(model_path)
                model.load_state_dict(snapshot['state_dict'])
            return model

        if RnnType.Appearance in self.model_list:
            self.appearance = load_model(AppearanceLSTM, app_model)

        if RnnType.Motion in self.model_list:
            self.motion = load_model(MotionLSTM, motion_model)

        if RnnType.Interaction in self.model_list:
            self.interaction = load_model(InteractionLSTM, interaction_model)

        if RnnType.BBox in self.model_list:
            self.bbar = load_model(BBoxLSTM, bbox_model)

        self.lstm = nn.LSTM(
            input_size=g_config.K * len(model_list),
            hidden_size=g_config.H,
            num_layers=1,
            batch_first=True,
        )

        self.fc1 = nn.Linear(g_config.H, 2)

        # One rate for both ends here. The input is K per sub-model -- 300
        # dense activations for the usual three -- and it is where an overfit
        # head arrives: dropping across the concatenation is what stops the
        # combined model from routing its decision entirely through
        # appearance, which is the failure this stage inherits. This rate is
        # the combined model's own: the sub-models are fine-tuned here as
        # well, so each still drops at its own configured rate on top of it.
        rate = _or_default(dropout, g_config.target_dropout)
        self.input_drop = nn.Dropout(rate)
        self.head_drop = nn.Dropout(rate)


    def forward(self, appearance_input=None, appearance_target=None, motion_input=None,
                motion_target=None, interaction_input=None, interaction_target=None,
                bbar_input=None, bbar_target=None):
        r"""
        :param appearance_input:    appearance features         (batch, time_step, input_size)
        :param appearance_target:   appearance target feature   (batch, 1, input_size)
        :param motion_input:        motion features             (batch, time_step, input_size)
        :param motion_target:       motion target features      (batch, 1, input_size)
        :param interaction_input:   interaction features        (batch, time_step, input_size)
        :param interaction_target:  interaction target feature  (batch, 1, input_size)
        :param bbar_input:          bbar features               (batch, time_step, input_size)
        :param bbar_target:         bbar target feature         (batch, 1, input_size)
        :return:
        """

        out_list = []
        if RnnType.Appearance in self.model_list:
            _, app_out = self.appearance(appearance_input, appearance_target)
            out_list.append(app_out)

        if RnnType.Motion in self.model_list:
            _, motion_out = self.motion(motion_input, motion_target)
            out_list.append(motion_out)

        if RnnType.Interaction in self.model_list:
            _, interaction_out = self.interaction(interaction_input, interaction_target)
            out_list.append(interaction_out)

        if RnnType.BBox in self.model_list:
            _, bbox_out = self.bbar(bbar_input, bbar_target)
            out_list.append(bbox_out)

        combined_input = self.input_drop(torch.cat(out_list, 2))
        r_out, (h_t, c_t) = self.lstm(combined_input, None)

        out = self.fc1(self.head_drop(r_out[:, -1, :]))

        return out
