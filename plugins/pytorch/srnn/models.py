# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import torch
from torchvision import models
from torch import nn
from enum import Enum


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
    def __init__(self):
        super(Siamese, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.num_fcin = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.num_fcin, 500)

    def forward(self, input1):
        output1 = self.resnet(input1)

        return output1


# LSTMs
# ==================================================================
class BaseLSTM(nn.Module):
    def __init__(self, f_in_num, normalized):
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

    # TODO: we may need to add hidden status from previous
    def forward(self, track_input, target_input):
        if self.normalized:
            # Put all input into one batch and normalize it
            all_input = torch.cat((track_input, target_input.unsqueeze(1)), 1)
            all_input_norm = self.bn(all_input.transpose(1, 2)).transpose(1, 2)
            track_input, target_input = all_input_norm[:, :-1], all_input_norm[:, -1]

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
            outs.append(self.fc2(relu_output))

        return torch.stack(outs, dim=1), torch.stack(relu_outs, dim=1)


class AppearanceLSTM(BaseLSTM):
    def __init__(self, normalized):
        super(AppearanceLSTM, self).__init__(g_config.A_F_num, normalized)


class InteractionLSTM(BaseLSTM):
    def __init__(self, normalized):
        super(InteractionLSTM, self).__init__(g_config.I_F_num, normalized)


class MotionLSTM(BaseLSTM):
    def __init__(self, normalized):
        super(MotionLSTM, self).__init__(g_config.M_F_num, normalized)


class BBoxLSTM(BaseLSTM):
    def __init__(self, normalized):
        super(BBoxLSTM, self).__init__(g_config.B_F_num, normalized)


# Target LSTM
# ==================================================================
class TargetLSTM(nn.Module):
    def __init__(self, app_model='', motion_model='', interaction_model='', bbox_model='',
                 model_list=(RnnType.Appearance, RnnType.Motion, RnnType.Interaction),
                 normalized=False, use_gpu_flag=True):
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

        combined_input = torch.cat(out_list, 2)
        r_out, (h_t, c_t) = self.lstm(combined_input, None)

        out = self.fc1(r_out[:, -1, :])

        return out
