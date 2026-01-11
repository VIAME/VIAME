# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import torch
import torch.utils.data as data
from torch import nn

import numpy as np

from .models import TargetLSTM, get_config, RnnType
from ..utilities import get_gpu_device

g_config = get_config()


class TargetRNNDataLoader(data.Dataset):
    def __init__(self, track_list, track_state_list, track_search_threshold, rnnType):
        self._track_list = track_list
        self._track_state_list = track_state_list
        self._track_search_threshold = track_search_threshold
        self._rnnType = rnnType
        self._data_list = self._make_dataset()

    def _make_dataset(self):
        _data_list = []
        for t, cur_track in enumerate(self._track_list):
            if len(cur_track) < g_config.timeStep:
                # if the track does not have enough track_state, we will duplicate to time-step and use Target_RNN_AIM_V
                _rnnType = RnnType.Target_RNN_AIM_V
                if _rnnType is self._rnnType:
                    cur_track = cur_track.duplicate_track_state(g_config.timeStep)
            else:
                _rnnType = RnnType.Target_RNN_AIM

            if _rnnType is self._rnnType:
                for ts, track_state in enumerate(self._track_state_list):
                    # distance between the two bbox's x instead of center
                    dis = abs(cur_track[-1].ref_bbox[0] - track_state.ref_bbox[0])

                    bbox_area_ratio = (float(cur_track[-1].ref_bbox[2] * cur_track[-1].ref_bbox[3]) /
                                       float(track_state.ref_bbox[2] * track_state.ref_bbox[3]))
                    if bbox_area_ratio < 1.0:
                        bbox_area_ratio = 1.0 / bbox_area_ratio

                    # in the search area, we prepare data and calculate the similarity score
                    if (dis < self._track_search_threshold * cur_track[-1].ref_bbox[2] and         # track dis constraint
                        dis < self._track_search_threshold * track_state.ref_bbox[2] and           # track_state dis constraint
                        bbox_area_ratio < self._track_search_threshold):                           # bbox area constraint
                        cur_data_item = []

                        #app feature and app target
                        app_f_tensor = torch.stack([cur_track[i].app_feature
                                                for i in range(-g_config.timeStep, 0)])
                        cur_data_item += app_f_tensor, track_state.app_feature

                        #motion feature and motion target
                        motion_f_tensor = torch.stack([cur_track[i].motion_feature
                                                for i in range(-g_config.timeStep, 0)])
                        motion_target_f = (np.asarray(track_state.ref_point,
                                            dtype=np.float32).reshape(g_config.M_F_num) -
                                           np.asarray(cur_track[-1].ref_point,
                                               dtype=np.float32).reshape(g_config.M_F_num))
                        cur_data_item += motion_f_tensor, torch.from_numpy(motion_target_f)

                        #interaction feature and interaction target
                        interaction_f_tensor = torch.stack([cur_track[i].interaction_feature
                                                for i in range(-g_config.timeStep, 0)])
                        cur_data_item += interaction_f_tensor, track_state.interaction_feature

                        #bbar feature and bbar target
                        bbar_f_tensor = torch.stack([cur_track[i].bbar_feature
                                                for i in range(-g_config.timeStep, 0)])
                        cur_data_item += bbar_f_tensor, track_state.bbar_feature

                        #corresponding position of the similarity matrix
                        cur_data_item += t, ts

                        _data_list.append(tuple(cur_data_item))
        return _data_list

    def __getitem__(self, index):
        return self._data_list[index]

    def __len__(self):
        return len(self._data_list)


class SRNNMatching(object):
    def __init__(self, targetRNN_full_model_path, targetRNN_AIM_V_model_path,
                 normalized, batch_size, gpu_list=None):
        self._device, self._use_gpu_flag = get_gpu_device(gpu_list)
        self._batch_size = batch_size

        def load_model(model_list, model_path):
            model = TargetLSTM(model_list=model_list, normalized=normalized, use_gpu_flag=self._use_gpu_flag)
            model = model.to(self._device)
            if self._use_gpu_flag:
                snapshot = torch.load(model_path)
            else:
                snapshot = torch.load(model_path, map_location='cpu')
            model.load_state_dict(snapshot['state_dict'])
            model.eval()
            if self._use_gpu_flag:
                # DataParallel also understands device_ids=None to mean "all devices", so we're good.
                model = torch.nn.DataParallel(model, device_ids=gpu_list)
            return model

        # load target AIM model, trained with fixed variable timestep
        full_model_list = (RnnType.Appearance, RnnType.Motion, RnnType.Interaction)
        self._targetRNN_full_model = load_model(full_model_list, targetRNN_full_model_path)

        # load target AIM_V model, but trained with variable timestep
        V_model_list = (RnnType.Appearance, RnnType.Motion, RnnType.Interaction)
        self._targetRNN_AIM_V_model = load_model(V_model_list, targetRNN_AIM_V_model_path)

    def __call__(self, track_list, track_state_list, track_search_threshold):
        tracks_num = len(track_list)
        track_states_num = len(track_state_list)

        if 0 in (tracks_num, track_states_num):
            # PyTorch doesn't handle zero-length dimensions cleanly
            # (see https://github.com/pytorch/pytorch/issues/5014);
            # this computes the intended result.
            return np.empty((tracks_num, track_states_num), dtype=np.float32)

        self._similarity_mat = torch.FloatTensor(tracks_num, track_states_num).fill_(1.0).to(self._device)

        kwargs = {'num_workers': 0, 'pin_memory': True}
        AIM_V_data_loader = torch.utils.data.DataLoader(
            TargetRNNDataLoader(track_list, track_state_list, track_search_threshold, RnnType.Target_RNN_AIM_V),
            batch_size=self._batch_size, shuffle=False, **kwargs)

        AIM_data_loader = torch.utils.data.DataLoader(
            TargetRNNDataLoader(track_list, track_state_list, track_search_threshold, RnnType.Target_RNN_AIM),
            batch_size=self._batch_size, shuffle=False, **kwargs)

        with torch.no_grad():
            self._est_similarity(AIM_V_data_loader, RnnType.Target_RNN_AIM_V)
            self._est_similarity(AIM_data_loader, RnnType.Target_RNN_AIM)

        return self._similarity_mat.cpu().numpy()


    def _est_similarity(self, loader, rnnType):
        for (app_f_list, app_target_f, motion_f_list, motion_target_f,
                interaction_f_list, interaction_target_f, bbar_f_list,
                bbar_target_f, t, ts) in loader:
            v_app_seq = app_f_list.to(self._device)
            v_app_target =  app_target_f.to(self._device)
            v_motion_seq = motion_f_list.to(self._device)
            v_motion_target = motion_target_f.to(self._device)
            v_interaction_seq = interaction_f_list.to(self._device)
            v_interaction_target = interaction_target_f.to(self._device)
            v_bbar_seq = bbar_f_list.to(self._device)
            v_bbar_target = bbar_target_f.to(self._device)
            if rnnType is RnnType.Target_RNN_AIM:
                output = self._targetRNN_full_model(v_app_seq, v_app_target, v_motion_seq,
                                        v_motion_target, v_interaction_seq,
                                        v_interaction_target, v_bbar_seq, v_bbar_target)
            elif rnnType is RnnType.Target_RNN_AIM_V:
                output = self._targetRNN_AIM_V_model(v_app_seq, v_app_target, v_motion_seq,
                                        v_motion_target, v_interaction_seq,
                                        v_interaction_target)

            F_softmax = nn.Softmax()
            output = F_softmax(output)
            pred = torch.max(output, 1)

            #label_mask = torch.ne(pred[1].data, 0)
            label_mask = torch.ne(pred[1].detach(), 0)

            r_idx = t.to(self._device)[label_mask]
            c_idx = ts.to(self._device)[label_mask]
            #val = -pred[0].data[label_mask]
            val = -pred[0].detach()[label_mask]

            if len(r_idx) != 0:
                for i in range(r_idx.size(0)):
                    self._similarity_mat[r_idx[i], c_idx[i]] = val[i]
