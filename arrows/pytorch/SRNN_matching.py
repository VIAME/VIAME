import torch
import torch.utils.data as data
from torch import nn
from torch.autograd import Variable

import numpy as np
import math

from kwiver.arrows.pytorch.models import TargetLSTM, get_config, RnnType

g_config = get_config()


class TargetRNNDataLoader(data.Dataset):

    def __init__(self, track_set, track_state_list, track_search_threshold, rnnType):
        self._track_set = track_set
        self._track_state_list = track_state_list
        self._track_search_threshold = track_search_threshold
        self._rnnType = rnnType

        self._tracks_num = len(track_set)
        self._track_states_num = len(track_state_list)

        self._data_list = self._make_dataset()
        
    def _make_dataset(self):
        _data_list = []

        for t in range(self._tracks_num):
            cur_track = self._track_set[t]
            
            if len(cur_track) < g_config.timeStep:
                # if the track does not have enough track_state, we will duplicate to time-step and use Target_RNN_AIM_V
                _rnnType = RnnType.Target_RNN_AIM_V
                if _rnnType is self._rnnType:
                    cur_track = cur_track.duplicate_track_state(g_config.timeStep)
            else:
                _rnnType = RnnType.Target_RNN_AIM

            # only process active and un-unpdated track
            if (cur_track.active_flag is True) and (cur_track.updated_flag is False) and (_rnnType is self._rnnType):
                for ts in range(self._track_states_num):

                    # distance between the two bbox's x instead of center
                    dis = math.fabs(cur_track[-1].bbox[0] - self._track_state_list[ts].bbox[0]) 

                    bbox_area_ratio = float(cur_track[-1].bbox[2] * cur_track[-1].bbox[3]) / \
                                        float(self._track_state_list[ts].bbox[2] * self._track_state_list[ts].bbox[3]) 
                    if bbox_area_ratio < 1.0:
                        bbox_area_ratio = 1.0 / bbox_area_ratio

                    # in the search area, we prepare data and calculate the similarity score
                    if (dis < self._track_search_threshold * cur_track[-1].bbox[2] and                      # track dis constraint  
                        dis < self._track_search_threshold * self._track_state_list[ts].bbox[2] and         # track_state dis constraint
                        bbox_area_ratio < self._track_search_threshold):                                    # bbox area constraint
                        cur_data_item = tuple()

                        #app feature and app target
                        app_f_tensor = torch.stack([cur_track[i].app_feature for i in range(-g_config.timeStep, 0)])
                        cur_data_item = cur_data_item + (app_f_tensor, )
                        cur_data_item = cur_data_item + (self._track_state_list[ts].app_feature, )

                        #motion feature and motion target
                        motion_f_tensor = torch.stack([cur_track[i].motion_feature for i in range(-g_config.timeStep, 0)])
                        cur_data_item = cur_data_item + (motion_f_tensor, )
                        motion_target_f = np.asarray(self._track_state_list[ts].bbox_center, dtype=np.float32).reshape(g_config.M_F_num) - \
                                          np.asarray(cur_track[-1].bbox_center, dtype=np.float32).reshape(g_config.M_F_num)
                        cur_data_item = cur_data_item + (torch.from_numpy(motion_target_f), )
            
                        #interaction feature and interaction target
                        interaction_f_tensor = torch.stack([cur_track[i].interaction_feature for i in range(-g_config.timeStep, 0)])
                        cur_data_item = cur_data_item + (interaction_f_tensor, )
                        cur_data_item = cur_data_item + (self._track_state_list[ts].interaction_feature, )

                        #bbar feature and bbar target
                        bbar_f_tensor = torch.stack([cur_track[i].bbar_feature for i in range(-g_config.timeStep, 0)])
                        cur_data_item = cur_data_item + (bbar_f_tensor, )
                        cur_data_item = cur_data_item + (self._track_state_list[ts].bbar_feature, )
            
                        #corresponding position of the similarity matrix
                        cur_data_item = cur_data_item + (t, ts,)
            
                        _data_list.append(cur_data_item)

        return _data_list

    def __getitem__(self, index):
        return self._data_list[index] 
    
    def __len__(self):
        return len(self._data_list)


class SRNN_matching(object):
    def __init__(self, targetRNN_full_model_path, targetRNN_AIM_V_model_path, batch_size):

        self._batch_size = batch_size


        # load target AIM model, trained with fixed variable timestep
        full_model_list = (RnnType.Appearance, RnnType.Motion, RnnType.Interaction)
        self._targetRNN_full_model = TargetLSTM(model_list=full_model_list)
        self._targetRNN_full_model = self._targetRNN_full_model.cuda()

        snapshot = torch.load(targetRNN_full_model_path)
        self._targetRNN_full_model.load_state_dict(snapshot['state_dict'])
        self._targetRNN_full_model.eval()
        self._targetRNN_full_model = torch.nn.DataParallel(self._targetRNN_full_model, device_ids=[0]).cuda()

        # load  target AIM_V model, but trained with variable timestep
        V_model_list = (RnnType.Appearance, RnnType.Motion, RnnType.Interaction)
        self._targetRNN_AIM_V_model = TargetLSTM(model_list=V_model_list).cuda()

        snapshot = torch.load(targetRNN_AIM_V_model_path)
        self._targetRNN_AIM_V_model.load_state_dict(snapshot['state_dict'])
        self._targetRNN_AIM_V_model.eval()
        self._targetRNN_AIM_V_model = torch.nn.DataParallel(self._targetRNN_AIM_V_model, device_ids=[0]).cuda()

    def __call__(self, track_set, track_state_list, track_search_threshold):
        tracks_num = len(track_set)
        track_states_num = len(track_state_list)

        self._similarity_mat = torch.FloatTensor(tracks_num, track_states_num).fill_(1.0).cuda()

        # obtain the dict: simailarity row idx->track_id
        track_idx_list = []
        for t in range(tracks_num):
            track_idx_list.append(track_set[t].id)

        kwargs = {'num_workers': 6, 'pin_memory': True}
        AIM_V_data_loader = torch.utils.data.DataLoader(
            TargetRNNDataLoader(track_set, track_state_list, track_search_threshold, RnnType.Target_RNN_AIM_V),
            batch_size=self._batch_size, shuffle=False, **kwargs)

        AIM_data_loader = torch.utils.data.DataLoader(
            TargetRNNDataLoader(track_set, track_state_list, track_search_threshold, RnnType.Target_RNN_AIM),
            batch_size=self._batch_size, shuffle=False, **kwargs)
            
        self._est_similarity(AIM_V_data_loader, RnnType.Target_RNN_AIM_V)
        self._est_similarity(AIM_data_loader, RnnType.Target_RNN_AIM)

        return self._similarity_mat.cpu().numpy(), track_idx_list


    def _est_similarity(self, loader, rnnType):

        for (app_f_list, app_target_f, motion_f_list, motion_target_f, interaction_f_list, interaction_target_f, bbar_f_list, bbar_target_f, t, ts) in loader:

            v_app_seq, v_app_target = Variable(app_f_list, volatile=True).cuda(), Variable(app_target_f, volatile=True).cuda()
            v_motion_seq, v_motion_target = Variable(motion_f_list, volatile=True).cuda(), Variable(motion_target_f, volatile=True).cuda()
            v_interaction_seq, v_interaction_target = Variable(interaction_f_list, volatile=True).cuda(), Variable(interaction_target_f, volatile=True).cuda()
            v_bbar_seq, v_bbar_target = Variable(bbar_f_list, volatile=True).cuda(), Variable(bbar_target_f, volatile=True).cuda()
        
            if rnnType is RnnType.Target_RNN_AIM:
                output = self._targetRNN_full_model(v_app_seq, v_app_target, v_motion_seq, v_motion_target, 
                                                    v_interaction_seq, v_interaction_target, v_bbar_seq, v_bbar_target)
            elif rnnType is RnnType.Target_RNN_AIM_V:
                output = self._targetRNN_AIM_V_model(v_app_seq, v_app_target, v_motion_seq, v_motion_target, 
                                                  v_interaction_seq, v_interaction_target)
    
            F_softmax = nn.Softmax()
            output = F_softmax(output)
            pred = torch.max(output, 1)
    
            label_mask = torch.ne(pred[1].data, 0)
            
            r_idx = t.cuda()[label_mask]
            c_idx = ts.cuda()[label_mask]
            val = -pred[0].data[label_mask]
            
            if len(r_idx) == 0:
                pass
            else:
                for i in range(r_idx.size(0)):
                    self._similarity_mat[r_idx[i], c_idx[i]] = val[i]

