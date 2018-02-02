import torch.utils.data as data
import numpy as np

from track import track_set, track, track_state
from models import get_config

g_config = get_config()


class TargetRNNDataLoader(data.Dataset):
    '''
    process one col of the similarity matrix

    similarity matrix: track_num x track_state_num

    here we banch processing one col of the matrix

    '''
    def __init__(self, track_set, track_state, transform=None):

        self.track_set = track_set
        self.track_state = track_state

        self.transform = transform

    def __getitem__(self, index):

        track = self.track_set[index]

        for idx, ts in enumerate(track[-g_config.timeStep:]):
            if idx == 0:
                app_f_list = ts.app_feature.reshape(1, g_config.A_F_num)
                motion_f_list = ts.motion_feature.reshape(1, g_config.M_F_num)
                interaction_f_list = ts.interaction_feature.reshape(1, g_config.I_F_num)
            else:
                app_f_list = np.append(app_f_list, ts.app_feature.reshape(1, g_config.A_F_num), axis=0)
                motion_f_list = np.append(motion_f_list, ts.motion_feature.reshape(1, g_config.M_F_num), axis=0)
                interaction_f_list = np.append(interaction_f_list, ts.interaction_feature.reshape(1, g_config.I_F_num),
                                               axis=0)

        app_target_f = self.track_state.app_feature.reshape(1, g_config.A_F_num)
        interaction_target_f = self.track_state.interaction_feature.reshape(1, g_config.I_F_num)
        motion_target_f = np.asarray(self.track_state.bbox_center).reshape(1, g_config.M_F_num) - \
                          np.asarray(track[-1].bbox_center).reshape(1, g_config.M_F_num)

        return app_f_list, app_target_f, motion_f_list, motion_target_f, interaction_f_list, interaction_target_f

    def __len__(self):
        return len(self.track_set)

