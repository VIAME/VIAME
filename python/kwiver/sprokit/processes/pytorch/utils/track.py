# ckwg +28
# Copyright 2018 by Kitware, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither name of Kitware, Inc. nor the names of any contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import torch
import collections

class track_state(object):
    def __init__(self, frame_id, bbox_center, interaction_feature, app_feature, bbox, 
                    detected_object, sys_frame_id, sys_frame_time):
        self.bbox_center = bbox_center

        '''a list [x, y, w, h]'''
        self.bbox = bbox

        # got required AMI features in torch.tensor format
        self.app_feature = app_feature
        self.motion_feature = torch.FloatTensor(2).zero_()
        self.interaction_feature = interaction_feature
        self.bbar_feature = torch.FloatTensor(2).zero_()

        self.track_id = -1
        self.frame_id = frame_id

        self.detected_object = detected_object

        self.sys_frame_id = sys_frame_id
        self.sys_frame_time = sys_frame_time

        # FIXME: the detected_object confidence does not work
        # For now, I just set the confidence = 1.0
        #self.conf = detectedObject.confidence()
        self.conf = 1.0


class track(object):
    def __init__(self, track_id):
        self.track_id = track_id
        self.track_state_list = []
        self.max_conf = 0.0
        self.updated_flag = False

    def __len__(self):
        return len(self.track_state_list)

    def __getitem__(self, idx):
        return self.track_state_list[idx]

    def __iter__(self):
        return iter(self.track_state_list)

    def append(self, new_track_state):
        if not self.track_state_list:
            new_track_state.motion_feature = torch.FloatTensor(2).zero_()
        else:
            pre_bbox_center = np.asarray(self.track_state_list[-1].bbox_center, dtype=np.float32).reshape(2)
            cur_bbox_center = np.asarray(new_track_state.bbox_center, dtype=np.float32).reshape(2)
            new_track_state.motion_feature = torch.from_numpy(cur_bbox_center - pre_bbox_center)

        new_track_state.track_id = self.track_id
        self.track_state_list.append(new_track_state)
        self.max_conf = max(self.max_conf, new_track_state.conf)

    def duplicate_track_state(self, timestep_len = 6):
        du_track = track(self.track_id)
        tsl = self.track_state_list
        tsl = [tsl[0]] * (timestep_len - len(tsl)) + tsl
        du_track.track_state_list = tsl
        du_track.updated_flag = self.updated_flag
        du_track.max_conf = self.max_conf

        return du_track


class track_set(object):
    def __init__(self):
        self.id_ts_dict = collections.OrderedDict()
        # We implement an ordered set by mapping to None
        self.active_id_set = collections.OrderedDict()

    def __len__(self):
        return len(self.id_ts_dict)

    def __iter__(self):
        return iter(self.id_ts_dict.values())

    def iter_active(self):
        return (self[i] for i in self.active_id_set)

    def __getitem__(self, track_id):
        try:
            return self.id_ts_dict[track_id]
        except KeyError:
            raise IndexError

    def get_all_track_id(self):
        return sorted(self.id_ts_dict)

    def get_max_track_id(self):
        return max(self.id_ts_dict) if self.id_ts_dict else 0

    def deactivate_track(self, track):
        del self.active_id_set[track.track_id]

    def active_count(self):
        return len(self.active_id_set)

    def add_new_track(self, track):
        if track.track_id in self.id_ts_dict:
            print("track ID exists in the track set!!!")
            raise RuntimeError

        self.id_ts_dict[track.track_id] = track
        self.active_id_set[track.track_id] = None

    def add_new_track_state(self, track_id, track_state):
        new_track = track(track_id)
        new_track.append(track_state)
        self.add_new_track(new_track)

    def add_new_track_state_list(self, start_track_id, ts_list, thresh=0.0):
        track_id = start_track_id
        for ts in ts_list:
            if ts.detected_object.confidence() >= thresh:
                self.add_new_track_state(track_id, ts)
                track_id += 1
        return track_id

    def update_track(self, track_id, new_track_state):
        self[track_id].append(new_track_state)

    def reset_updated_flag(self):
        for track in self:
            track.updated_flag = False


if __name__ == '__main__':
    t = track(0)
    for i in range(10):
        t.append(track_state((i, i*i*0.1), [], []))

    for item in t[:]:
        print(item.motion_feature)
