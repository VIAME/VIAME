# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import numpy as np
import torch
import collections

class track_state(object):
    def __init__(self, frame_id, bbox_center, ref_point,
                 interaction_feature, app_feature, bbox, ref_bbox,
                 detected_object, sys_frame_id, sys_frame_time):
        self.bbox_center = bbox_center
        self.ref_point = ref_point

        '''a list [x, y, w, h]'''
        self.bbox = bbox
        self.ref_bbox = ref_bbox

        # got required AMI features in torch.tensor format
        self.app_feature = app_feature
        self.motion_feature = torch.FloatTensor(2).zero_()
        self.interaction_feature = interaction_feature
        self.bbar_feature = torch.FloatTensor(2).zero_()

        self.frame_id = frame_id

        self.detected_object = detected_object

        self.sys_frame_id = sys_frame_id
        self.sys_frame_time = sys_frame_time

        # FIXME: the detected_object confidence does not work
        # For now, I just set the confidence = 1.0
        #self.conf = detectedObject.confidence
        self.conf = 1.0


class track(object):
    def __init__(self, track_id):
        self.track_id = track_id
        self._all_track_states = []
        self._current_track_states = []

    @property
    def full_history(self):
        return self._all_track_states

    def __len__(self):
        return len(self._current_track_states)

    def __getitem__(self, idx):
        return self._current_track_states[idx]

    def __iter__(self):
        return iter(self._current_track_states)

    def append(self, new_track_state, on_duplicate=None):
        ats, cts = self._all_track_states, self._current_track_states
        if not ats or ats[-1].frame_id < new_track_state.frame_id:
            pass
        elif ats[-1].frame_id == new_track_state.frame_id:
            if on_duplicate is None or on_duplicate == 'error':
                raise ValueError("Cannot append state with duplicate frame ID")
            elif on_duplicate == 'replace':
                if not cts:
                    raise ValueError
                ats.pop()
                cts.pop()
            else:
                raise ValueError("Unknown value for on_duplicate")
        else:
            raise ValueError("Cannot append state with earlier frame ID")

        if not cts:
            new_track_state.motion_feature = torch.FloatTensor(2).zero_()
        else:
            pre_ref_point = np.asarray(cts[-1].ref_point, dtype=np.float32).reshape(2)
            cur_ref_point = np.asarray(new_track_state.ref_point, dtype=np.float32).reshape(2)
            new_track_state.motion_feature = torch.from_numpy(cur_ref_point - pre_ref_point)

        ats.append(new_track_state)
        cts.append(new_track_state)

    def duplicate_track_state(self, timestep_len = 6):
        du_track = track(self.track_id)
        tsl = self._current_track_states
        tsl = [tsl[0]] * (timestep_len - len(tsl)) + tsl
        du_track._all_track_states = tsl
        du_track._current_track_states = tsl[:]

        return du_track

    def clear(self):
        self._current_track_states = []


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
        return self.id_ts_dict[track_id]

    def get_all_track_id(self):
        return sorted(self.id_ts_dict)

    def get_max_track_id(self):
        return max(self.id_ts_dict) if self.id_ts_dict else 0

    def deactivate_track(self, track):
        del self.active_id_set[track.track_id]

    def deactivate_all_tracks(self):
        self.active_id_set.clear()

    def active_count(self):
        return len(self.active_id_set)

    def make_track(self, track_id, on_exist=None):
        """Create a new track in this track_set with the provided track ID,
        mark it as active, and return it.

        If track_id is the ID of an existing track, behavior is
        controlled by the value of on_exist as follows:
        - "error" (default): ValueError is raised
        - "resume": the existing track is reused after reactivating it
          if necessary
        - "restart": the existing track's current history is cleared
          and then it is reused after reactivating it if necessary

        Other values of on_exist are invalid and will result in a
        ValueError.

        """
        if track_id not in self.id_ts_dict:
            new_track = track(track_id)
            self.id_ts_dict[track_id] = new_track
        elif on_exist is None or on_exist == 'error':
            raise ValueError("Track ID exists in the track set!")
        elif on_exist in ('resume', 'restart'):
            new_track = self.id_ts_dict[track_id]
            if on_exist == 'restart':
                new_track.clear()
        else:
            raise ValueError("Unrecognized value for on_exist")
        self.active_id_set[track_id] = None
        return new_track


if __name__ == '__main__':
    t = track(0)
    for i in range(10):
        t.append(track_state((i, i*i*0.1), [], []))

    for item in t[:]:
        print(item.motion_feature)
