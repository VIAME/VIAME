# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Pair stereo detections using deep descriptor cosine distance.

This sprokit process takes two object track sets (left and right cameras)
and pairs detections across cameras by comparing deep feature descriptors
attached to each detection. Pairing is accumulated over multiple frames
to improve reliability.

Algorithm per step:
1. Extract descriptors from detections at the current frame
2. Compute cosine distance matrix between unpaired left/right descriptors
3. Greedy 1-to-1 matching below distance_threshold
4. Accumulate votes for consistent pairings across frames
5. When a pair reaches min_frames_for_pairing votes, confirm the pairing
6. Remap confirmed right track IDs to their paired left track IDs on output
"""

import numpy as np

from kwiver.sprokit.pipeline import process
from kwiver.sprokit.processes.kwiver_process import KwiverProcess
from kwiver.vital.types import (
    ObjectTrackSet, ObjectTrackState, Track, Timestamp
)


class PairStereoTracks(KwiverProcess):

    # --------------------------------------------------------------------------
    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

        # Config traits
        self.add_config_trait("distance_threshold", "distance_threshold", '0.3',
                              'Maximum cosine distance for a match (0.0 = identical, '
                              '1.0 = orthogonal). Lower values are more strict.')
        self.declare_config_using_trait('distance_threshold')

        self.add_config_trait("min_frames_for_pairing", "min_frames_for_pairing", '3',
                              'Number of consistent frames a pair must be observed '
                              'before confirming the pairing.')
        self.declare_config_using_trait('min_frames_for_pairing')

        # Set up port flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        # Input ports
        self.declare_input_port_using_trait('timestamp', required)
        self.declare_input_port_using_trait('object_track_set', required)

        # We need a second track set input, declare manually
        self.add_port_trait('object_track_set2', 'object_track_set',
                            'Second (right) camera object track set')
        self.declare_input_port_using_trait('object_track_set2', required)

        # Output ports
        self.declare_output_port_using_trait('object_track_set', optional)

        self.add_port_trait('object_track_set2_out', 'object_track_set',
                            'Output second (right) camera object track set with remapped IDs')
        self.declare_output_port_using_trait('object_track_set2_out', optional)

    # --------------------------------------------------------------------------
    def _configure(self):
        self._distance_threshold = float(self.config_value('distance_threshold'))
        self._min_frames = int(self.config_value('min_frames_for_pairing'))

        # Track pairing state
        # Key: (left_track_id, right_track_id), Value: vote count
        self._pair_votes = {}

        # Confirmed pairs: right_id -> left_id
        self._confirmed_pairs = {}

        self._base_configure()

    # --------------------------------------------------------------------------
    def _step(self):
        timestamp = self.grab_input_using_trait('timestamp')
        tracks1 = self.grab_input_using_trait('object_track_set')
        tracks2 = self.grab_value_using_trait('object_track_set2')

        frame_id = timestamp.get_frame() if timestamp.has_valid_frame() else -1

        if tracks1 is not None and tracks2 is not None and frame_id >= 0:
            self._process_frame(tracks1, tracks2, frame_id)

        # Remap right track IDs for confirmed pairs
        output_tracks2 = self._remap_tracks(tracks2, frame_id)

        # Push outputs
        self.push_to_port_using_trait('object_track_set',
                                      tracks1 if tracks1 else ObjectTrackSet())
        self.push_to_port_using_trait('object_track_set2_out',
                                      output_tracks2 if output_tracks2 else ObjectTrackSet())

    # --------------------------------------------------------------------------
    def _process_frame(self, tracks1, tracks2, frame_id):
        """Extract descriptors and accumulate pairing votes."""

        # Get descriptors for current frame detections
        left_descs = {}   # track_id -> normalized descriptor vector
        right_descs = {}

        for trk in tracks1.tracks():
            tid = trk.id
            # Skip if already confirmed as a pair target
            if tid in self._confirmed_pairs.values():
                continue
            desc = self._get_track_descriptor_at_frame(trk, frame_id)
            if desc is not None:
                left_descs[tid] = desc

        for trk in tracks2.tracks():
            tid = trk.id
            # Skip if already confirmed
            if tid in self._confirmed_pairs:
                continue
            desc = self._get_track_descriptor_at_frame(trk, frame_id)
            if desc is not None:
                right_descs[tid] = desc

        if not left_descs or not right_descs:
            return

        # Build descriptor matrices
        left_ids = list(left_descs.keys())
        right_ids = list(right_descs.keys())

        left_mat = np.array([left_descs[tid] for tid in left_ids])
        right_mat = np.array([right_descs[tid] for tid in right_ids])

        # Normalize rows
        left_norms = np.linalg.norm(left_mat, axis=1, keepdims=True)
        right_norms = np.linalg.norm(right_mat, axis=1, keepdims=True)

        left_norms[left_norms == 0] = 1.0
        right_norms[right_norms == 0] = 1.0

        left_normed = left_mat / left_norms
        right_normed = right_mat / right_norms

        # Cosine distance matrix: 1 - dot product
        similarity = np.dot(left_normed, right_normed.T)
        cost_matrix = 1.0 - similarity

        # Greedy 1-to-1 matching
        n1, n2 = cost_matrix.shape
        used_right = set()

        # Sort all cells by cost
        indices = []
        for i in range(n1):
            for j in range(n2):
                if cost_matrix[i, j] <= self._distance_threshold:
                    indices.append((cost_matrix[i, j], i, j))
        indices.sort()

        used_left = set()
        for cost, i, j in indices:
            if i in used_left or j in used_right:
                continue
            used_left.add(i)
            used_right.add(j)

            lid = left_ids[i]
            rid = right_ids[j]
            pair_key = (lid, rid)

            self._pair_votes[pair_key] = self._pair_votes.get(pair_key, 0) + 1

            # Check if pair is confirmed
            if self._pair_votes[pair_key] >= self._min_frames:
                self._confirmed_pairs[rid] = lid

    # --------------------------------------------------------------------------
    def _get_track_descriptor_at_frame(self, track, frame_id):
        """Extract descriptor from a track's detection at the given frame."""
        for state in track:
            if state.frame_id == frame_id:
                det = state.detection
                if det is not None and det.descriptor is not None:
                    desc = det.descriptor
                    arr = np.array(desc.todoublearray())
                    if arr.size > 0:
                        return arr
        return None

    # --------------------------------------------------------------------------
    def _remap_tracks(self, tracks2, frame_id):
        """Rebuild tracks2 with confirmed right track IDs remapped to left IDs."""
        if tracks2 is None or not self._confirmed_pairs:
            return tracks2

        output_tracks = []

        for trk in tracks2.tracks():
            tid = trk.id
            if tid in self._confirmed_pairs:
                # Remap: create new track with the left track ID
                new_track = Track(id=self._confirmed_pairs[tid])
                for state in trk:
                    new_track.append(state)
                output_tracks.append(new_track)
            else:
                output_tracks.append(trk)

        return ObjectTrackSet(output_tracks)


# ==============================================================================
def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory

    module_name = 'python:kwiver.pytorch.pair_stereo_tracks'

    if process_factory.is_process_module_loaded(module_name):
        return

    process_factory.add_process(
        'pair_stereo_tracks_pytorch',
        'Pair stereo detections using deep descriptor cosine distance',
        PairStereoTracks)

    process_factory.mark_process_module_as_loaded(module_name)
