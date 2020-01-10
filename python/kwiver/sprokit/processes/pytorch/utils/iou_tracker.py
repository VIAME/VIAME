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

class IOUTracker(object):
    def __init__(self, iou_accept_threshold, iou_reject_threshold):
        self._iou_accept_threshold = iou_accept_threshold
        self._iou_reject_threshold = iou_reject_threshold

    def __call__(self, track_set, track_state_list):
        return self._track_iou(track_set, track_state_list)

    def _track_iou(self, track_set, track_state_list):
        """Append track states with a sufficiently high IOU and mark the
        destination tracks as updated.  Return a list of unused track
        states.

        """
        # IOU based tracking
        if not track_state_list:
            return []

        track_state_list = track_state_list[:]

        for track in track_set.iter_active():
            # If there is exactly one detection at or above the
            # reject threshold, and it's additionally at or above
            # the accept threshold, add it to the track.
            # Otherwise do nothing.

            # Get IOUs
            ious = [self._iou_score(track[-1].ref_bbox, x.ref_bbox)
                    for x in track_state_list]
            # Get dets with IOU over reject threshold
            nonreject_dets = [(ts, iou) for ts, iou in zip(track_state_list, ious)
                              if iou >= self._iou_reject_threshold]
            # Do nothing unless there's only one det and it's at
            # or above the accept threshold
            if len(nonreject_dets) != 1:
                continue
            match, iou = nonreject_dets[0]
            if iou < self._iou_accept_threshold:
                continue

            # Add the matching det
            track.updated_flag = True
            track_set.update_track(track.track_id, match)
            # remove matching detection from detections
            del track_state_list[track_state_list.index(match)]

        return track_state_list

    def _iou_score(self, bbox1, bbox2):
        """
        Calculates the intersection-over-union of two bounding boxes.
        Args:
            bbox1 list of float: bounding box in format x1,y1,w,h
            bbox2 list of float: bounding box in format x1,y1,w,h.
        Returns:
            int: intersection-over-union of bbox1, bbox2
        """

        (x1, y1, w1, h1) = bbox1
        (x2, y2, w2, h2) = bbox2

        # get the overlap rectangle
        overlap_x0 = max(x1, x2)
        overlap_y0 = max(y1, y2)
        overlap_x1 = min(x1 + w1, x2 + w2)
        overlap_y1 = min(y1 + h1, y2 + h2)

        # check if there is an overlap
        if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
            return 0

        # if yes, calculate the ratio of the overlap to each ROI size and the unified size
        size_1 = w1 * h1
        size_2 = w2 * h2
        size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
        size_union = size_1 + size_2 - size_intersection

        return float(size_intersection) / size_union
