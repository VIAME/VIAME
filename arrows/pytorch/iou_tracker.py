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
        # IOU based tracking
        if track_state_list:
            for track in track_set.iter_active():
                # get det with highest iou
                best_match = max(track_state_list,
                        key=lambda x: self._iou_score(track[-1].bbox, x.bbox))
                # sort the track state list in order to check whether multiple bboxes overlap with the current track
                sorted_ts_list = sorted(track_state_list,
                        key=lambda x: self._iou_score(track[-1].bbox, x.bbox))
                best_iou_score = self._iou_score(track[-1].bbox, best_match.bbox)
                if best_iou_score >= self._iou_accept_threshold:
                    # if no other bbox with overlap larger than iou_reject_threshold
                    if (len(sorted_ts_list) >= 2 and
                        (self._iou_score(track[-1].bbox, sorted_ts_list[1].bbox)
                         < self._iou_reject_threshold)):
                        track.updated_flag = True
                        track_set.update_track(track.track_id, best_match)
                        # remove best matching detection from detections
                        del track_state_list[track_state_list.index(best_match)]
        return track_set, track_state_list

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
