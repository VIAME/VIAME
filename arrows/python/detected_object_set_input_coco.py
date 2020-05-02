# ckwg +29
# Copyright 2020 by Kitware, Inc.
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

from __future__ import print_function

import json

from vital.algo import DetectedObjectSetInput
import vital.types as vt

class DetectedObjectSetInputCoco(DetectedObjectSetInput):
    def __init__(self):
        DetectedObjectSetInput.__init__(self)
        self.loaded = False
        self.file = None

    def get_configuration(self):
        cfg = super(DetectedObjectSetInput, self).get_configuration()
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)

    def check_configuration(self, cfg):
        return True

    def open(self, file_name):
        self.file = open(file_name)
        self.loaded = False

    def close(self):
        if self.file:
            self.file.close()

    def read_set(self):
        self._ensure_loaded()
        if self.frame >= self.stop_frame:
            return None
        fname, annots = self.frame_info.get(self.frame, ('', ()))
        det_objs = self.__to_detected_object_set(annots)
        self.frame += 1
        return det_objs, fname

    def read_set_by_path(self, image_path):
        self._ensure_loaded()
        annots = self.frame_info_by_path.get(image_path, ())
        return self.__to_detected_object_set(annots)

    def __to_detected_object_set(self, annots):
        """Convert list of annotations to a DetectedObjectSet"""
        det_objs = []
        for ann in annots:
            x, y, w, h = ann['bbox']
            score = ann.get('score', 1.)
            det_objs.append(vt.DetectedObject(
                bbox=vt.BoundingBox(x, y, x + w, y + h),
                confidence=score,
                classifications=vt.DetectedObjectType(
                    self.categories[ann['category_id']],
                    score,
                ),
            ))
        return vt.DetectedObjectSet(det_objs)

    def _ensure_loaded(self):
        if self.loaded:
            return
        data = json.load(self.file)

        categories = {cat['id']: cat['name'] for cat in data['categories']}
        assert len(categories) == len(data['categories'])
        assert len(categories) == len(set(categories.values()))

        # For compatibility with other parts of KWIVER, we treat image
        # IDs as frame numbers

        # Map from frame number to a pair of image name and list of
        # detections
        frame_info = {im['id']: (im['file_name'], [])
                      for im in data['images']}
        assert len(frame_info) == len(data['images'])
        for ann in data['annotations']:
            frame_info[ann['image_id']][1].append(ann)
        frame_info_by_path = {}
        for fname, annots in frame_info.values():
            frame_info_by_path.setdefault(fname, []).extend(annots)

        if frame_info:
            self.frame, self.stop_frame = min(frame_info), max(frame_info) + 1
        else:
            self.frame, self.stop_frame = 0, 0
        self.categories = categories
        self.frame_info = frame_info
        self.frame_info_by_path = frame_info_by_path
        self.loaded = True

def __vital_algorithm_register__():
    from vital.algo import algorithm_factory
    implementation_name = "coco"
    if algorithm_factory.has_algorithm_impl_name(
            DetectedObjectSetInputCoco.static_type_name(),
            implementation_name,
    ):
        return
    algorithm_factory.add_algorithm(
        implementation_name,
        "Read detections in from COCO-style JSON",
        DetectedObjectSetInputCoco,
    )
    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
