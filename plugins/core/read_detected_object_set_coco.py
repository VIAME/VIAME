# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Read detections from COCO-format JSON files.

The COCO format is a widely used annotation format that contains:
- categories: list of category definitions with 'id' and 'name'
- images: list of image definitions with 'id' and 'file_name'
- annotations: list of annotations with 'image_id', 'category_id', 'bbox', and optionally 'score' and 'segmentation'
"""

from __future__ import print_function

import json

from kwiver.vital.algo import DetectedObjectSetInput
import kwiver.vital.types as vt


class ReadDetectedObjectSetCoco(DetectedObjectSetInput):
    """
    Detected object set reader for COCO-format JSON files.

    COCO format contains:
    - categories: list of {id, name} category definitions
    - images: list of {id, file_name} image definitions
    - annotations: list of {image_id, category_id, bbox, score?, segmentation?}

    The bbox is in [x, y, width, height] format.
    """

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
        """Convert list of annotations to a DetectedObjectSet."""
        det_objs = []
        for ann in annots:
            x, y, w, h = ann['bbox']
            score = ann.get('score', 1.)
            do = vt.DetectedObject(
                bbox=vt.BoundingBoxD(x, y, x + w, y + h),
                confidence=score,
                classifications=vt.DetectedObjectType(
                    self.categories[ann['category_id']],
                    score,
                ),
            )
            if 'segmentation' in ann:
                seg = ann['segmentation']
                # Support [x1, y1, ..., xn, yn] and [[x1, y1, ..., xn, yn]]
                if len(seg) == 1:
                    seg = seg[0]
                if len(seg) % 2 != 0:
                    msg = "Polygon must have an even number of coordinates"
                    raise NotImplementedError(msg)
                do.set_flattened_polygon(seg)
            det_objs.append(do)
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
    from kwiver.vital.algo import algorithm_factory

    implementation_name = "coco"

    if algorithm_factory.has_algorithm_impl_name(
            ReadDetectedObjectSetCoco.static_type_name(),
            implementation_name,
    ):
        return

    algorithm_factory.add_algorithm(
        implementation_name,
        "Read detections from COCO-style JSON format",
        ReadDetectedObjectSetCoco,
    )

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
