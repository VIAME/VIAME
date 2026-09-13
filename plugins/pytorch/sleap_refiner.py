# This file is part of VIAME, and is distributed under an OSI-approved
# BSD 3-Clause License. See the root LICENSE file for details.
"""Add trainable SLEAP-NN keypoints to existing detections."""
from kwiver.vital.algo import RefineDetections
from kwiver.vital.types import DetectedObjectSet, Point2d
import numpy as np

from viame.pytorch.sleap_common import (
    SleapPredictor, as_rgb, crop_detection, read_config, transform_points,
)
from viame.pytorch.utilities import register_vital_algorithm


class SleapRefiner(RefineDetections):
    DEFAULTS = dict(weight='', device='auto', batch_size=16,
                    keypoint_threshold=0.2, overwrite_existing=False)

    def __init__(self):
        RefineDetections.__init__(self)
        self.options = dict(self.DEFAULTS)
        self.predictor = None

    def get_configuration(self):
        cfg = super().get_configuration()
        for key, value in self.options.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)
        options = read_config(self.DEFAULTS, cfg)
        if options['batch_size'] < 1 or not np.isfinite(options['keypoint_threshold']) or options['keypoint_threshold'] < 0:
            raise ValueError('batch_size must be positive and keypoint_threshold nonnegative')
        self.predictor = SleapPredictor(options['weight'], options['device'], options['keypoint_threshold'])
        self.options = options
        return True

    def check_configuration(self, cfg_in):
        from pathlib import Path
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)
        try:
            options = read_config(self.DEFAULTS, cfg)
            return (Path(options['weight']).is_file() and options['batch_size'] > 0
                    and np.isfinite(options['keypoint_threshold']) and options['keypoint_threshold'] >= 0)
        except (ValueError, KeyError):
            return False

    def refine(self, image_data, detections):
        if detections is None or len(detections) == 0:
            return DetectedObjectSet()
        if self.predictor is None:
            raise RuntimeError('SLEAP refiner has not been configured')
        image = as_rgb(image_data.asarray())
        h, w = image.shape[:2]
        result = [det.clone() for det in detections]
        names = self.predictor.names
        needed = {name.lower() for name in names}
        pending = []

        def flush():
            if not pending:
                return
            points, scores = self.predictor.predict([entry[1] for entry in pending])
            for (index, _, affine), crop_points, confidence in zip(pending, points, scores):
                det = result[index]
                existing = {name.lower(): (name, pt) for name, pt in det.keypoints.items()}
                if self.options['overwrite_existing']:
                    # Replace only this model's named slots; keep unrelated points.
                    det.clear_keypoints()
                    for lower, (name, pt) in existing.items():
                        if lower not in needed:
                            det.add_keypoint(name, pt)
                for name, xy, score in zip(names, transform_points(crop_points, affine, inverse=True), confidence):
                    if name.lower() in existing and not self.options['overwrite_existing']:
                        continue
                    if not np.isfinite(xy).all() or not np.isfinite(score) or score < self.options['keypoint_threshold']:
                        continue
                    if 0 <= xy[0] < w and 0 <= xy[1] < h:
                        point = Point2d()
                        point.value = [float(xy[0]), float(xy[1])]
                        det.add_keypoint(name, point)
            pending.clear()

        for index, det in enumerate(result):
            if not self.options['overwrite_existing'] and needed.issubset({name.lower() for name in det.keypoints}):
                continue
            box = det.bounding_box
            cropped = crop_detection(image, [box.min_x(), box.min_y(), box.max_x(), box.max_y()],
                                     self.predictor.size, self.predictor.padding)
            if cropped is None:
                continue
            crop, affine = cropped
            pending.append((index, crop, affine))
            if len(pending) == self.options['batch_size']:
                flush()
        flush()
        output = DetectedObjectSet()
        for det in result:
            output.add(det)
        return output


def __vital_algorithm_register__():
    register_vital_algorithm(SleapRefiner, 'sleap', 'SLEAP-NN keypoints on existing detections')
