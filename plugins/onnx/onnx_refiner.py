# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
ONNX detection reclassifier (kwiver vital algorithm ``onnx`` for
``refine_detections``).

The onnxruntime counterpart of the ``netharn`` refiner: crops a chip per input
detection, classifies it, and rewrites the detection's type. Chip geometry,
filters, target-scale normalization and prior blending are ports of
``viame.pytorch.netharn_refiner`` and must stay behaviourally identical to it.
"""
from __future__ import annotations

import math

import numpy as np

from kwiver.vital.algo import RefineDetections


def _vital_config_update(cfg, cfg_in):
    if isinstance(cfg_in, dict):
        for key, value in cfg_in.items():
            if cfg.has_value(key):
                cfg.set_value(key, str(value))
    else:
        cfg.merge_config(cfg_in)
    return cfg


def _strip_taxon_id(name):
    """``'rock-1234'`` -> ``'rock'`` (matches the netharn refiner)."""
    i = name.rfind('-')
    return name[:i] if i > 0 and name[i + 1:].isdigit() else name


def _safe_crop(img, x1, y1, x2, y2):
    """Crop, zero-padding wherever the box falls outside the image."""
    height, width = img.shape[:2]
    out_h, out_w = max(y2 - y1, 0), max(x2 - x1, 0)
    if out_h == 0 or out_w == 0:
        return np.zeros((1, 1, img.shape[2]), dtype=img.dtype)
    out = np.zeros((out_h, out_w, img.shape[2]), dtype=img.dtype)
    sx1, sy1 = max(x1, 0), max(y1, 0)
    sx2, sy2 = min(x2, width), min(y2, height)
    if sx2 > sx1 and sy2 > sy1:
        out[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = img[sy1:sy2, sx1:sx2]
    return out


class OnnxRefiner(RefineDetections):
    """Reclassify detections with an ONNX image classifier."""

    def __init__(self):
        RefineDetections.__init__(self)
        self._config = {
            'model': "",
            'device': "cpu",
            'batch_size': "4",
            'area_pivot': "0",
            'area_lower_bound': "0",
            'area_upper_bound': "0",
            'border_exclude': "-1",
            'chip_method': "",
            'chip_width': "",
            'chip_expansion': "1.0",
            'average_prior': "False",
            'prior_weight': "0.5",
            'prior_ignore_class': "",
            'prior_taxonomy_file': "",
            'scale_type_file': "",
        }
        self._predictor = None
        self._taxonomy = {}
        self._target_type_scales = {}

    # -- kwiver config plumbing --
    def get_configuration(self):
        cfg = super(RefineDetections, self).get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        from viame.compat import strtobool
        cfg = self.get_configuration()
        _vital_config_update(cfg, cfg_in)
        for key in self._config.keys():
            self._config[key] = str(cfg.get_value(key))

        self._border_exclude = int(self._config['border_exclude'] or -1)
        self._area_lower_bound = float(self._config['area_lower_bound'] or 0)
        self._area_upper_bound = float(self._config['area_upper_bound'] or 0)
        self._chip_expansion = float(self._config['chip_expansion'] or 1.0)
        self._average_prior = bool(strtobool(
            self._config['average_prior'] or "False"))
        self._prior_weight = float(self._config['prior_weight'] or 0.5)
        self._prior_ignore_class = self._config['prior_ignore_class']

        self._taxonomy = {}
        if self._config['prior_taxonomy_file']:
            for line in open(self._config['prior_taxonomy_file']):
                toks = line.split()
                if len(toks) > 1:
                    for tok in toks[1:]:
                        self._taxonomy[tok] = toks[0]

        self._target_type_scales = {}
        if self._config['scale_type_file']:
            for line in open(self._config['scale_type_file']):
                toks = line.split()
                if len(toks) > 1:
                    self._target_type_scales[toks[0]] = float(toks[1])

        batch_size = self._config['batch_size']
        batch_size = 4 if batch_size in ("", "auto") else int(batch_size)

        from viame.onnx.onnx_clf_predictor import OnnxClassifierPredictor
        self._predictor = OnnxClassifierPredictor(
            self._config['model'],
            device=self._config['device'] or "cpu",
            batch_size=batch_size)
        return True

    def check_configuration(self, cfg):
        if not cfg.get_value("model"):
            print("OnnxRefiner: a 'model' package/onnx path is required")
            return False
        return True

    # -- helpers --
    def compute_scale_factor(self, detections, min_scale=0.10, max_scale=10.0):
        """Mean sqrt(target_area / observed_area) over typed detections."""
        cumulative, count = 0.0, 0
        for item in detections:
            if item.type is None:
                continue
            class_lbl = item.type.get_most_likely_class()
            if class_lbl not in self._target_type_scales:
                continue
            box_area = float(item.bounding_box.width() *
                             item.bounding_box.height())
            if box_area < 1.0:
                continue
            cumulative += math.sqrt(
                self._target_type_scales[class_lbl] / box_area)
            count += 1
        output = 1.0 if count == 0 else cumulative / count
        return float(min(max(output, min_scale), max_scale))

    # -- inference --
    def refine(self, image_data, detections):
        import cv2
        from kwiver.vital.types import DetectedObjectSet, DetectedObjectType

        if len(detections) == 0:
            return detections
        if self._predictor is None:
            raise RuntimeError("OnnxRefiner: set_configuration first")

        img = image_data.asarray().astype('uint8')
        if img.ndim == 2:
            img = np.stack((img,) * 3, axis=-1)
        scale = 1.0
        img_max_x, img_max_y = img.shape[1], img.shape[0]

        if self._target_type_scales:
            scale = self.compute_scale_factor(detections)
            if scale != 1.0:
                img_max_x = int(img_max_x * scale)
                img_max_y = int(img_max_y * scale)
                img = cv2.resize(img, (img_max_x, img_max_y))

        image_chips, detection_ids = [], []
        for i, det in enumerate(detections):
            bbox = det.bounding_box
            bbox_min_x = int(bbox.min_x() * scale)
            bbox_max_x = int(bbox.max_x() * scale)
            bbox_min_y = int(bbox.min_y() * scale)
            bbox_max_y = int(bbox.max_y() * scale)

            method = self._config['chip_method']
            if method in ("fixed_width", "native_square"):
                if method == "fixed_width":
                    chip_width = int(self._config['chip_width'])
                else:
                    chip_width = max(bbox_max_x - bbox_min_x,
                                     bbox_max_y - bbox_min_y)
                half_width = int(chip_width / 2)
                bbox_min_x = int((bbox_min_x + bbox_max_x) / 2) - half_width
                bbox_min_y = int((bbox_min_y + bbox_max_y) / 2) - half_width
                bbox_max_x = bbox_min_x + chip_width
                bbox_max_y = bbox_min_y + chip_width

            if self._chip_expansion != 1.0:
                bbox_width = int((bbox_max_x - bbox_min_x) * self._chip_expansion)
                bbox_height = int((bbox_max_y - bbox_min_y) * self._chip_expansion)
                bbox_min_x = int((bbox_min_x + bbox_max_x) / 2 - bbox_width / 2)
                bbox_min_y = int((bbox_min_y + bbox_max_y) / 2 - bbox_height / 2)
                bbox_max_x = bbox_min_x + bbox_width
                bbox_max_y = bbox_min_y + bbox_height

            if self._border_exclude > 0:
                if bbox_min_x <= self._border_exclude:
                    continue
                if bbox_min_y <= self._border_exclude:
                    continue
                if bbox_max_x >= img_max_x - self._border_exclude:
                    continue
                if bbox_max_y >= img_max_y - self._border_exclude:
                    continue
            else:
                bbox_min_x = max(bbox_min_x, 0)
                bbox_min_y = max(bbox_min_y, 0)
                bbox_max_x = min(bbox_max_x, img_max_x)
                bbox_max_y = min(bbox_max_y, img_max_y)

            bbox_area = (bbox_max_x - bbox_min_x) * (bbox_max_y - bbox_min_y)
            if self._area_lower_bound > 0 and bbox_area < self._area_lower_bound:
                continue
            if self._area_upper_bound > 0 and bbox_area > self._area_upper_bound:
                continue

            image_chips.append(
                _safe_crop(img, bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y))
            detection_ids.append(i)

        probs = self._predictor.predict(image_chips)
        model_classes = list(self._predictor.category_names)

        output = DetectedObjectSet()
        next_id = 0
        for i, det in enumerate(detections):
            if next_id >= len(detection_ids) or i != detection_ids[next_id]:
                output.add(det)
                continue

            class_names = list(model_classes)
            class_scores = [float(p) for p in probs[next_id]]
            next_id += 1

            if self._average_prior and det.type is not None:
                w = self._prior_weight
                priors = det.type
                if self._taxonomy:
                    # hierarchical: pool prior mass by canonical group
                    group_mass = {}
                    for name in priors.class_names():
                        if name == self._prior_ignore_class:
                            continue
                        g = self._taxonomy.get(_strip_taxon_id(name))
                        if g is not None:
                            group_mass[g] = group_mass.get(g, 0.0) + priors.score(name)
                    for j in range(len(class_scores)):
                        g = self._taxonomy.get(_strip_taxon_id(class_names[j]))
                        class_scores[j] = class_scores[j] * (1.0 - w) \
                            + w * group_mass.get(g, 0.0)
                else:
                    # flat: matched by name, so mismatched vocabularies concatenate
                    for j in range(len(class_scores)):
                        class_scores[j] = class_scores[j] * (1.0 - w)
                    for name in priors.class_names():
                        if name == self._prior_ignore_class:
                            continue
                        weighted = priors.score(name) * w
                        if name in class_names:
                            class_scores[class_names.index(name)] += weighted
                        else:
                            class_names.append(name)
                            class_scores.append(weighted)

            det.type = DetectedObjectType(class_names, class_scores)
            output.add(det)

        return output


def __vital_algorithm_register__():
    from viame.core.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        OnnxRefiner,
        "onnx",
        "ONNX detection reclassifier (onnxruntime, no torch)",
    )
