# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Whole-frame ONNX classifier (kwiver vital algorithm ``onnx_classifier``).

The onnxruntime counterpart of ``netharn_classifier``: same contract, same
output, no torch. Like that algorithm it abuses the ImageObjectDetector
interface, since kwiver has no classifier type -- it emits a single detection
covering the whole frame whose type carries every class probability.

Scores match netharn exactly. netharn decodes with
``CategoryTree.decision(..., criterion='entropy')``, which for the flat class
trees these models use reduces to plain argmax / max-probability (verified
against the torch models), so the frame score here is ``max(probs)``.

Example:
    >>> # xdoctest: +REQUIRES(env:VIAME_SMOKE)
    >>> self = OnnxClassifier()
    >>> self.set_configuration(dict(model='/path/to/model.onnx', device='cpu'))
    >>> dets = self.detect(self.demo_image())
"""
from __future__ import annotations

import numpy as np

from kwiver.vital.algo import ImageObjectDetector


def _vital_config_update(cfg, cfg_in):
    if isinstance(cfg_in, dict):
        for key, value in cfg_in.items():
            if cfg.has_value(key):
                cfg.set_value(key, str(value))
    else:
        cfg.merge_config(cfg_in)
    return cfg


class OnnxClassifier(ImageObjectDetector):
    """Full-frame classifier backed by an ONNX graph."""

    def __init__(self):
        ImageObjectDetector.__init__(self)
        self._config = {
            "model": "",           # package dir / .onnx / .zip
            "device": "cpu",       # cpu | cuda | cuda:N
            "batch_size": "1",     # one frame per call, so this stays 1
            "negative_class": "",  # rename this class to "no_<model name>"
        }
        self._predictor = None
        self._negative_name = ""

    # -- kwiver config plumbing --
    def get_configuration(self):
        cfg = super(ImageObjectDetector, self).get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        import os
        cfg = self.get_configuration()
        _vital_config_update(cfg, cfg_in)
        for key in self._config.keys():
            self._config[key] = str(cfg.get_value(key))

        from viame.onnx.onnx_clf_predictor import OnnxClassifierPredictor
        self._predictor = OnnxClassifierPredictor(
            self._config["model"],
            device=self._config["device"] or "cpu",
            batch_size=int(self._config["batch_size"] or 1))

        if self._config["negative_class"]:
            basename = os.path.basename(self._config["model"])
            basename = os.path.splitext(basename)[0]
            self._negative_name = "no_" + basename
        return True

    def check_configuration(self, cfg):
        if not cfg.get_value("model"):
            print("OnnxClassifier: a 'model' package/onnx path is required")
            return False
        return True

    # -- class renaming (mirrors NetharnClassifier) --
    def rename_class(self, classname):
        if self._config["negative_class"] and \
           classname == self._config["negative_class"]:
            return self._negative_name
        return classname

    def rename_classes(self, classnames):
        return [self.rename_class(name) for name in classnames]

    # -- inference --
    def detect(self, image_data):
        from kwiver.vital.types import (BoundingBoxD, DetectedObject,
                                        DetectedObjectSet, DetectedObjectType)
        if self._predictor is None:
            raise RuntimeError("OnnxClassifier: set_configuration first")

        full_rgb = image_data.asarray().astype("uint8")
        height, width = full_rgb.shape[:2]

        probs = self._predictor.predict([full_rgb])[0]
        class_names = self.rename_classes(self._predictor.category_names)
        score = float(np.max(probs)) if probs.size else 0.0

        detected_objects = DetectedObjectSet()
        object_type = DetectedObjectType(list(class_names),
                                         [float(p) for p in probs])
        detected_objects.add(DetectedObject(
            BoundingBoxD(0, 0, width, height), score, object_type))
        return detected_objects

    @classmethod
    def demo_image(cls):
        from kwiver.vital.types import Image, ImageContainer
        return ImageContainer(Image(np.zeros((64, 64, 3), dtype=np.uint8)))


def __vital_algorithm_register__():
    from viame.core.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        OnnxClassifier,
        "onnx_classifier",
        "Whole-frame ONNX classifier (onnxruntime, no torch)",
    )
