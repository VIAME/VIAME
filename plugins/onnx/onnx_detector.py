# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Generic ONNX ImageObjectDetector for VIAME (kwiver vital algorithm ``onnx``).

A thin kwiver adapter around :class:`OnnxPredictor`: it runs any
object-detection ONNX graph described by a ``.modelspec.json`` sidecar (or by
explicit config), using onnxruntime. Unlike ``kwcoco_detector_kit`` (which is
pinned to that project's DEIMv2 export convention and lives in the pytorch
plugin tree), this detector is architecture-agnostic via the predictor's
pluggable decoder, so it belongs in ``plugins/onnx``.

Whole-frame only -- like every VIAME detector, wrap it in ``ocv_windowed`` to
tile large imagery (essential for these sea-lion models; see the pipelines).

Example:
    >>> # xdoctest: +REQUIRES(env:VIAME_SMOKE)
    >>> self = OnnxDetector()
    >>> self.set_configuration(dict(model='/path/to/pkg', device='cpu'))
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


def _image_to_rgb_numpy(image_data):
    """kwiver ImageContainer -> HxWx3 uint8 RGB.

    The container delivers channels in BGR order, so the last step reverses
    them; the model expects RGB. Matches the reference conversion used by the
    other VIAME detectors."""
    arr = image_data.image().asarray().astype("uint8")
    if arr.ndim == 2:
        arr = np.stack((arr,) * 3, axis=-1)
    elif arr.shape[2] == 1:
        arr = np.stack((arr[:, :, 0],) * 3, axis=-1)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.shape[2] == 3:
        arr = arr[:, :, ::-1].copy()   # BGR -> RGB
    return arr


def _to_kwiver_detections(dets, predictor):
    """[{label, bbox_xyxy, score}] -> DetectedObjectSet (integer xyxy box,
    single top-class DetectedObjectType per detection)."""
    try:
        from kwiver.vital.types import BoundingBoxD
    except ImportError:
        from kwiver.vital.types import BoundingBox as BoundingBoxD
    from kwiver.vital.types import (DetectedObjectSet, DetectedObject,
                                    DetectedObjectType)

    out = DetectedObjectSet()
    for d in dets:
        b = np.round(d["bbox_xyxy"]).astype(np.int32)
        bbox = BoundingBoxD(int(b[0]), int(b[1]), int(b[2]), int(b[3]))
        score = float(d["score"])
        dot = DetectedObjectType(predictor.class_name(int(d["label"])), score)
        out.add(DetectedObject(bbox, score, dot))
    return out


class OnnxDetector(ImageObjectDetector):
    """ImageObjectDetector backed by a generic ONNX graph."""

    def __init__(self):
        ImageObjectDetector.__init__(self)
        self._config = {
            "model": "",          # package dir / .onnx / .zip
            "device": "cpu",      # cpu | cuda | cuda:N
            "score_thresh": "",   # blank -> modelspec default
            "nms_thresh": "",     # blank -> modelspec default
            "decoder": "",        # blank -> modelspec default (detr/baked/yolo)
        }
        self._predictor = None

    # -- kwiver config plumbing --
    def get_configuration(self):
        cfg = super(ImageObjectDetector, self).get_configuration()
        for k, v in self._config.items():
            cfg.set_value(k, str(v))
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        _vital_config_update(cfg, cfg_in)
        for k in self._config.keys():
            self._config[k] = str(cfg.get_value(k))

        from viame.onnx.onnx_predictor import OnnxPredictor
        opt = lambda s: None if s in ("", "None") else s
        self._predictor = OnnxPredictor(
            self._config["model"],
            device=self._config["device"] or "cpu",
            score_thresh=(None if opt(self._config["score_thresh"]) is None
                          else float(self._config["score_thresh"])),
            nms_thresh=(None if opt(self._config["nms_thresh"]) is None
                        else float(self._config["nms_thresh"])),
            decoder=opt(self._config["decoder"]),
        )
        return True

    def check_configuration(self, cfg):
        if not cfg.get_value("model"):
            print("OnnxDetector: a 'model' package/onnx path is required")
            return False
        return True

    # -- inference --
    def detect(self, image_data):
        if self._predictor is None:
            raise RuntimeError("OnnxDetector: set_configuration first")
        rgb = _image_to_rgb_numpy(image_data)
        dets = self._predictor.predict_image(rgb)
        return _to_kwiver_detections(dets, self._predictor)

    @classmethod
    def demo_image(cls):
        from kwiver.vital.types import Image, ImageContainer
        return ImageContainer(Image(np.zeros((64, 64, 3), dtype=np.uint8)))


def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory
    impl = "onnx"
    if algorithm_factory.has_algorithm_impl_name(
            OnnxDetector.static_type_name(), impl):
        return
    algorithm_factory.add_algorithm(
        impl, "Generic ONNX object detector (onnxruntime, no torch)",
        OnnxDetector)
    algorithm_factory.mark_algorithm_as_loaded(impl)
