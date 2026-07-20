# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #
"""
VIAME ImageObjectDetector wrapping a kwcoco_detector_kit ONNX export.

The kit's OnnxPredictor handles all inference; this file is a thin kwiver
adapter. No PyTorch or DEIMv2 dependency — only onnxruntime + kwimage.

Export a package with:
    python -m kwcoco_detector_kit export-onnx <workdir>

Then point the ``package`` config key at the resulting directory or .zip.

Example:
    >>> # xdoctest: +REQUIRES(env:VIAME_SMOKE)
    >>> import sys, pathlib
    >>> sys.path.insert(0, str(pathlib.Path('~/code/VIAME/plugins/pytorch').expanduser()))
    >>> from kwcoco_detector_kit_detector import *  # NOQA
    >>> package = '/data/users/jon.crall/kcd_sealion/workdirs/.../export/'
    >>> image_data = KwcocoDetectorKitDetector.demo_image()
    >>> self = KwcocoDetectorKitDetector()
    >>> self.set_configuration(dict(package=package, device='cpu'))
    >>> detected_objects = self.detect(image_data)
    >>> print(f'found {len(detected_objects)} detections')
"""
from __future__ import annotations

import kwconf

from kwiver.vital.algo import ImageObjectDetector
from viame.pytorch.utilities import (
    image_to_rgb_numpy,
    kwimage_to_kwiver_detections,
    register_vital_algorithm,
    vital_config_update,
)


class KwcocoDetectorKitConfig(kwconf.Config):
    """Configuration for KwcocoDetectorKitDetector."""

    package = kwconf.Value(None, help=(
        'Path to the exported ONNX package directory or .zip archive. '
        'Must contain a .onnx file and (optionally) a .modelspec.json sidecar.'
    ))
    device = kwconf.Value('cpu', help='onnxruntime device: "cpu", "cuda", "cuda:0"')
    score_thresh = kwconf.Value(None, parser=float, help=(
        'Detection score threshold. Defaults to the modelspec value when None.'
    ))
    nms_thresh = kwconf.Value(None, parser=float, help=(
        'NMS IoU threshold. Advisory only — NMS is baked into the ONNX graph.'
    ))

    def __post_init__(self):
        super().__post_init__()
        if self.score_thresh is not None:
            self.score_thresh = float(self.score_thresh)
        if self.nms_thresh is not None:
            self.nms_thresh = float(self.nms_thresh)


class KwcocoDetectorKitDetector(ImageObjectDetector):
    """
    VIAME detector wrapping a kwcoco_detector_kit ONNX export.

    Point ``package`` at the directory produced by ``python -m kwcoco_detector_kit
    export-onnx`` — it contains a ``.onnx`` model and a ``.modelspec.json``
    sidecar that describe all inference parameters (input size, preprocessing,
    category names).  No PyTorch or DEIMv2 installation required.
    """

    def __init__(self):
        ImageObjectDetector.__init__(self)
        self._config = KwcocoDetectorKitConfig()
        self._predictor = None

    # ------------------------------------------------------------------
    # kwiver config protocol
    # ------------------------------------------------------------------

    def get_configuration(self):
        cfg = super(ImageObjectDetector, self).get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value) if value is not None else '')
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)
        for key in self._config.keys():
            raw = cfg.get_value(key)
            self._config[key] = None if raw == '' else raw
        self._config.__post_init__()
        self._build_model()
        return True

    def check_configuration(self, cfg):
        return cfg.has_value('package') and cfg.get_value('package') != ''

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_model(self):
        # Vendored predictor — VIAME inference does NOT depend on
        # kwcoco_detector_kit being installed. See the provenance header in
        # kwcoco_detector_kit_onnx_predictor.py; resync with the kit's
        # dev/vendor_onnx_to_viame.py.
        from viame.pytorch.kwcoco_detector_kit_onnx_predictor import OnnxPredictor
        self._predictor = OnnxPredictor(
            self._config.package,
            device=self._config.device or 'cpu',
            score_thresh=self._config.score_thresh,
            nms_thresh=self._config.nms_thresh,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, image_data):
        if self._predictor is None:
            raise RuntimeError(
                'KwcocoDetectorKitDetector: call set_configuration before detect'
            )
        rgb = image_to_rgb_numpy(image_data)
        dets = self._predictor.predict_image_kwimage(rgb)
        dets = dets.numpy()
        return kwimage_to_kwiver_detections(dets)

    # ------------------------------------------------------------------
    # Convenience / smoke-test helpers
    # ------------------------------------------------------------------

    @classmethod
    def demo_image(cls):
        """Return a kwiver ImageContainer wrapping a synthetic test image."""
        import numpy as np
        from kwiver.vital.types import Image, ImageContainer
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        return ImageContainer(Image(arr))


def __vital_algorithm_register__():
    register_vital_algorithm(
        KwcocoDetectorKitDetector,
        'kwcoco_detector_kit',
        'DEIMv2 / kwcoco_detector_kit ONNX detector',
    )
