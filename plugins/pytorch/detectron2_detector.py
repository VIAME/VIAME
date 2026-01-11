# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Detectron2 object detector implementation for VIAME.

This module provides a KWIVER ImageObjectDetector implementation using
Facebook's Detectron2 framework. It supports various architectures including
Faster R-CNN, Mask R-CNN, and other detection models.

Dependencies:
    - detectron2: Can be installed via geowatch_tpl or directly from Facebook
    - torch, numpy, einops

Example usage:
    >>> from pytorch.detectron2_detector import Detectron2Detector
    >>> detector = Detectron2Detector()
    >>> cfg_in = dict(
    ...     checkpoint_fpath='COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
    ...     base='COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
    ... )
    >>> detector.set_configuration(cfg_in)
    >>> detected_objects = detector.detect(image_data)
"""

from __future__ import print_function

import os

import numpy as np
import scriptconfig as scfg
import ubelt as ub

from kwiver.vital.algo import ImageObjectDetector

from .utilities import (
    vital_config_update,
    kwimage_to_kwiver_detections,
    resolve_device_str,
    register_vital_algorithm,
    parse_bool,
)


class Detectron2DetectorConfig(scfg.DataConfig):
    """
    Configuration for :class:`Detectron2Detector`.

    Attributes:
        checkpoint_fpath: Path to Detectron2 checkpoint file or model zoo name
        base: Base model configuration (e.g., 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml')
        cfg: Optional custom configuration overrides (YAML string or path)
        score_thresh: Minimum confidence score for detections (default 0.0)
        nms_thresh: Non-maximum suppression threshold (default 0.0, meaning no NMS)
        device: Device to run inference on ('auto', 'cpu', 'cuda', 'cuda:N')
        class_names: Comma-separated list of class names (optional, for enriching model)
    """
    checkpoint_fpath = scfg.Value(
        'noop',
        help='Path to Detectron2 checkpoint (.pth/.pkl) or model zoo identifier'
    )
    base = scfg.Value(
        'auto',
        help='Base config from model zoo (e.g., COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml)'
    )
    cfg = scfg.Value(
        '',
        help='Custom config overrides as YAML string or path to config file'
    )
    score_thresh = scfg.Value(
        0.0,
        help='Minimum detection confidence threshold'
    )
    nms_thresh = scfg.Value(
        0.0,
        help='NMS threshold (0.0 means no NMS)'
    )
    device = scfg.Value(
        'auto',
        help='Device to run on: auto, cpu, cuda, or cuda:N'
    )
    class_names = scfg.Value(
        '',
        help='Comma-separated class names to enrich checkpoint metadata'
    )

    def __post_init__(self):
        super().__post_init__()
        # Parse score_thresh and nms_thresh as floats
        import kwutil
        self.score_thresh = kwutil.Yaml.coerce(self.score_thresh)
        self.nms_thresh = kwutil.Yaml.coerce(self.nms_thresh)


class Detectron2Detector(ImageObjectDetector):
    """
    Implementation of ImageObjectDetector using Detectron2.

    Detectron2 is Facebook AI Research's next generation library for
    object detection and segmentation. This wrapper enables using
    Detectron2 models within the KWIVER pipeline framework.

    Supported architectures:
        - Faster R-CNN
        - Mask R-CNN
        - RetinaNet
        - And other models from the Detectron2 model zoo

    CommandLine:
        xdoctest -m plugins/pytorch/detectron2_detector.py Detectron2Detector --show

    Example:
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/VIAME/plugins'))
        >>> from pytorch.detectron2_detector import *  # NOQA
        >>> self = Detectron2Detector()
        >>> image_data = self.demo_image()
        >>> cfg_in = dict(
        >>>     checkpoint_fpath='COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
        >>>     base='COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
        >>> )
        >>> self.set_configuration(cfg_in)
        >>> detected_objects = self.detect(image_data)
    """

    def __init__(self):
        ImageObjectDetector.__init__(self)
        self._config = Detectron2DetectorConfig()
        self._predictor = None
        self._classes = None

    def demo_image(self):
        """
        Returns an image which can be run through the detector.

        Downloads and returns a sample image of sea lions for testing.

        Returns:
            ImageContainer: an image of sea lions
        """
        from PIL import Image as PILImage
        from kwiver.vital.util import VitalPIL
        from kwiver.vital.types import ImageContainer

        url = 'https://data.kitware.com/api/v1/file/6011a5ae2fa25629b919fe6c/download'
        image_fpath = ub.grabdata(
            url, fname='sealion2010.jpg', appname='viame',
            hash_prefix='f016550faa2c96ef4fdca', hasher='sha512')
        pil_img = PILImage.open(image_fpath)
        image_data = ImageContainer(VitalPIL.from_pil(pil_img))
        return image_data

    def get_configuration(self):
        """
        Get the algorithm configuration.

        Returns:
            kwiver.vital.config.config.Config: Configuration object
        """
        cfg = super(ImageObjectDetector, self).get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        """
        Set the algorithm configuration.

        Args:
            cfg_in: Configuration dictionary or Config object

        Returns:
            bool: True on success
        """
        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)

        for key in self._config.keys():
            self._config[key] = str(cfg.get_value(key))

        self._config.__post_init__()

        # Set underscore-prefixed attributes for convenient access
        for key, value in self._config.items():
            setattr(self, "_" + key, value)

        self._build_model()
        return True

    def _build_model(self):
        """
        Build and initialize the Detectron2 predictor.

        This method handles:
        - Setting up the correct device
        - Importing Detectron2 (via geowatch_tpl or direct import)
        - Configuring the model from checkpoint and base config
        - Setting up class names if provided
        """
        # Windows-specific workaround
        if os.name == 'nt':
            os.environ["KWIMAGE_DISABLE_TORCHVISION_NMS"] = "1"

        device = resolve_device_str(self._device)
        print(f"[Detectron2Detector] Loading model on {device}")

        # Try to import detectron2, first via geowatch_tpl, then directly
        try:
            import geowatch_tpl
            detectron2 = geowatch_tpl.import_submodule('detectron2')  # NOQA
            from geowatch.tasks.detectron2 import predict as d2pred
            use_geowatch = True
        except ImportError:
            # Fall back to direct detectron2 import
            import detectron2  # NOQA
            use_geowatch = False

        if use_geowatch:
            # Use geowatch's Detectron2 predictor interface
            config = d2pred.DetectronPredictCLI()
            config['checkpoint_fpath'] = self._checkpoint_fpath
            config['base'] = self._base
            self._predictor = d2pred.Detectron2Predictor(config)
            self._predictor.prepare_config_backend()
        else:
            # Use direct Detectron2 API
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            from detectron2 import model_zoo

            cfg = get_cfg()

            # Load base configuration
            if self._base and self._base != 'auto':
                try:
                    cfg.merge_from_file(model_zoo.get_config_file(self._base))
                except Exception:
                    # Try as direct path
                    if ub.Path(self._base).exists():
                        cfg.merge_from_file(self._base)

            # Load checkpoint
            if self._checkpoint_fpath and self._checkpoint_fpath != 'noop':
                checkpoint_path = self._checkpoint_fpath
                try:
                    # Try model zoo first
                    checkpoint_path = model_zoo.get_checkpoint_url(self._checkpoint_fpath)
                except Exception:
                    # Use as direct path
                    pass
                cfg.MODEL.WEIGHTS = checkpoint_path

            # Apply custom config if provided
            if self._cfg:
                if ub.Path(self._cfg).exists():
                    cfg.merge_from_file(self._cfg)
                else:
                    # Parse as YAML string
                    import yaml
                    custom_cfg = yaml.safe_load(self._cfg)
                    if custom_cfg:
                        cfg.merge_from_list(
                            [item for pair in custom_cfg.items() for item in pair]
                        )

            # Set device
            cfg.MODEL.DEVICE = device

            # Set thresholds
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(self._score_thresh)
            if float(self._nms_thresh) > 0:
                cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = float(self._nms_thresh)

            self._predictor = DefaultPredictor(cfg)

        # Set up class names
        if self._class_names:
            self._classes = [c.strip() for c in self._class_names.split(',')]
        else:
            # Try to get class names from model metadata
            self._classes = self._get_class_names()

        if self._classes:
            print(f"[Detectron2Detector] Model loaded with {len(self._classes)} classes")
        else:
            print("[Detectron2Detector] Model loaded (class names not available)")

    def _get_class_names(self):
        """
        Attempt to retrieve class names from the model.

        Returns:
            list or None: List of class names if available
        """
        try:
            # Try to get from detectron2's MetadataCatalog
            from detectron2.data import MetadataCatalog
            # Common COCO metadata
            metadata = MetadataCatalog.get("coco_2017_val")
            return list(metadata.thing_classes)
        except Exception:
            pass

        return None

    def check_configuration(self, cfg):
        """
        Check if the configuration is valid.

        Args:
            cfg: Configuration object

        Returns:
            bool: True if configuration is valid
        """
        if not cfg.has_value("checkpoint_fpath"):
            print("[Detectron2Detector] A checkpoint path must be specified!")
            return False

        checkpoint = cfg.get_value("checkpoint_fpath")
        if checkpoint == 'noop' or not checkpoint:
            print("[Detectron2Detector] A valid checkpoint path must be specified!")
            return False

        return True

    def detect(self, image_data):
        """
        Perform object detection on an image.

        Args:
            image_data (kwiver.vital.types.ImageContainer): Input image

        Returns:
            kwiver.vital.types.DetectedObjectSet: Detected objects
        """
        import kwimage
        import torch

        # Convert kwiver image to numpy array
        full_rgb = image_data.asarray().astype('uint8')

        # Check if using geowatch predictor or direct detectron2
        if hasattr(self._predictor, 'predict_image'):
            # Geowatch interface expects CHW tensor
            import einops
            im_chw = torch.Tensor(einops.rearrange(full_rgb, 'h w c -> c h w'))
            detections = self._predictor.predict_image(im_chw)
        else:
            # Direct detectron2 interface expects BGR numpy array
            import cv2
            im_bgr = cv2.cvtColor(full_rgb, cv2.COLOR_RGB2BGR)

            with torch.no_grad():
                outputs = self._predictor(im_bgr)

            # Convert detectron2 outputs to kwimage.Detections
            instances = outputs["instances"].to("cpu")

            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            class_idxs = instances.pred_classes.numpy()

            # Get class names
            if self._classes:
                classes = self._classes
            else:
                # Use integer class IDs as strings
                unique_classes = np.unique(class_idxs)
                classes = [str(i) for i in range(max(unique_classes) + 1)]

            detections = kwimage.Detections(
                boxes=kwimage.Boxes(boxes, 'ltrb'),
                scores=scores,
                class_idxs=class_idxs,
                classes=classes,
            )

            # Handle segmentation masks if available
            if instances.has("pred_masks"):
                masks = instances.pred_masks.numpy()
                segmentations = []
                for mask in masks:
                    seg = kwimage.Mask(mask, format='c_mask')
                    segmentations.append(seg)
                detections.data['segmentations'] = segmentations

        # Apply score threshold
        score_thresh = float(self._score_thresh)
        if score_thresh > 0:
            flags = detections.scores >= score_thresh
            detections = detections.compress(flags)

        # Convert to kwiver format
        output = kwimage_to_kwiver_detections(detections)
        return output


def __vital_algorithm_register__():
    register_vital_algorithm(
        Detectron2Detector,
        "detector_detectron2",
        "Detectron2 object detection routine"
    )
