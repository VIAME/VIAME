# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
SAM2-based point segmentation algorithm.

This module provides a concrete implementation of SegmentViaPoints using
SAM2 (Segment Anything Model 2) for point-based segmentation.
"""

import contextlib
import sys

import numpy as np
import scriptconfig as scfg

from kwiver.vital.algo import SegmentViaPoints

from viame.pytorch.utilities import vital_config_update, register_vital_algorithm


class SAM2SegmenterConfig(scfg.DataConfig):
    """Configuration for SAM2 segment via points algorithm."""

    checkpoint = scfg.Value('', help='Path to model checkpoint')
    cfg = scfg.Value('configs/sam2.1/sam2.1_hiera_b+.yaml', help='Path to model config')
    device = scfg.Value('cuda', help='Device to run on (cuda, cpu, auto)')


class SAM2Segmenter(SegmentViaPoints):
    """
    SAM2-based point segmentation implementation.

    Uses SAM2's point-based prompting to segment objects in images.
    """

    def __init__(self):
        SegmentViaPoints.__init__(self)
        self._config = SAM2SegmenterConfig()
        self._predictor = None
        self._model = None

    def get_configuration(self):
        cfg = super(SegmentViaPoints, self).get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)

        for key in self._config.keys():
            self._config[key] = str(cfg.get_value(key))

        self._config.__post_init__()

        for key, value in self._config.items():
            setattr(self, "_" + key, value)

        self._init_model()
        return True

    def check_configuration(self, cfg):
        return True

    def _log(self, message):
        print(f"[SAM2Segmenter] {message}", file=sys.stderr, flush=True)

    def _init_model(self):
        """Initialize the SAM2 model."""
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        checkpoint = self._config.checkpoint or None
        cfg = self._config.cfg
        device = self._config.device

        self._log("Initializing SAM2 model...")
        self._log(f"  Config: {cfg}")
        self._log(f"  Checkpoint: {checkpoint}")
        self._log(f"  Device: {device}")

        self._model = build_sam2(
            config_file=cfg,
            ckpt_path=checkpoint,
            device=device,
            mode='eval',
            apply_postprocessing=True,
        )
        self._predictor = SAM2ImagePredictor(self._model)
        self._log("model initialized successfully")

    def segment(self, image, points, point_labels):
        """
        Perform point-based segmentation on an image.

        Args:
            image: ImageContainer with the image to segment
            points: Vector of Point2d objects indicating prompt locations
            point_labels: Vector of int labels (1=foreground, 0=background)

        Returns:
            DetectedObjectSet containing segmented objects with masks
        """
        import torch

        from kwiver.vital.types import DetectedObjectSet, DetectedObject, DetectedObjectType
        try:
            from kwiver.vital.types import BoundingBoxD
        except ImportError:
            from kwiver.vital.types import BoundingBox as BoundingBoxD
        from kwiver.vital.types.types import ImageContainer, Image

        # Convert image to numpy array
        img_array = image.image().asarray()

        # Ensure RGB format
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        # Ensure uint8
        if img_array.dtype != np.uint8:
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)

        # Set image on predictor
        self._predictor.set_image(img_array)

        # Convert points to numpy arrays
        point_coords = np.array([[p.value(0), p.value(1)] for p in points], dtype=np.float32)
        point_labels_arr = np.array(point_labels, dtype=np.int32)

        # Run inference with appropriate autocast
        device = getattr(self._predictor, 'device', None)
        if device is None and hasattr(self._predictor, 'model'):
            device = next(self._predictor.model.parameters()).device

        if device is not None and str(device).startswith('cuda'):
            autocast_context = torch.autocast(str(device).split(':')[0], dtype=torch.bfloat16)
        else:
            autocast_context = contextlib.nullcontext()

        with torch.inference_mode(), autocast_context:
            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels_arr,
                multimask_output=True,
            )

        # Create detected object set
        detected_objects = DetectedObjectSet()

        # Select best mask
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        score = float(scores[best_idx])

        # Get bounding box from mask
        ys, xs = np.where(mask)
        if len(xs) > 0 and len(ys) > 0:
            x1, y1 = float(xs.min()), float(ys.min())
            x2, y2 = float(xs.max()), float(ys.max())

            bbox = BoundingBoxD(x1, y1, x2, y2)
            dot = DetectedObjectType("object", score)
            detected_obj = DetectedObject(bbox, score, dot)

            # Create mask image (crop to bounding box)
            mask_crop = mask[int(y1):int(y2)+1, int(x1):int(x2)+1].astype(np.uint8)
            detected_obj.mask = ImageContainer(Image(mask_crop))

            detected_objects.add(detected_obj)

        return detected_objects


def __vital_algorithm_register__():
    """Register the SAM2 segment via points algorithm with KWIVER."""
    register_vital_algorithm(
        SAM2Segmenter,
        "sam2",
        "SAM2-based point segmentation algorithm"
    )
