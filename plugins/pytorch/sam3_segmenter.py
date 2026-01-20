# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
SAM3-based point segmentation algorithm.

This module provides a concrete implementation of SegmentViaPoints using
SAM3 (Segment Anything Model 3) for point-based segmentation.

The model is shared with SAM3TextQuery when using the same checkpoint/device
to avoid loading duplicates into memory.
"""

import contextlib
import sys

import numpy as np
import scriptconfig as scfg

from kwiver.vital.algo import SegmentViaPoints

from viame.pytorch.utilities import vital_config_update, register_vital_algorithm
from viame.pytorch.sam3_utilities import SharedSAM3ModelCache


class SAM3SegmenterConfig(scfg.DataConfig):
    """Configuration for SAM3 segment via points algorithm."""

    checkpoint = scfg.Value('', help='Path to model checkpoint')
    model_config = scfg.Value('', help='Path to model config JSON')
    device = scfg.Value('cuda', help='Device to run on (cuda, cpu, auto)')


class SAM3Segmenter(SegmentViaPoints):
    """
    SAM3-based point segmentation implementation.

    Uses SAM3's point-based prompting to segment objects in images.
    The model is shared with other SAM3 algorithms (like SAM3TextQuery)
    when configured with the same checkpoint and device.
    """

    def __init__(self):
        SegmentViaPoints.__init__(self)
        self._config = SAM3SegmenterConfig()
        self._predictor = None
        self._model = None
        self._model_lock = None

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
        print(f"[SAM3Segmenter] {message}", file=sys.stderr, flush=True)

    def _init_model(self):
        """Initialize the SAM3 model using the shared cache."""
        checkpoint = self._config.checkpoint or None
        model_config = self._config.model_config or None
        device = self._config.device

        self._log("Initializing SAM3 model...")
        self._log(f"  Checkpoint: {checkpoint or 'HuggingFace default'}")
        self._log(f"  Model config: {model_config or 'auto'}")
        self._log(f"  Device: {device}")

        # Get or create shared model
        self._model, self._predictor = SharedSAM3ModelCache.get_or_create(
            checkpoint=checkpoint,
            model_config=model_config,
            device=device,
            logger=self._log,
        )

        # Get the lock for this model configuration
        self._model_lock = SharedSAM3ModelCache.get_lock(
            checkpoint=checkpoint,
            model_config=model_config,
            device=device,
        )

        self._log("Model initialized successfully")

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

        # Convert points to numpy arrays
        point_coords = np.array([[p.value(0), p.value(1)] for p in points], dtype=np.float32)
        point_labels_arr = np.array(point_labels, dtype=np.int32)

        # Get device for autocast
        device = getattr(self._predictor, 'device', None)
        if device is None and hasattr(self._predictor, 'model'):
            device = next(self._predictor.model.parameters()).device

        if device is not None and str(device).startswith('cuda'):
            autocast_context = torch.autocast(str(device).split(':')[0], dtype=torch.bfloat16)
        else:
            autocast_context = contextlib.nullcontext()

        # Use lock to ensure thread safety when using shared predictor
        with self._model_lock:
            # Set image on predictor
            self._predictor.set_image(img_array)

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
    """Register the SAM3 segment via points algorithm with KWIVER."""
    register_vital_algorithm(
        SAM3Segmenter,
        "sam3",
        "SAM3-based point segmentation algorithm"
    )
