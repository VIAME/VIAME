# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
SAM3-based text query algorithm for object detection and track refinement.

This module provides a concrete implementation of PerformTextQuery using
SAM3 (Segment Anything Model 3) for text-guided detection and segmentation.

The model is shared with SAM3Segmenter when using the same checkpoint/device
to avoid loading duplicates into memory.
"""

import contextlib
import sys

import numpy as np
import scriptconfig as scfg

from kwiver.vital.algo import PerformTextQuery

from viame.pytorch.utilities import vital_config_update, register_vital_algorithm
from viame.pytorch.sam3_utilities import SharedSAM3ModelCache


class SAM3TextQueryConfig(scfg.DataConfig):
    """Configuration for SAM3 text query algorithm."""

    checkpoint = scfg.Value('', help='Path to model checkpoint')
    model_config = scfg.Value('', help='Path to model config JSON')
    device = scfg.Value('cuda', help='Device to run on (cuda, cpu, auto)')

    detection_threshold = scfg.Value(0.3, help='Confidence threshold for text detections')
    max_detections = scfg.Value(10, help='Maximum detections per image')
    iou_threshold = scfg.Value(0.3, help='IoU threshold for track association')


class SAM3TextQuery(PerformTextQuery):
    """
    SAM3-based text query implementation.

    Uses SAM3's text-guided detection and segmentation capabilities to find
    objects matching natural language descriptions.

    The model is shared with other SAM3 algorithms (like SAM3Segmenter)
    when configured with the same checkpoint and device.
    """

    def __init__(self):
        PerformTextQuery.__init__(self)
        self._config = SAM3TextQueryConfig()
        self._model = None
        self._predictor = None
        self._model_lock = None

    def get_configuration(self):
        cfg = super(PerformTextQuery, self).get_configuration()
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
        print(f"[SAM3TextQuery] {message}", file=sys.stderr, flush=True)

    def _init_model(self):
        """Initialize the SAM3 model using the shared cache."""
        checkpoint = self._config.checkpoint or None
        model_config = self._config.model_config or None
        device = self._config.device

        self._log("Initializing SAM3 model for text query...")
        self._log(f"  Checkpoint: {checkpoint or 'HuggingFace default'}")
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

    def _image_to_rgb_numpy(self, image_container):
        """Convert a KWIVER image container to RGB numpy array."""
        img_np = image_container.image().asarray().astype('uint8')

        if len(img_np.shape) == 2:
            img_np = np.stack((img_np,) * 3, axis=-1)
        elif img_np.shape[2] == 1:
            img_np = np.stack((img_np[:, :, 0],) * 3, axis=-1)
        elif img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]

        return img_np

    def _run_text_query(self, image_np, text, threshold, max_detections):
        """Run text-based detection on an image."""
        import torch

        # Check if model supports text query
        if hasattr(self._model, 'predict_with_text'):
            device = self._config.device
            if str(device).startswith('cuda'):
                autocast_context = torch.autocast(str(device).split(':')[0], dtype=torch.bfloat16)
            else:
                autocast_context = contextlib.nullcontext()

            with self._model_lock:
                with torch.inference_mode(), autocast_context:
                    results = self._model.predict_with_text(
                        image=image_np,
                        text=text,
                        box_threshold=threshold,
                    )
                    return self._process_text_results(results, text, max_detections)

        # Fallback: use automatic mask generation and return all masks
        self._log("Model does not support text query, using mask generation")
        return self._run_auto_mask_fallback(image_np, text, max_detections)

    def _run_auto_mask_fallback(self, image_np, text, max_detections):
        """Fallback when text query is not supported."""
        import torch

        device = self._config.device
        if str(device).startswith('cuda'):
            autocast_context = torch.autocast(str(device).split(':')[0], dtype=torch.bfloat16)
        else:
            autocast_context = contextlib.nullcontext()

        detections = []

        # Try to use mask generator if available
        if hasattr(self._model, 'mask_generator') or hasattr(self._model, 'generate_masks'):
            with self._model_lock:
                with torch.inference_mode(), autocast_context:
                    if hasattr(self._model, 'mask_generator'):
                        masks_data = self._model.mask_generator.generate(image_np)
                    else:
                        masks_data = self._model.generate_masks(image_np)

                    for mask_info in masks_data[:max_detections]:
                        if isinstance(mask_info, dict):
                            mask = mask_info.get('segmentation', mask_info.get('mask'))
                            score = mask_info.get('predicted_iou', mask_info.get('score', 0.5))
                            bbox = mask_info.get('bbox')
                        else:
                            mask = mask_info
                            score = 0.5
                            bbox = None

                        if mask is None:
                            continue

                        # Get bounds from mask
                        ys, xs = np.where(mask)
                        if len(xs) == 0 or len(ys) == 0:
                            continue

                        if bbox is not None and len(bbox) == 4:
                            x, y, w, h = bbox
                            box = [float(x), float(y), float(x + w), float(y + h)]
                        else:
                            box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

                        detections.append({
                            "box": box,
                            "mask": mask,
                            "score": float(score),
                            "label": text,
                        })

        return detections

    def _process_text_results(self, results, text, max_detections):
        """Process results from text-based prediction."""
        detections = []

        if isinstance(results, dict):
            boxes = results.get('boxes', results.get('bboxes', []))
            masks = results.get('masks', results.get('segmentations', []))
            scores = results.get('scores', results.get('confidences', []))
            labels = results.get('labels', results.get('classes', [text] * len(boxes)))
        elif isinstance(results, (list, tuple)) and len(results) >= 2:
            boxes, masks = results[0], results[1]
            scores = results[2] if len(results) > 2 else [0.5] * len(boxes)
            labels = results[3] if len(results) > 3 else [text] * len(boxes)
        else:
            return detections

        for i in range(min(len(boxes), max_detections)):
            box = boxes[i] if i < len(boxes) else None
            mask = masks[i] if i < len(masks) else None
            score = scores[i] if i < len(scores) else 0.5
            label = labels[i] if i < len(labels) else text

            if box is not None:
                if hasattr(box, 'tolist'):
                    box = box.tolist()
                box = [float(b) for b in box]

            if hasattr(score, 'item'):
                score = score.item()
            if hasattr(label, 'item'):
                label = str(label.item())
            elif not isinstance(label, str):
                label = str(label)

            detections.append({
                "box": box or [0, 0, 0, 0],
                "mask": mask.numpy() if hasattr(mask, 'numpy') else mask,
                "score": float(score),
                "label": label,
            })

        return detections

    def perform_query(self, text_query, image_containers, timestamps=None, input_track_sets=None):
        """
        Perform text-based detection/segmentation using SAM3.

        Args:
            text_query: Natural language description of objects to detect
            image_containers: Vector of images to process
            timestamps: Optional timestamps for each image
            input_track_sets: Optional existing tracks to refine

        Returns:
            Vector of ObjectTrackSet, one per input image
        """
        from kwiver.vital.types import (
            ObjectTrackSet, ObjectTrackState, Track,
            DetectedObject, DetectedObjectType,
        )
        try:
            from kwiver.vital.types import BoundingBoxD
        except ImportError:
            from kwiver.vital.types import BoundingBox as BoundingBoxD
        from kwiver.vital.types.types import ImageContainer, Image

        num_images = len(image_containers)
        if timestamps and len(timestamps) != num_images:
            raise ValueError("timestamps length must match image_containers length")
        if input_track_sets and len(input_track_sets) != num_images:
            raise ValueError("input_track_sets length must match image_containers length")

        output_track_sets = []
        next_track_id = 0

        for img_idx, image_container in enumerate(image_containers):
            timestamp = timestamps[img_idx] if timestamps else None
            frame_id = timestamp.get_frame() if timestamp else img_idx

            existing_tracks = None
            if input_track_sets and img_idx < len(input_track_sets):
                existing_tracks = input_track_sets[img_idx]

            image_np = self._image_to_rgb_numpy(image_container)

            detections = self._run_text_query(
                image_np,
                text_query,
                float(self._config.detection_threshold),
                int(self._config.max_detections),
            )

            if existing_tracks is not None:
                track_set, next_track_id = self._associate_with_tracks(
                    detections, existing_tracks, frame_id, next_track_id
                )
            else:
                track_set, next_track_id = self._create_tracks_from_detections(
                    detections, frame_id, next_track_id
                )

            output_track_sets.append(track_set)

        return output_track_sets

    def _create_tracks_from_detections(self, detections, frame_id, next_track_id):
        """Create new tracks from detection results."""
        from kwiver.vital.types import (
            ObjectTrackSet, ObjectTrackState, Track,
            DetectedObject, DetectedObjectType,
        )
        try:
            from kwiver.vital.types import BoundingBoxD
        except ImportError:
            from kwiver.vital.types import BoundingBox as BoundingBoxD
        from kwiver.vital.types.types import ImageContainer, Image

        tracks = []

        for det in detections:
            box = det.get('box', [0, 0, 0, 0])
            score = det.get('score', 0.0)
            label = det.get('label', 'object')
            mask = det.get('mask')

            bbox = BoundingBoxD(
                float(box[0]), float(box[1]),
                float(box[2]), float(box[3])
            )

            dot = DetectedObjectType(label, float(score))
            detected_obj = DetectedObject(bbox, float(score), dot)

            if mask is not None:
                x1, y1, x2, y2 = [int(v) for v in box]
                mask_crop = mask[y1:y2+1, x1:x2+1].astype(np.uint8)
                if mask_crop.size > 0:
                    detected_obj.mask = ImageContainer(Image(mask_crop))

            track_state = ObjectTrackState(frame_id, detected_obj)
            track = Track(next_track_id)
            track.append(track_state)
            tracks.append(track)

            next_track_id += 1

        return ObjectTrackSet(tracks), next_track_id

    def _associate_with_tracks(self, detections, existing_tracks, frame_id, next_track_id):
        """Associate detections with existing tracks using IoU matching."""
        from kwiver.vital.types import (
            ObjectTrackSet, ObjectTrackState, Track,
            DetectedObject, DetectedObjectType,
        )
        try:
            from kwiver.vital.types import BoundingBoxD
        except ImportError:
            from kwiver.vital.types import BoundingBox as BoundingBoxD
        from kwiver.vital.types.types import ImageContainer, Image

        iou_threshold = float(self._config.iou_threshold)
        tracks = list(existing_tracks.tracks())

        track_boxes = []
        for track in tracks:
            last_state = None
            for state in track:
                last_state = state
            if last_state:
                det_obj = last_state.detection()
                bb = det_obj.bounding_box()
                track_boxes.append([bb.min_x(), bb.min_y(), bb.max_x(), bb.max_y()])
            else:
                track_boxes.append(None)

        used_detections = set()
        updated_tracks = []

        for track_idx, track in enumerate(tracks):
            best_det_idx = None
            best_iou = iou_threshold

            if track_boxes[track_idx] is not None:
                for det_idx, det in enumerate(detections):
                    if det_idx in used_detections:
                        continue

                    det_box = det.get('box', [0, 0, 0, 0])
                    iou = self._compute_iou(track_boxes[track_idx], det_box)

                    if iou > best_iou:
                        best_iou = iou
                        best_det_idx = det_idx

            if best_det_idx is not None:
                det = detections[best_det_idx]
                used_detections.add(best_det_idx)

                box = det.get('box', [0, 0, 0, 0])
                score = det.get('score', 0.0)
                label = det.get('label', 'object')
                mask = det.get('mask')

                bbox = BoundingBoxD(
                    float(box[0]), float(box[1]),
                    float(box[2]), float(box[3])
                )
                dot = DetectedObjectType(label, float(score))
                detected_obj = DetectedObject(bbox, float(score), dot)

                if mask is not None:
                    x1, y1, x2, y2 = [int(v) for v in box]
                    mask_crop = mask[y1:y2+1, x1:x2+1].astype(np.uint8)
                    if mask_crop.size > 0:
                        detected_obj.mask = ImageContainer(Image(mask_crop))

                track_state = ObjectTrackState(frame_id, detected_obj)
                track.append(track_state)

            updated_tracks.append(track)

        for det_idx, det in enumerate(detections):
            if det_idx in used_detections:
                continue

            box = det.get('box', [0, 0, 0, 0])
            score = det.get('score', 0.0)
            label = det.get('label', 'object')
            mask = det.get('mask')

            bbox = BoundingBoxD(
                float(box[0]), float(box[1]),
                float(box[2]), float(box[3])
            )
            dot = DetectedObjectType(label, float(score))
            detected_obj = DetectedObject(bbox, float(score), dot)

            if mask is not None:
                x1, y1, x2, y2 = [int(v) for v in box]
                mask_crop = mask[y1:y2+1, x1:x2+1].astype(np.uint8)
                if mask_crop.size > 0:
                    detected_obj.mask = ImageContainer(Image(mask_crop))

            track_state = ObjectTrackState(frame_id, detected_obj)
            track = Track(next_track_id)
            track.append(track_state)
            updated_tracks.append(track)

            next_track_id += 1

        return ObjectTrackSet(updated_tracks), next_track_id

    def _compute_iou(self, box1, box2):
        """Compute IoU between two boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area


def __vital_algorithm_register__():
    """Register the SAM3 text query algorithm with KWIVER."""
    register_vital_algorithm(
        SAM3TextQuery,
        "sam3",
        "SAM3-based text query for object detection and track refinement"
    )
