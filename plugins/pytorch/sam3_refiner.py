# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
SAM3 (Segment Anything Model 3) Track Refiner

This refiner uses SAM 2.1 for refining object tracks with text prompts.
It operates on a per-frame basis to:
- Keep existing tracks and extend with new objects from text query that don't overlap
- Re-segment masks using SAM on existing track bounding boxes
- Filter and remove low-quality tracks based on SAM mask quality
- Adjust bounding boxes based on refined masks
- Add points to detections
- Remove tracks if they no longer match the query
"""

import scriptconfig as scfg
import numpy as np

from kwiver.vital.algo import RefineTracks, RefineDetections
from kwiver.vital.types import (
    BoundingBoxD, DetectedObject, DetectedObjectSet, DetectedObjectType,
    ObjectTrackState, Track, ObjectTrackSet, ImageContainer
)
from kwiver.vital.util import VitalPIL
from PIL import Image as PILImage

from viame.pytorch.sam3_utilities import (
    SAM3BaseConfig, SAM3ModelManager,
    mask_to_polygon, mask_to_points, box_from_mask, compute_iou,
    image_to_rgb_numpy, get_autocast_context, parse_bool
)
from viame.pytorch.utilities import vital_config_update


class SAM3RefinerConfig(SAM3BaseConfig):
    """
    Configuration for SAM3Refiner.

    Extends SAM3BaseConfig with track refinement specific options.
    """
    # Track refinement parameters
    iou_threshold = scfg.Value(
        0.5,
        help='IoU threshold for matching new detections to existing tracks'
    )
    min_mask_area = scfg.Value(
        10,
        help='Minimum mask area in pixels; tracks with smaller masks are removed'
    )
    resegment_existing = scfg.Value(
        True,
        help='Whether to re-segment existing track bounding boxes with SAM'
    )
    add_new_objects = scfg.Value(
        True,
        help='Whether to add new objects detected by text query that do not overlap'
    )
    filter_by_quality = scfg.Value(
        True,
        help='If True, remove tracks with poor mask quality'
    )
    adjust_boxes = scfg.Value(
        True,
        help='Whether to adjust bounding boxes based on refined masks'
    )
    max_new_objects = scfg.Value(
        50,
        help='Maximum number of new objects to add per frame'
    )
    # Cross-frame tracking for new objects detected by text query.
    # When enabled, new objects found on one frame are matched to the same
    # object on subsequent frames by IoU, producing multi-frame tracks
    # instead of single-frame detections.
    track_new_objects = scfg.Value(
        True,
        help='Track text-query objects across frames using IoU matching'
    )
    track_match_threshold = scfg.Value(
        0.3,
        help='IoU threshold for matching a new detection to a tracked object'
    )
    lost_track_frames = scfg.Value(
        10,
        help='Frames without a detection before a tracked object is dropped'
    )


def _ensure_binary_mask(mask):
    """Ensure mask is a numpy uint8 binary array suitable for contour finding."""
    if not isinstance(mask, np.ndarray):
        import torch
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        else:
            mask = np.array(mask)
    return (mask > 0.5).astype(np.uint8)


def _set_polygon_on_detection(det, mask, simplification):
    """Set a flattened polygon on a detection from a binary mask."""
    binary_mask = _ensure_binary_mask(mask)
    poly_pts = mask_to_polygon(binary_mask, simplification)
    if poly_pts is not None:
        det.set_flattened_polygon(poly_pts)


class SAM3Refiner(RefineTracks):
    """
    SAM3-based track refiner.

    This refiner uses SAM 2.1 for per-frame track refinement operations.
    It can improve mask quality, add new objects, and filter tracks.

    Key features:
    - Re-segments existing track bounding boxes with SAM for better masks
    - Detects new objects using Grounding DINO text queries
    - Adds non-overlapping new detections as new tracks
    - Filters out tracks with low-quality masks
    - Adjusts bounding boxes to fit refined masks
    - Generates polygon and/or point outputs from masks

    Example:
        >>> from viame.pytorch.sam3_refiner import SAM3Refiner
        >>> refiner = SAM3Refiner()
        >>> refiner.set_configuration({'text_query': 'fish, crab'})
        >>> refined_tracks = refiner.refine(timestamp, image, tracks)
    """

    def __init__(self):
        RefineTracks.__init__(self)
        self._config = SAM3RefinerConfig()
        self._model_manager = SAM3ModelManager()
        self._next_track_id = 1  # For generating new track IDs

        # Cross-frame tracking state for text-query objects.
        # Maps track_id -> { 'last_box': [...], 'class_name': str,
        #                     'lost': int, 'history': [ObjectTrackState, ...] }
        self._tracked_objects = {}
        self._track_new_objects = True
        self._track_match_threshold = 0.3
        self._lost_track_frames = 10

    def get_configuration(self):
        """Get the algorithm configuration."""
        cfg = super(RefineTracks, self).get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        """Set the algorithm configuration and initialize models."""
        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)

        for key in self._config.keys():
            self._config[key] = str(cfg.get_value(key))

        self._config.__post_init__()

        # Convert types
        self._iou_threshold = float(self._config.iou_threshold)
        self._min_mask_area = int(self._config.min_mask_area)
        self._resegment_existing = parse_bool(self._config.resegment_existing)
        self._add_new_objects = parse_bool(self._config.add_new_objects)
        self._filter_by_quality = parse_bool(self._config.filter_by_quality)
        self._adjust_boxes = parse_bool(self._config.adjust_boxes)
        self._max_new_objects = int(self._config.max_new_objects)
        self._detection_threshold = float(self._config.detection_threshold)
        self._text_threshold = float(self._config.text_threshold)
        self._polygon_simplification = float(self._config.polygon_simplification)
        self._num_points = int(self._config.num_points)
        self._output_type = self._config.output_type
        self._text_query_list = self._config.text_query_list
        self._track_new_objects = parse_bool(self._config.track_new_objects)
        self._track_match_threshold = float(self._config.track_match_threshold)
        self._lost_track_frames = int(self._config.lost_track_frames)

        # Initialize models (image predictor mode, not video)
        self._model_manager.init_models(self._config, use_video_predictor=False)

        return True

    def check_configuration(self, cfg):
        """Check if the configuration is valid."""
        return True

    def refine(self, ts, image_data, tracks):
        """
        Refine tracks for the current frame.

        When ``track_new_objects`` is enabled (the default), text-query
        detections are matched across frames by IoU so that the same
        physical object keeps a single track ID, producing multi-frame
        tracks instead of isolated per-frame detections.

        Args:
            ts: Timestamp for the current frame
            image_data: Image container for the current frame
            tracks: ObjectTrackSet containing tracks to refine

        Returns:
            ObjectTrackSet: Refined tracks
        """
        if not ts.has_valid_frame():
            raise RuntimeError("Frame timestamps must contain frame IDs")

        frame_id = ts.get_frame()

        # Convert image to numpy RGB
        img_np = image_to_rgb_numpy(image_data)

        # Extract current frame's track states from the *input* track set
        track_states = {}  # track_id -> (track, state, detection)
        max_track_id = 0

        for track in tracks.tracks():
            track_id = track.id
            max_track_id = max(max_track_id, track_id)

            for state in track:
                if state.frame_id == frame_id:
                    detection = state.detection()
                    track_states[track_id] = (track, state, detection)
                    break

        # Update next track ID to be higher than any existing
        self._next_track_id = max(self._next_track_id, max_track_id + 1)
        # Also keep it above any tracked-object IDs from previous frames
        for tid in self._tracked_objects:
            self._next_track_id = max(self._next_track_id, tid + 1)

        # -----------------------------------------------------------------
        # Collect boxes for segmentation
        # -----------------------------------------------------------------
        boxes_to_segment = []
        box_sources = []  # ('existing', track_id) or ('new', score, class_name, box)

        # Re-segment existing input-track boxes
        if self._resegment_existing:
            for tid, (track, state, det) in track_states.items():
                bbox = det.bounding_box
                box = [bbox.min_x(), bbox.min_y(), bbox.max_x(), bbox.max_y()]
                boxes_to_segment.append(box)
                box_sources.append(('existing', tid))

        # Also re-segment any tracked objects from previous frames whose
        # box we are propagating (they are not in the input track_states)
        tracked_propagated = {}  # index in boxes_to_segment -> tracked tid
        if self._track_new_objects:
            for tid, tdata in self._tracked_objects.items():
                if tid not in track_states:
                    boxes_to_segment.append(list(tdata['last_box']))
                    tracked_propagated[len(boxes_to_segment) - 1] = tid
                    box_sources.append(('propagated', tid))

        # Detect new objects with text query
        new_detections = []
        if self._add_new_objects:
            new_detections = self._model_manager.detect_with_text(
                img_np,
                self._text_query_list,
                self._detection_threshold,
                self._text_threshold
            )

            # Build set of all boxes we should avoid overlapping with
            # (existing input tracks + propagated tracked objects)
            existing_boxes = [
                [det.bounding_box.min_x(), det.bounding_box.min_y(),
                 det.bounding_box.max_x(), det.bounding_box.max_y()]
                for _, (_, _, det) in track_states.items()
            ]
            for tdata in self._tracked_objects.values():
                existing_boxes.append(list(tdata['last_box']))

            for box, score, class_name in new_detections:
                overlaps = False
                for existing_box in existing_boxes:
                    if compute_iou(box, existing_box) > self._iou_threshold:
                        overlaps = True
                        break

                if not overlaps and len([s for s in box_sources if s[0] == 'new']) < self._max_new_objects:
                    boxes_to_segment.append(box)
                    box_sources.append(('new', score, class_name, box))

        # -----------------------------------------------------------------
        # Segment all boxes with SAM
        # -----------------------------------------------------------------
        if len(boxes_to_segment) > 0:
            masks = self._model_manager.segment_with_sam(img_np, boxes_to_segment)
        else:
            masks = []

        # -----------------------------------------------------------------
        # Process results and build output tracks
        # -----------------------------------------------------------------
        output_tracks = []
        processed_track_ids = set()
        # Track IDs that received a detection on this frame (for lost counting)
        seen_tracked_ids = set()

        for i, (mask, source) in enumerate(zip(masks, box_sources)):
            mask_area = np.sum(mask)

            # Filter by minimum mask area
            if self._filter_by_quality and mask_area < self._min_mask_area:
                if source[0] == 'existing':
                    processed_track_ids.add(source[1])
                continue

            if source[0] == 'existing':
                # Update existing input track
                tid = source[1]
                track, old_state, old_det = track_states[tid]
                processed_track_ids.add(tid)

                new_det = self._create_refined_detection(
                    old_det, mask, self._adjust_boxes
                )
                new_state = ObjectTrackState(ts, new_det)

                new_history = []
                for state in track:
                    if state.frame_id == frame_id:
                        new_history.append(new_state)
                    else:
                        new_history.append(state)

                new_track = Track(tid, new_history)
                output_tracks.append(new_track)

                # Register/update as tracked object so the box is
                # propagated to subsequent frames
                if self._track_new_objects:
                    bbox = new_det.bounding_box
                    new_box = [bbox.min_x(), bbox.min_y(),
                               bbox.max_x(), bbox.max_y()]
                    class_name = ''
                    try:
                        class_name = old_det.type.get_most_likely_class()
                    except Exception:
                        pass
                    if tid in self._tracked_objects:
                        self._tracked_objects[tid]['last_box'] = new_box
                        self._tracked_objects[tid]['lost'] = 0
                        self._tracked_objects[tid]['history'].append(new_state)
                    else:
                        self._tracked_objects[tid] = {
                            'class_name': class_name,
                            'last_box': new_box,
                            'lost': 0,
                            'history': [new_state],
                        }
                    seen_tracked_ids.add(tid)

            elif source[0] == 'propagated':
                # A tracked object from a previous frame re-segmented here
                tid = source[1]
                seen_tracked_ids.add(tid)

                det, bbox = self._detection_from_mask(
                    mask, boxes_to_segment[i],
                    self._tracked_objects[tid]['class_name'], 1.0
                )
                if det is None:
                    continue

                new_state = ObjectTrackState(ts, det)
                self._tracked_objects[tid]['history'].append(new_state)
                self._tracked_objects[tid]['last_box'] = [
                    bbox.min_x(), bbox.min_y(), bbox.max_x(), bbox.max_y()
                ]
                self._tracked_objects[tid]['lost'] = 0

            else:
                # New text-query detection
                score, class_name = source[1], source[2]
                det_box = source[3]

                det, bbox = self._detection_from_mask(
                    mask, det_box, class_name, score
                )
                if det is None:
                    continue

                if self._track_new_objects:
                    # Try to match against tracked objects by IoU
                    new_box = [bbox.min_x(), bbox.min_y(),
                               bbox.max_x(), bbox.max_y()]
                    matched_tid = self._match_tracked_object(
                        new_box, class_name, seen_tracked_ids
                    )

                    if matched_tid is not None:
                        # Extend existing tracked object
                        seen_tracked_ids.add(matched_tid)
                        new_state = ObjectTrackState(ts, det)
                        self._tracked_objects[matched_tid]['history'].append(new_state)
                        self._tracked_objects[matched_tid]['last_box'] = new_box
                        self._tracked_objects[matched_tid]['lost'] = 0
                    else:
                        # Start a new tracked object
                        tid = self._next_track_id
                        self._next_track_id += 1
                        new_state = ObjectTrackState(ts, det)
                        self._tracked_objects[tid] = {
                            'class_name': class_name,
                            'last_box': new_box,
                            'lost': 0,
                            'history': [new_state],
                        }
                        seen_tracked_ids.add(tid)
                else:
                    # Original behaviour: one-frame track per detection
                    new_state = ObjectTrackState(ts, det)
                    new_track = Track(self._next_track_id, [new_state])
                    self._next_track_id += 1
                    output_tracks.append(new_track)

        # -----------------------------------------------------------------
        # Update lost counts and emit tracked-object tracks
        # -----------------------------------------------------------------
        if self._track_new_objects:
            expired = []
            for tid, tdata in self._tracked_objects.items():
                if tid not in seen_tracked_ids:
                    tdata['lost'] += 1
                if tdata['lost'] > self._lost_track_frames:
                    expired.append(tid)

            # Emit tracked objects that weren't already output via the
            # input-track processing path (avoids duplicate track IDs)
            for tid, tdata in self._tracked_objects.items():
                if tid not in processed_track_ids and len(tdata['history']) > 0:
                    output_tracks.append(Track(tid, list(tdata['history'])))

            # Remove expired tracked objects *after* emitting them
            for tid in expired:
                del self._tracked_objects[tid]

        # Include existing input tracks that weren't processed
        if not self._resegment_existing:
            for tid, (track, state, det) in track_states.items():
                if tid not in processed_track_ids:
                    output_tracks.append(track)
                    processed_track_ids.add(tid)

        # Include input tracks that have no state for current frame
        for track in tracks.tracks():
            tid = track.id
            if tid not in processed_track_ids and tid not in track_states:
                output_tracks.append(track)

        return ObjectTrackSet(output_tracks)

    def _match_tracked_object(self, box, class_name, already_seen):
        """Match a detection box to an existing tracked object by IoU."""
        best_tid = None
        best_iou = self._track_match_threshold
        for tid, tdata in self._tracked_objects.items():
            if tid in already_seen:
                continue
            if tdata['class_name'] != class_name:
                continue
            iou = compute_iou(box, tdata['last_box'])
            if iou > best_iou:
                best_iou = iou
                best_tid = tid
        return best_tid

    def _detection_from_mask(self, mask, det_box, class_name, score):
        """Create a DetectedObject from a mask, returning (det, bbox) or (None, None)."""
        if not isinstance(mask, np.ndarray):
            import torch
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            else:
                mask = np.array(mask)

        if self._adjust_boxes:
            bbox = box_from_mask(mask)
            if bbox is None:
                return None, None
        else:
            bbox = BoundingBoxD(det_box[0], det_box[1], det_box[2], det_box[3])

        dot = DetectedObjectType(class_name, score)
        det = DetectedObject(bbox, score, dot)

        if self._output_type in ('polygon', 'both'):
            _set_polygon_on_detection(det, mask, self._polygon_simplification)

        if self._output_type in ('points', 'both'):
            mask_to_points(mask, self._num_points)

        return det, bbox

    def _create_refined_detection(self, old_det, mask, adjust_box):
        """
        Create a refined detection from an existing detection and new mask.

        Args:
            old_det: Original DetectedObject
            mask: New binary mask from SAM
            adjust_box: Whether to adjust bounding box to fit mask

        Returns:
            DetectedObject: Refined detection
        """
        # Get bounding box
        if adjust_box:
            bbox = box_from_mask(mask)
            if bbox is None:
                bbox = old_det.bounding_box
        else:
            bbox = old_det.bounding_box

        # Copy classification
        det_type = old_det.type
        confidence = old_det.confidence

        # Create new detection
        new_det = DetectedObject(bbox, confidence, det_type)

        # Add polygon
        if self._output_type in ('polygon', 'both'):
            _set_polygon_on_detection(new_det, mask, self._polygon_simplification)

        # Add points
        if self._output_type in ('points', 'both'):
            points = mask_to_points(mask, self._num_points)

        return new_det


class Sam3DetectionRefiner(RefineDetections):
    """
    SAM3-based Detection Refiner

    This refiner uses SAM3 to add segmentation masks to detections.
    It operates on DetectedObjectSet and adds polygon masks to each detection.

    Key features:
    - Re-segments detection bounding boxes with SAM3 for high-quality masks
    - Supports loading from local model files or HuggingFace
    - Generates polygon outputs from masks
    - Can optionally overwrite existing masks

    Example:
        >>> from viame.pytorch.sam3_refiner import Sam3DetectionRefiner
        >>> refiner = Sam3DetectionRefiner()
        >>> refiner.set_configuration({})
        >>> refined_dets = refiner.refine(image, detections)
    """

    def __init__(self):
        RefineDetections.__init__(self)

        self._kwiver_config = {
            'sam_model_id': 'facebook/sam2.1-hiera-large',
            'model_config': '',
            'device': 'cuda',
            'overwrite_existing': 'True',
            'output_type': 'polygon',
            'polygon_simplification': '0.01',
        }

        self._model_manager = SAM3ModelManager()

    def get_configuration(self):
        """Get the algorithm configuration."""
        cfg = super(RefineDetections, self).get_configuration()
        for key, value in self._kwiver_config.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        """Set the algorithm configuration and initialize models."""
        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)

        for key in self._kwiver_config.keys():
            self._kwiver_config[key] = str(cfg.get_value(key))

        # Create a minimal config object for model initialization
        class MinimalConfig:
            pass

        model_config = MinimalConfig()
        model_config.sam_model_id = self._kwiver_config['sam_model_id']
        model_config.model_config = self._kwiver_config.get('model_config', '')
        if model_config.model_config == '':
            model_config.model_config = None
        model_config.grounding_model_id = None  # Not needed for detection refinement
        model_config.device = self._kwiver_config['device']

        # Initialize SAM model only (no Grounding DINO needed)
        self._model_manager.init_models(model_config, use_video_predictor=False)

        # Parse config values
        self._overwrite_existing = parse_bool(self._kwiver_config['overwrite_existing'])
        self._output_type = self._kwiver_config['output_type']
        self._polygon_simplification = float(self._kwiver_config['polygon_simplification'])

        return True

    def check_configuration(self, cfg):
        """Check if the configuration is valid."""
        return True

    def refine(self, image_data, detections):
        """
        Refine detections by adding segmentation masks.

        Args:
            image_data: Image container
            detections: DetectedObjectSet to refine

        Returns:
            DetectedObjectSet: Refined detections with masks
        """
        import torch

        if len(detections) == 0:
            return DetectedObjectSet()

        # Convert image to numpy RGB
        img_np = image_to_rgb_numpy(image_data)

        # Collect boxes for segmentation
        boxes = []
        for det in detections:
            bbox = det.bounding_box
            boxes.append([bbox.min_x(), bbox.min_y(), bbox.max_x(), bbox.max_y()])

        # Segment all boxes with SAM
        masks = self._model_manager.segment_with_sam(img_np, boxes)

        # Create output detection set
        output = DetectedObjectSet()

        for det, mask in zip(detections, masks):
            # Add polygon if requested
            if self._output_type in ('polygon', 'both'):
                existing_poly = det.get_flattened_polygon()
                if len(existing_poly) == 0 or self._overwrite_existing:
                    _set_polygon_on_detection(det, mask, self._polygon_simplification)

            # Add mask (relative to bounding box)
            if det.mask is None or self._overwrite_existing:
                bbox = det.bounding_box
                x1, y1 = int(bbox.min_x()), int(bbox.min_y())
                x2, y2 = int(bbox.max_x()), int(bbox.max_y())

                # Ensure bounds are within mask dimensions
                h, w = mask.shape
                x1 = max(0, min(x1, w - 1))
                x2 = max(x1 + 1, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(y1 + 1, min(y2, h))

                relative_mask = mask[y1:y2, x1:x2].astype(np.uint8)
                if relative_mask.size > 0:
                    pil_img = PILImage.fromarray(relative_mask)
                    vital_img = ImageContainer(VitalPIL.from_pil(pil_img))
                    det.mask = vital_img

            output.add(det)

        return output


def __vital_algorithm_register__():
    """Register SAM3 refiner algorithms with KWIVER."""
    from kwiver.vital.algo import algorithm_factory

    # Register SAM3Refiner (RefineTracks)
    track_impl_name = "sam3"

    if not algorithm_factory.has_algorithm_impl_name(
            SAM3Refiner.static_type_name(), track_impl_name):
        algorithm_factory.add_algorithm(
            track_impl_name,
            "SAM3 (Segment Anything Model 3) based track refiner with text queries",
            SAM3Refiner
        )
        algorithm_factory.mark_algorithm_as_loaded(track_impl_name)

    # Register Sam3DetectionRefiner (RefineDetections)
    det_impl_name = "sam3"

    if not algorithm_factory.has_algorithm_impl_name(
            Sam3DetectionRefiner.static_type_name(), det_impl_name):
        algorithm_factory.add_algorithm(
            det_impl_name,
            "SAM3 (SAM 2.1) based detection refiner for adding segmentation masks",
            Sam3DetectionRefiner
        )
        algorithm_factory.mark_algorithm_as_loaded(det_impl_name)
