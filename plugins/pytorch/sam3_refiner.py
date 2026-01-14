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

from kwiver.vital.algo import RefineTracks
from kwiver.vital.types import (
    BoundingBoxD, DetectedObject, DetectedObjectSet, DetectedObjectType,
    ObjectTrackState, Track, ObjectTrackSet, Polygon
)

from viame.pytorch.sam3_utilities import (
    SAM3BaseConfig, SAM3ModelManager,
    mask_to_polygon, mask_to_points, box_from_mask, compute_iou,
    image_to_rgb_numpy
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
        self._resegment_existing = self._config.resegment_existing in ('True', 'true', '1', True)
        self._add_new_objects = self._config.add_new_objects in ('True', 'true', '1', True)
        self._filter_by_quality = self._config.filter_by_quality in ('True', 'true', '1', True)
        self._adjust_boxes = self._config.adjust_boxes in ('True', 'true', '1', True)
        self._max_new_objects = int(self._config.max_new_objects)
        self._detection_threshold = float(self._config.detection_threshold)
        self._text_threshold = float(self._config.text_threshold)
        self._polygon_simplification = float(self._config.polygon_simplification)
        self._num_points = int(self._config.num_points)
        self._output_type = self._config.output_type
        self._text_query_list = self._config.text_query_list

        # Initialize models (image predictor mode, not video)
        self._model_manager.init_models(self._config, use_video_predictor=False)

        return True

    def check_configuration(self, cfg):
        """Check if the configuration is valid."""
        return True

    def refine(self, ts, image_data, tracks):
        """
        Refine tracks for the current frame.

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

        # Extract current frame's track states
        track_states = {}  # track_id -> (track, state, detection)
        max_track_id = 0

        for track in tracks.tracks():
            track_id = track.id()
            max_track_id = max(max_track_id, track_id)

            for state in track:
                if state.frame() == frame_id:
                    detection = state.detection()
                    track_states[track_id] = (track, state, detection)
                    break

        # Update next track ID to be higher than any existing
        self._next_track_id = max(self._next_track_id, max_track_id + 1)

        # Collect boxes for segmentation
        boxes_to_segment = []
        box_sources = []  # ('existing', track_id) or ('new', score, class_name)

        # Add existing track boxes for re-segmentation
        if self._resegment_existing:
            for tid, (track, state, det) in track_states.items():
                bbox = det.bounding_box
                box = [bbox.min_x(), bbox.min_y(), bbox.max_x(), bbox.max_y()]
                boxes_to_segment.append(box)
                box_sources.append(('existing', tid))

        # Detect new objects with text query
        new_detections = []
        if self._add_new_objects:
            new_detections = self._model_manager.detect_with_text(
                img_np,
                self._text_query_list,
                self._detection_threshold,
                self._text_threshold
            )

            # Filter new detections that don't overlap with existing
            existing_boxes = [
                [det.bounding_box.min_x(), det.bounding_box.min_y(),
                 det.bounding_box.max_x(), det.bounding_box.max_y()]
                for _, (_, _, det) in track_states.items()
            ]

            for box, score, class_name in new_detections:
                overlaps = False
                for existing_box in existing_boxes:
                    if compute_iou(box, existing_box) > self._iou_threshold:
                        overlaps = True
                        break

                if not overlaps and len([s for s in box_sources if s[0] == 'new']) < self._max_new_objects:
                    boxes_to_segment.append(box)
                    box_sources.append(('new', score, class_name))

        # Segment all boxes with SAM
        if len(boxes_to_segment) > 0:
            masks = self._model_manager.segment_with_sam(img_np, boxes_to_segment)
        else:
            masks = []

        # Process results and build output tracks
        output_tracks = []
        processed_track_ids = set()

        for i, (mask, source) in enumerate(zip(masks, box_sources)):
            mask_area = np.sum(mask)

            # Filter by minimum mask area
            if self._filter_by_quality and mask_area < self._min_mask_area:
                if source[0] == 'existing':
                    # Keep track but mark as potentially lost - don't include in output
                    tid = source[1]
                    processed_track_ids.add(tid)
                continue

            if source[0] == 'existing':
                # Update existing track
                tid = source[1]
                track, old_state, old_det = track_states[tid]
                processed_track_ids.add(tid)

                # Create refined detection
                new_det = self._create_refined_detection(
                    old_det, mask, self._adjust_boxes
                )

                # Create new track state
                new_state = ObjectTrackState(ts, new_det.bounding_box,
                                            new_det.confidence, new_det)

                # Rebuild track with new state for this frame
                new_history = []
                for state in track:
                    if state.frame() == frame_id:
                        new_history.append(new_state)
                    else:
                        new_history.append(state)

                new_track = Track(tid, new_history)
                output_tracks.append(new_track)

            else:
                # Create new track from detection
                score, class_name = source[1], source[2]

                # Get box from mask if adjusting, otherwise use detection box
                if self._adjust_boxes:
                    bbox = box_from_mask(mask)
                    if bbox is None:
                        continue
                else:
                    box = boxes_to_segment[i]
                    bbox = BoundingBoxD(box[0], box[1], box[2], box[3])

                # Create detection
                dot = DetectedObjectType(class_name, score)
                det = DetectedObject(bbox, score, dot)

                # Add polygon/points
                if self._output_type in ('polygon', 'both'):
                    polygon = mask_to_polygon(mask, self._polygon_simplification)
                    if polygon is not None:
                        det.set_polygon(polygon)

                if self._output_type in ('points', 'both'):
                    points = mask_to_points(mask, self._num_points)
                    # Store points in detection (implementation-specific)

                # Create track state and track
                new_state = ObjectTrackState(ts, bbox, score, det)
                new_track = Track(self._next_track_id, [new_state])
                self._next_track_id += 1

                output_tracks.append(new_track)

        # Include existing tracks that weren't processed (e.g., if resegment_existing is False)
        # but still have states for this frame
        if not self._resegment_existing:
            for tid, (track, state, det) in track_states.items():
                if tid not in processed_track_ids:
                    output_tracks.append(track)
                    processed_track_ids.add(tid)

        # Include tracks that have no state for current frame (preserve history)
        for track in tracks.tracks():
            tid = track.id()
            if tid not in processed_track_ids and tid not in track_states:
                output_tracks.append(track)

        return ObjectTrackSet(output_tracks)

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
            polygon = mask_to_polygon(mask, self._polygon_simplification)
            if polygon is not None:
                new_det.set_polygon(polygon)

        # Add points
        if self._output_type in ('points', 'both'):
            points = mask_to_points(mask, self._num_points)
            # Store points (implementation-specific)

        return new_det


def __vital_algorithm_register__():
    """Register the SAM3Refiner algorithm with KWIVER."""
    from kwiver.vital.algo import algorithm_factory

    implementation_name = "sam3_refiner"

    if algorithm_factory.has_algorithm_impl_name(
            SAM3Refiner.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(
        implementation_name,
        "SAM3 (Segment Anything Model 3) based track refiner with text queries",
        SAM3Refiner
    )

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
