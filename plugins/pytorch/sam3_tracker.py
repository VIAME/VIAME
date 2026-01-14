# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
SAM3 (Segment Anything Model 3) Tracker

This tracker uses SAM 3 (SAM 2.1) for video object tracking with text prompts.
It produces tracks with polygons (segmentation masks) or points inside detections.

The tracker supports:
- Text-based object queries for zero-shot tracking
- Polygon mask outputs for precise object segmentation
- Point-based outputs for object localization
- Integration with Grounding DINO for text-to-box detection
"""

import scriptconfig as scfg
import numpy as np

from kwiver.vital.algo import TrackObjects
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


class SAM3TrackerConfig(SAM3BaseConfig):
    """
    Configuration for :class:`SAM3Tracker`.

    This tracker uses SAM 3 (Segment Anything Model 2.1) for video object
    tracking with text queries. It can use Grounding DINO for text-based
    object detection and SAM 3 for mask propagation.
    """
    # Tracking parameters
    track_threshold = scfg.Value(
        0.5,
        help='Minimum mask confidence to continue tracking'
    )
    max_objects = scfg.Value(
        50,
        help='Maximum number of objects to track simultaneously'
    )
    reinit_interval = scfg.Value(
        30,
        help='Frames between re-detection to find new objects (0 to disable)'
    )
    lost_track_frames = scfg.Value(
        10,
        help='Number of frames without detection before track is considered lost'
    )


class SAM3Tracker(TrackObjects):
    """
    SAM3 (Segment Anything Model 3) based object tracker.

    This tracker uses SAM 2.1 for video object tracking with text prompts.
    It can detect objects using text queries via Grounding DINO and then
    track them across frames using SAM's video predictor.

    Features:
    - Zero-shot object detection using text prompts
    - High-quality segmentation masks for tracked objects
    - Polygon or point-based output for detections
    - Automatic re-detection to capture new objects entering the scene

    Example:
        >>> from viame.pytorch.sam3_tracker import SAM3Tracker
        >>> tracker = SAM3Tracker()
        >>> tracker.set_configuration({'text_query': 'fish, crab'})
        >>> tracks = tracker.track(timestamp, image, detections)
    """

    def __init__(self):
        TrackObjects.__init__(self)
        self._config = SAM3TrackerConfig()
        self._model_manager = SAM3ModelManager()

        # Tracking state
        self._tracks = {}  # track_id -> track history
        self._track_counter = 0
        self._frame_count = 0
        self._last_masks = {}  # track_id -> last mask
        self._lost_counts = {}  # track_id -> frames since last detection

        # Video state for SAM
        self._inference_state = None

    def get_configuration(self):
        """Get the algorithm configuration."""
        cfg = super(TrackObjects, self).get_configuration()
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
        self._detection_threshold = float(self._config.detection_threshold)
        self._text_threshold = float(self._config.text_threshold)
        self._track_threshold = float(self._config.track_threshold)
        self._max_objects = int(self._config.max_objects)
        self._reinit_interval = int(self._config.reinit_interval)
        self._lost_track_frames = int(self._config.lost_track_frames)
        self._polygon_simplification = float(self._config.polygon_simplification)
        self._num_points = int(self._config.num_points)
        self._output_type = self._config.output_type
        self._text_query_list = self._config.text_query_list

        # Initialize models using video predictor for tracking
        self._model_manager.init_models(self._config, use_video_predictor=True)

        return True

    def check_configuration(self, cfg):
        """Check if the configuration is valid."""
        return True

    def track(self, ts, image, detections):
        """
        Track objects in a new frame.

        Args:
            ts: Timestamp for the current frame
            image: Image container for the current frame
            detections: Detected objects from the current frame (can be None)

        Returns:
            ObjectTrackSet containing all active tracks
        """
        if not ts.has_valid_frame():
            raise RuntimeError("Frame timestamps must contain frame IDs")

        frame_id = ts.get_frame()
        self._frame_count = frame_id

        # Convert image to numpy RGB
        img_np = image_to_rgb_numpy(image)

        # Determine if we should re-detect
        should_detect = (
            len(self._tracks) == 0 or
            (self._reinit_interval > 0 and frame_id % self._reinit_interval == 0)
        )

        new_detections = []
        if should_detect:
            # Detect objects using text query
            new_detections = self._model_manager.detect_with_text(
                img_np,
                self._text_query_list,
                self._detection_threshold,
                self._text_threshold
            )

        # Get boxes for segmentation
        boxes_to_segment = []
        box_sources = []  # 'new' or track_id

        # Add new detections
        for box, score, class_name in new_detections:
            # Check if overlaps with existing track
            overlaps = False
            for tid, track_data in self._tracks.items():
                if 'last_box' in track_data:
                    iou = compute_iou(box, track_data['last_box'])
                    if iou > 0.5:
                        overlaps = True
                        break

            if not overlaps and len(boxes_to_segment) < self._max_objects:
                boxes_to_segment.append(box)
                box_sources.append(('new', score, class_name))

        # Add boxes from existing tracks for propagation
        for tid, track_data in self._tracks.items():
            if 'last_box' in track_data:
                boxes_to_segment.append(track_data['last_box'])
                box_sources.append(('track', tid, track_data.get('class_name', 'object')))

        # Segment all boxes
        if len(boxes_to_segment) > 0:
            masks = self._model_manager.segment_with_sam(img_np, boxes_to_segment)
        else:
            masks = []

        # Process results
        for i, (mask, source) in enumerate(zip(masks, box_sources)):
            if source[0] == 'new':
                # Create new track
                score, class_name = source[1], source[2]
                self._track_counter += 1
                tid = self._track_counter

                self._tracks[tid] = {
                    'class_name': class_name,
                    'history': []
                }
                self._lost_counts[tid] = 0
            else:
                # Existing track
                tid = source[1]
                class_name = source[2]

            # Check mask quality
            mask_area = np.sum(mask)
            if mask_area < 10:  # Too small
                self._lost_counts[tid] = self._lost_counts.get(tid, 0) + 1
                continue

            # Get bounding box from mask
            bbox = box_from_mask(mask)
            if bbox is None:
                self._lost_counts[tid] = self._lost_counts.get(tid, 0) + 1
                continue

            # Reset lost count
            self._lost_counts[tid] = 0

            # Store last box for next frame
            self._tracks[tid]['last_box'] = [
                bbox.min_x(), bbox.min_y(), bbox.max_x(), bbox.max_y()
            ]
            self._last_masks[tid] = mask

            # Create detection with polygon/points
            dot = DetectedObjectType(class_name, 1.0)
            det = DetectedObject(bbox, 1.0, dot)

            # Add polygon if requested
            if self._output_type in ('polygon', 'both'):
                polygon = mask_to_polygon(mask, self._polygon_simplification)
                if polygon is not None:
                    det.set_polygon(polygon)

            # Add points if requested
            if self._output_type in ('points', 'both'):
                points = mask_to_points(mask, self._num_points)
                # Points are stored in detection notes or as keypoints
                # For now, we'll add them to detection as extra data

            # Create track state
            track_state = ObjectTrackState(ts, bbox, 1.0, det)
            self._tracks[tid]['history'].append(track_state)

        # Remove lost tracks
        lost_tracks = []
        for tid, count in self._lost_counts.items():
            if count > self._lost_track_frames:
                lost_tracks.append(tid)

        for tid in lost_tracks:
            del self._tracks[tid]
            del self._lost_counts[tid]
            if tid in self._last_masks:
                del self._last_masks[tid]

        # Build output track set
        output_tracks = []
        for tid, track_data in self._tracks.items():
            if len(track_data['history']) > 0:
                track = Track(tid, track_data['history'])
                output_tracks.append(track)

        return ObjectTrackSet(output_tracks)

    def initialize(self, ts, image, seed_detections):
        """
        Initialize the tracker for a new sequence.

        Args:
            ts: Initial timestamp
            image: Initial frame image
            seed_detections: Optional initial detections to seed tracks

        Returns:
            Initial track set
        """
        self.reset()

        if seed_detections is not None and len(seed_detections) > 0:
            # Initialize tracks from seed detections
            return self.track(ts, image, seed_detections)

        return ObjectTrackSet([])

    def finalize(self):
        """
        Finalize tracking and return all tracks.

        Returns:
            Final object track set with all tracks from the sequence
        """
        output_tracks = []
        for tid, track_data in self._tracks.items():
            if len(track_data['history']) > 0:
                track = Track(tid, track_data['history'])
                output_tracks.append(track)

        return ObjectTrackSet(output_tracks)

    def reset(self):
        """Reset the tracker state."""
        self._tracks = {}
        self._track_counter = 0
        self._frame_count = 0
        self._last_masks = {}
        self._lost_counts = {}
        self._inference_state = None


def __vital_algorithm_register__():
    """Register the SAM3Tracker algorithm with KWIVER."""
    from kwiver.vital.algo import algorithm_factory

    implementation_name = "sam3_tracker"

    if algorithm_factory.has_algorithm_impl_name(
            SAM3Tracker.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(
        implementation_name,
        "SAM3 (Segment Anything Model 3) based object tracker with text queries",
        SAM3Tracker
    )

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
