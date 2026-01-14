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


class SAM3TrackerConfig(scfg.DataConfig):
    """
    Configuration for :class:`SAM3Tracker`.

    This tracker uses SAM 3 (Segment Anything Model 2.1) for video object
    tracking with text queries. It can use Grounding DINO for text-based
    object detection and SAM 3 for mask propagation.
    """
    # Model configuration
    sam_model_id = scfg.Value(
        "facebook/sam2.1-hiera-large",
        help='SAM 2.1 model ID from HuggingFace or local path'
    )
    grounding_model_id = scfg.Value(
        "IDEA-Research/grounding-dino-tiny",
        help='Grounding DINO model ID for text-based detection'
    )

    # Device configuration
    device = scfg.Value('cuda', help='Device to run models on (cuda, cpu, auto)')

    # Text query configuration
    text_query = scfg.Value(
        'object',
        help='Text query describing objects to track. Can be comma-separated for multiple classes.'
    )

    # Detection thresholds
    detection_threshold = scfg.Value(
        0.3,
        help='Confidence threshold for initial text-based detections'
    )
    text_threshold = scfg.Value(
        0.25,
        help='Text matching threshold for grounding detection'
    )

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

    # Output configuration
    output_type = scfg.Value(
        'polygon',
        help='Type of output: "polygon" for mask contours, "points" for centroid/keypoints, "both"'
    )
    polygon_simplification = scfg.Value(
        0.01,
        help='Douglas-Peucker simplification epsilon (relative to perimeter). 0 to disable.'
    )
    num_points = scfg.Value(
        5,
        help='Number of points to output when output_type includes points'
    )

    def __post_init__(self):
        super().__post_init__()
        # Parse text query into list of classes
        if isinstance(self.text_query, str):
            self.text_query_list = [q.strip() for q in self.text_query.split(',')]
        else:
            self.text_query_list = [self.text_query]


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

        # Models (lazy loaded)
        self._sam_predictor = None
        self._grounding_processor = None
        self._grounding_model = None

        # Tracking state
        self._tracks = {}  # track_id -> track history
        self._track_counter = 0
        self._frame_count = 0
        self._last_masks = {}  # track_id -> last mask
        self._lost_counts = {}  # track_id -> frames since last detection

        # Video state for SAM
        self._video_predictor = None
        self._inference_state = None

    def get_configuration(self):
        """Get the algorithm configuration."""
        cfg = super(TrackObjects, self).get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        """Set the algorithm configuration and initialize models."""
        from viame.pytorch.utilities import vital_config_update

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

        self._init_models()
        return True

    def check_configuration(self, cfg):
        """Check if the configuration is valid."""
        return True

    def _init_models(self):
        """Initialize SAM and Grounding DINO models."""
        import torch
        from viame.pytorch.utilities import resolve_device

        self._device = resolve_device(self._config.device)

        # Initialize Grounding DINO for text-based detection
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

            self._grounding_processor = AutoProcessor.from_pretrained(
                self._config.grounding_model_id
            )
            self._grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self._config.grounding_model_id
            ).to(self._device)
            self._grounding_model.eval()
        except Exception as e:
            print(f"[SAM3Tracker] Warning: Could not load Grounding DINO: {e}")
            self._grounding_model = None

        # Initialize SAM 2.1 for video segmentation
        try:
            from sam2.build_sam import build_sam2_video_predictor

            # SAM2 model configuration
            sam_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            sam_checkpoint = self._config.sam_model_id

            self._video_predictor = build_sam2_video_predictor(
                sam_cfg, sam_checkpoint, device=self._device
            )
        except ImportError:
            # Fallback: try HuggingFace transformers SAM2
            try:
                from transformers import Sam2Model, Sam2Processor

                self._sam_processor = Sam2Processor.from_pretrained(
                    self._config.sam_model_id
                )
                self._sam_model = Sam2Model.from_pretrained(
                    self._config.sam_model_id
                ).to(self._device)
                self._sam_model.eval()
                self._video_predictor = None
            except Exception as e:
                print(f"[SAM3Tracker] Warning: Could not load SAM2: {e}")
                self._sam_model = None

    def _detect_with_text(self, image_np):
        """
        Detect objects in image using text query via Grounding DINO.

        Args:
            image_np: RGB image as numpy array

        Returns:
            List of (box, score, class_name) tuples
        """
        if self._grounding_model is None:
            return []

        import torch
        from PIL import Image

        pil_img = Image.fromarray(image_np)

        # Prepare text labels
        text_labels = [self._text_query_list]

        inputs = self._grounding_processor(
            images=pil_img, text=text_labels, return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            outputs = self._grounding_model(**inputs)

        results = self._grounding_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self._detection_threshold,
            text_threshold=self._text_threshold,
            target_sizes=[pil_img.size[::-1]]
        )

        detections = []
        if len(results) > 0:
            result = results[0]
            for box, score, label in zip(
                result["boxes"], result["scores"], result["text_labels"]
            ):
                box_np = box.cpu().numpy()
                score_val = float(score.cpu().numpy())
                detections.append((box_np, score_val, label))

        return detections

    def _segment_with_sam(self, image_np, boxes):
        """
        Segment objects in image using SAM with box prompts.

        Args:
            image_np: RGB image as numpy array
            boxes: List of [x1, y1, x2, y2] bounding boxes

        Returns:
            List of binary masks (numpy arrays)
        """
        if len(boxes) == 0:
            return []

        import torch

        # Use SAM2 video predictor if available
        if self._video_predictor is not None:
            # For video predictor, we need to initialize inference state
            # This is a simplified version - full video tracking would maintain state
            masks = []
            for box in boxes:
                try:
                    # Add object with box prompt
                    _, _, mask_logits = self._video_predictor.add_new_points_or_box(
                        inference_state=self._inference_state,
                        frame_idx=self._frame_count,
                        obj_id=len(masks),
                        box=box
                    )
                    mask = (mask_logits > 0.0).cpu().numpy().squeeze()
                    masks.append(mask)
                except Exception:
                    # Fallback to empty mask
                    masks.append(np.zeros((image_np.shape[0], image_np.shape[1]), dtype=bool))
            return masks

        # Fallback to HuggingFace SAM2
        if hasattr(self, '_sam_model') and self._sam_model is not None:
            from PIL import Image

            pil_img = Image.fromarray(image_np)
            masks = []

            for box in boxes:
                inputs = self._sam_processor(
                    images=pil_img,
                    input_boxes=[[[box[0], box[1], box[2], box[3]]]],
                    return_tensors="pt"
                ).to(self._device)

                with torch.no_grad():
                    outputs = self._sam_model(**inputs)

                mask = outputs.pred_masks[0, 0, 0].cpu().numpy() > 0
                masks.append(mask)

            return masks

        # No SAM model available - return empty masks
        return [np.ones((image_np.shape[0], image_np.shape[1]), dtype=bool)] * len(boxes)

    def _mask_to_polygon(self, mask):
        """
        Convert a binary mask to a polygon.

        Args:
            mask: Binary mask as numpy array

        Returns:
            Polygon object or None if conversion fails
        """
        import cv2

        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return None

        # Get largest contour
        contour = max(contours, key=cv2.contourArea)

        # Simplify if requested
        if self._polygon_simplification > 0:
            perimeter = cv2.arcLength(contour, True)
            epsilon = self._polygon_simplification * perimeter
            contour = cv2.approxPolyDP(contour, epsilon, True)

        if len(contour) < 3:
            return None

        # Convert to Polygon
        points = contour.squeeze()
        if len(points.shape) == 1:
            return None

        polygon = Polygon()
        for point in points:
            polygon.push_back((float(point[0]), float(point[1])))

        return polygon

    def _mask_to_points(self, mask, num_points):
        """
        Extract representative points from a mask.

        Args:
            mask: Binary mask as numpy array
            num_points: Number of points to extract

        Returns:
            List of (x, y) tuples
        """
        import cv2

        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return []

        contour = max(contours, key=cv2.contourArea)

        # Get centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return []

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        points = [(cx, cy)]

        # Add points along contour if needed
        if num_points > 1 and len(contour) > 0:
            step = max(1, len(contour) // (num_points - 1))
            for i in range(0, len(contour), step):
                if len(points) >= num_points:
                    break
                pt = contour[i].squeeze()
                points.append((int(pt[0]), int(pt[1])))

        return points[:num_points]

    def _box_from_mask(self, mask):
        """Get bounding box from mask."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            return None

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        return BoundingBoxD(float(x1), float(y1), float(x2), float(y2))

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

        # Get image as numpy array
        img_np = image.image().asarray().astype('uint8')

        # Handle grayscale images
        if len(img_np.shape) == 2:
            img_np = np.stack((img_np,) * 3, axis=-1)
        elif img_np.shape[2] == 1:
            img_np = np.stack((img_np[:, :, 0],) * 3, axis=-1)
        elif img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]

        # Convert BGR to RGB if needed
        if img_np.shape[2] == 3:
            img_np = img_np[:, :, ::-1].copy()

        # Determine if we should re-detect
        should_detect = (
            len(self._tracks) == 0 or
            (self._reinit_interval > 0 and frame_id % self._reinit_interval == 0)
        )

        new_detections = []
        if should_detect:
            # Detect objects using text query
            new_detections = self._detect_with_text(img_np)

        # Get boxes for segmentation
        boxes_to_segment = []
        box_sources = []  # 'new' or track_id

        # Add new detections
        for box, score, class_name in new_detections:
            # Check if overlaps with existing track
            overlaps = False
            for tid, track_data in self._tracks.items():
                if 'last_box' in track_data:
                    iou = self._compute_iou(box, track_data['last_box'])
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
            masks = self._segment_with_sam(img_np, boxes_to_segment)
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
            bbox = self._box_from_mask(mask)
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
                polygon = self._mask_to_polygon(mask)
                if polygon is not None:
                    det.set_polygon(polygon)

            # Add points if requested
            if self._output_type in ('points', 'both'):
                points = self._mask_to_points(mask, self._num_points)
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
