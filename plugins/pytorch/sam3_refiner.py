# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
SAM3 (Segment Anything Model 3) Track Refiner

This refiner uses SAM 2.1 for refining object tracks with text prompts.
It uses SAM3's native video predictor for temporal tracking with memory
attention, providing high-quality mask propagation across frames.

Features:
- Re-segments existing track bounding boxes with SAM for better masks
- Detects new objects using Grounding DINO text queries
- Propagates seed boxes across frames using SAM3 video predictor
- Adds non-overlapping new detections as new tracks
- Filters out tracks with low-quality masks
- Adjusts bounding boxes to fit refined masks
- Generates polygon and/or point outputs from masks
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
    # Whether to propagate tracked objects across frames using SAM3 video
    # predictor.  Enable for track-user-selections (seed boxes forwarded
    # across frames).  Disable for text-query pipelines where grounding
    # DINO re-detects on every frame independently.
    propagate_tracked = scfg.Value(
        True,
        help='Propagate seed boxes across frames using SAM3 video predictor'
    )
    # How often (in frames) to re-run text detection to find new objects
    # entering the scene.  Set to 0 to only detect on the first frame.
    reinit_interval = scfg.Value(
        10,
        help='Frames between text re-detection for new objects (0=first only)'
    )
    # SAM3 video predictor internal detection confidence threshold.
    # Lowering this lets SAM3 detect less prominent objects.
    video_detection_threshold = scfg.Value(
        0.3,
        help='SAM3 video predictor detection score threshold (default 0.5 in SAM3)'
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
    SAM3-based track refiner with native video tracking.

    Uses SAM3's video predictor for temporal mask propagation when
    ``propagate_tracked`` is enabled.  Seed boxes (single-state input
    tracks) are added as prompts and propagated across all subsequent
    frames with SAM3's memory-attention mechanism.

    When ``propagate_tracked`` is disabled (text-query pipelines),
    each frame is processed independently with the image predictor.

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
        self._next_track_id = 100000

        # Video predictor state (used when propagate_tracked=True)
        self._video_predictor = None
        self._pil_frames = []          # accumulated PIL images for init_state
        self._frame_prompts = {}       # frame_idx -> [(obj_id, box_rel_xywh)]
        self._text_prompt_frames = {}  # frame_idx -> text_query_string
        self._obj_id_to_class = {}     # obj_id -> class_name
        self._propagated_tracks = {}   # obj_id -> [ObjectTrackState, ...]
        self._timestamps = {}          # frame_idx -> timestamp
        self._img_width = 0
        self._img_height = 0

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
        self._propagate_tracked = parse_bool(self._config.propagate_tracked)
        self._reinit_interval = int(self._config.reinit_interval)

        # When propagation is disabled (per-frame mode), load the image
        # predictor and grounding DINO now.
        # When propagation is enabled, defer to the video predictor which
        # handles text detection natively — no grounding DINO needed.
        self._video_predictor_initialized = False
        if not self._propagate_tracked:
            self._model_manager.init_models(self._config, use_video_predictor=False)

        return True

    def check_configuration(self, cfg):
        """Check if the configuration is valid."""
        return True

    # ------------------------------------------------------------------
    # Video predictor helpers
    # ------------------------------------------------------------------

    def _ensure_video_predictor(self):
        """Lazily initialize the video predictor on first use."""
        if not self._video_predictor_initialized and self._propagate_tracked:
            self._model_manager.init_models(
                self._config, use_video_predictor=True,
            )
            self._video_predictor = self._model_manager._video_predictor
            self._video_predictor_initialized = True

            # Apply detection threshold if the predictor supports it
            thresh = float(self._config.video_detection_threshold)
            if hasattr(self._video_predictor, 'score_threshold_detection'):
                self._video_predictor.score_threshold_detection = thresh

    def _run_video_propagation(self):
        """
        Run SAM3 video predictor on accumulated frames and prompts.
        Builds the inference state from collected PIL images, adds all
        box prompts, and propagates.  Returns a dict of
        (obj_id, frame_idx) -> (binary_mask, box_xyxy).
        """
        import torch

        if not self._pil_frames:
            return {}
        if not self._frame_prompts and not self._text_prompt_frames:
            return {}

        self._ensure_video_predictor()
        if self._video_predictor is None:
            return {}

        torch.cuda.empty_cache()

        # Build inference state from PIL image list
        state = self._video_predictor.init_state(
            self._pil_frames, offload_video_to_cpu=True,
        )

        # Add all collected box prompts (from input seed tracks)
        for frame_idx, prompts in self._frame_prompts.items():
            for obj_id, box_rel_xywh in prompts:
                self._video_predictor.add_prompt(
                    state,
                    frame_idx=frame_idx,
                    boxes_xywh=[box_rel_xywh],
                    box_labels=[1],
                    obj_id=obj_id,
                )

        # Add text prompts for object detection on marked frames
        for frame_idx, text_query in self._text_prompt_frames.items():
            self._video_predictor.add_prompt(
                state,
                frame_idx=frame_idx,
                text_str=text_query,
            )

        # Suppress tqdm progress bars from SAM3's propagation
        import tqdm
        _orig_init = tqdm.tqdm.__init__
        def _quiet_init(self, *args, **kwargs):
            kwargs['disable'] = True
            _orig_init(self, *args, **kwargs)
        tqdm.tqdm.__init__ = _quiet_init

        try:
            # Forward propagation
            results = {}
            for frame_idx, frame_results in self._video_predictor.propagate_in_video(state):
                self._collect_frame_results(results, frame_idx, frame_results)

            # Reverse propagation to fill in frames before mid-video detections.
            # Only go back as far as the reinit interval to limit the buffer.
            max_reverse = self._reinit_interval if self._reinit_interval > 0 else None
            for frame_idx, frame_results in self._video_predictor.propagate_in_video(
                state, reverse=True, max_frame_num_to_track=max_reverse,
            ):
                self._collect_frame_results(results, frame_idx, frame_results,
                                            overwrite=False)
        finally:
            tqdm.tqdm.__init__ = _orig_init

        # Free video state
        try:
            self._video_predictor.reset_state(state)
        except Exception:
            pass

        return results

    def _collect_frame_results(self, results, frame_idx, frame_results,
                               overwrite=True):
        """Extract per-object masks from a propagation frame result."""
        obj_ids = np.array(frame_results['out_obj_ids'])
        boxes_xywh = np.array(frame_results['out_boxes_xywh'])
        masks = np.array(frame_results['out_binary_masks'])

        for i, oid in enumerate(obj_ids):
            oid = int(oid)
            key = (oid, frame_idx)
            if not overwrite and key in results:
                continue
            bx = boxes_xywh[i]
            ax1 = bx[0] * self._img_width
            ay1 = bx[1] * self._img_height
            ax2 = (bx[0] + bx[2]) * self._img_width
            ay2 = (bx[1] + bx[3]) * self._img_height
            results[key] = (masks[i], [ax1, ay1, ax2, ay2])

    # ------------------------------------------------------------------
    # Main refine method
    # ------------------------------------------------------------------

    def refine(self, ts, image_data, tracks):
        """
        Refine tracks for the current frame.

        When ``propagate_tracked`` is enabled, seed boxes are tracked
        across frames using SAM3's native video predictor with memory
        attention.  When disabled, each frame is processed independently.
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
        img_np = image_to_rgb_numpy(image_data)

        # Extract current frame's track states from input
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

        self._next_track_id = max(self._next_track_id, max_track_id + 1)

        if self._propagate_tracked:
            return self._refine_with_video_predictor(
                ts, frame_id, img_np, tracks, track_states
            )
        else:
            return self._refine_per_frame(
                ts, frame_id, img_np, tracks, track_states
            )

    # ------------------------------------------------------------------
    # Video-predictor path (propagate_tracked=True)
    # ------------------------------------------------------------------

    def _refine_with_video_predictor(self, ts, frame_id, img_np,
                                     tracks, track_states):
        """
        Refine using SAM3 video predictor for native temporal tracking.

        Accumulates frames and seed box prompts.  On each call, runs
        SAM3 video propagation over the full buffer to produce masks
        for all tracked objects on all frames seen so far.  Only the
        current frame's results are used for track output; previous
        frames' results were already emitted.
        """
        # Convert to PIL and store
        pil = PILImage.fromarray(img_np)
        self._pil_frames.append(pil)
        if self._img_width == 0:
            self._img_width, self._img_height = pil.size

        local_frame_idx = len(self._pil_frames) - 1
        self._timestamps[local_frame_idx] = ts

        # Collect seed box prompts from input tracks on this frame
        for tid, (track, state, det) in track_states.items():
            bbox = det.bounding_box
            x1, y1 = bbox.min_x(), bbox.min_y()
            x2, y2 = bbox.max_x(), bbox.max_y()
            w, h = self._img_width, self._img_height
            box_rel = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]
            self._frame_prompts.setdefault(local_frame_idx, []).append(
                (tid, box_rel)
            )
            class_name = ''
            try:
                class_name = det.type.get_most_likely_class()
            except Exception:
                pass
            self._obj_id_to_class[tid] = class_name

        # Mark this frame for text-query detection during propagation.
        # SAM3's video predictor handles text detection natively via
        # text_str prompts — no grounding DINO needed.  We add text
        # prompts on the first frame and periodically thereafter to
        # catch objects entering the scene mid-video.
        if self._add_new_objects and self._text_query_list:
            text_query = ', '.join(self._text_query_list)
            is_first = (local_frame_idx == 0)
            is_reinit = (self._reinit_interval > 0
                         and local_frame_idx > 0
                         and local_frame_idx % self._reinit_interval == 0)
            if is_first or is_reinit:
                self._text_prompt_frames.setdefault(
                    local_frame_idx, text_query)

        # Accumulate only — propagation runs once in finalize() after
        # all frames have been collected.
        return ObjectTrackSet([])

    def finalize(self):
        """
        Called by the pipeline after all frames have been processed.
        Runs SAM3 video propagation over the full accumulated buffer
        and returns the complete set of tracked objects.
        """
        if not self._propagate_tracked:
            return ObjectTrackSet([])

        self._run_finalize_propagation()

        output_tracks = []
        for obj_id, history in self._propagated_tracks.items():
            if len(history) > 0:
                output_tracks.append(Track(obj_id, list(history)))

        return ObjectTrackSet(output_tracks)

    def _run_finalize_propagation(self):
        """
        Run SAM3 video propagation on the full accumulated buffer.
        Called once after all frames have been collected.
        """
        if not self._pil_frames:
            return
        if not self._frame_prompts and not self._text_prompt_frames:
            return

        all_results = self._run_video_propagation()

        # Collect the set of object IDs we explicitly prompted
        prompted_ids = set()
        for prompts in self._frame_prompts.values():
            for obj_id, _ in prompts:
                prompted_ids.add(obj_id)

        self._propagated_tracks.clear()
        for (oid, fidx), (mask, box_xyxy) in all_results.items():
            # When add_new_objects is disabled (track selections), only
            # keep tracks the user explicitly seeded — drop SAM3
            # auto-discovered objects.
            if not self._add_new_objects and oid not in prompted_ids:
                continue

            mask_area = np.sum(mask)
            if self._filter_by_quality and mask_area < self._min_mask_area:
                continue

            class_name = self._obj_id_to_class.get(oid, '')
            if not class_name and self._text_query_list:
                class_name = self._text_query_list[0]
            if not class_name:
                class_name = 'unknown'
            bbox = box_from_mask(mask)
            if bbox is None:
                bbox = BoundingBoxD(box_xyxy[0], box_xyxy[1],
                                    box_xyxy[2], box_xyxy[3])

            confidence = 1.0
            dot = DetectedObjectType(class_name, confidence)
            det = DetectedObject(bbox, confidence, dot)

            if self._output_type in ('polygon', 'both'):
                _set_polygon_on_detection(det, mask, self._polygon_simplification)

            if fidx not in self._timestamps:
                continue
            frame_ts = self._timestamps[fidx]

            new_state = ObjectTrackState(frame_ts, det)
            self._propagated_tracks.setdefault(oid, []).append(new_state)

        # Include input seed detections that SAM3 may not have yielded
        for local_idx, prompts in self._frame_prompts.items():
            if local_idx not in self._timestamps:
                continue
            frame_ts = self._timestamps[local_idx]
            for tid, box_rel in prompts:
                if (tid, local_idx) not in all_results:
                    w, h = self._img_width, self._img_height
                    ax1 = box_rel[0] * w
                    ay1 = box_rel[1] * h
                    ax2 = (box_rel[0] + box_rel[2]) * w
                    ay2 = (box_rel[1] + box_rel[3]) * h
                    bbox = BoundingBoxD(ax1, ay1, ax2, ay2)
                    class_name = self._obj_id_to_class.get(tid, 'unknown')
                    dot = DetectedObjectType(class_name, 1.0)
                    det = DetectedObject(bbox, 1.0, dot)
                    state = ObjectTrackState(frame_ts, det)
                    self._propagated_tracks.setdefault(tid, []).append(state)

        for oid in self._propagated_tracks:
            self._propagated_tracks[oid].sort(key=lambda s: s.frame_id)

    # ------------------------------------------------------------------
    # Per-frame path (propagate_tracked=False, for text queries)
    # ------------------------------------------------------------------

    def _refine_per_frame(self, ts, frame_id, img_np, tracks, track_states):
        """
        Refine each frame independently using the image predictor.
        Used for text-query pipelines where grounding DINO detects on
        every frame and no cross-frame propagation is needed.
        """
        boxes_to_segment = []
        box_sources = []  # ('existing', tid) or ('new', score, class_name)

        # Re-segment existing input-track boxes
        if self._resegment_existing:
            for tid, (track, state, det) in track_states.items():
                bbox = det.bounding_box
                box = [bbox.min_x(), bbox.min_y(), bbox.max_x(), bbox.max_y()]
                boxes_to_segment.append(box)
                box_sources.append(('existing', tid))

        # Detect new objects with text query
        if self._add_new_objects:
            new_detections = self._model_manager.detect_with_text(
                img_np, self._text_query_list,
                self._detection_threshold, self._text_threshold,
            )
            suppress_boxes = [
                [det.bounding_box.min_x(), det.bounding_box.min_y(),
                 det.bounding_box.max_x(), det.bounding_box.max_y()]
                for _, (_, _, det) in track_states.items()
            ]
            for box, score, class_name in new_detections:
                overlaps = False
                for sb in suppress_boxes:
                    if compute_iou(box, sb) > self._iou_threshold:
                        overlaps = True
                        break
                new_count = len([s for s in box_sources if s[0] == 'new'])
                if not overlaps and new_count < self._max_new_objects:
                    boxes_to_segment.append(box)
                    box_sources.append(('new', score, class_name))

        # Segment all boxes with SAM image predictor
        if len(boxes_to_segment) > 0:
            masks = self._model_manager.segment_with_sam(img_np, boxes_to_segment)
        else:
            masks = []

        # Build output tracks
        output_tracks = []
        processed_track_ids = set()

        for i, (mask, source) in enumerate(zip(masks, box_sources)):
            mask_area = np.sum(mask)
            if self._filter_by_quality and mask_area < self._min_mask_area:
                if source[0] == 'existing':
                    processed_track_ids.add(source[1])
                continue

            if source[0] == 'existing':
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

                output_tracks.append(Track(tid, new_history))
            else:
                score, class_name = source[1], source[2]
                det = self._detection_from_mask(
                    mask, boxes_to_segment[i], class_name, score
                )
                if det is not None:
                    new_state = ObjectTrackState(ts, det)
                    tid = self._next_track_id
                    self._next_track_id += 1
                    output_tracks.append(Track(tid, [new_state]))

        # Pass through unprocessed input tracks
        if not self._resegment_existing:
            for tid, (track, state, det) in track_states.items():
                if tid not in processed_track_ids:
                    output_tracks.append(track)
                    processed_track_ids.add(tid)

        for track in tracks.tracks():
            tid = track.id
            if tid not in processed_track_ids and tid not in track_states:
                output_tracks.append(track)

        return ObjectTrackSet(output_tracks)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detection_from_mask(self, mask, det_box, class_name, score):
        """Create a DetectedObject from a mask, or None if invalid."""
        if not isinstance(mask, np.ndarray):
            import torch
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            else:
                mask = np.array(mask)

        if self._adjust_boxes:
            bbox = box_from_mask(mask)
            if bbox is None:
                return None
        else:
            bbox = BoundingBoxD(det_box[0], det_box[1], det_box[2], det_box[3])

        dot = DetectedObjectType(class_name, score)
        det = DetectedObject(bbox, score, dot)

        if self._output_type in ('polygon', 'both'):
            _set_polygon_on_detection(det, mask, self._polygon_simplification)

        return det

    def _create_refined_detection(self, old_det, mask, adjust_box):
        """Create a refined detection from an existing detection and new mask."""
        if adjust_box:
            bbox = box_from_mask(mask)
            if bbox is None:
                bbox = old_det.bounding_box
        else:
            bbox = old_det.bounding_box

        det_type = old_det.type
        confidence = old_det.confidence
        new_det = DetectedObject(bbox, confidence, det_type)

        if self._output_type in ('polygon', 'both'):
            _set_polygon_on_detection(new_det, mask, self._polygon_simplification)

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
