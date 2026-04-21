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
    # Threshold for a detection to be promoted to a new tracked object.
    # SAM3 default is 0.7 which is very aggressive — lower for recall.
    video_new_det_threshold = scfg.Value(
        0.1,
        help='Min score for a new detection to become a tracked object (default 0.7 in SAM3)'
    )
    # Hotstart delay: SAM3 holds outputs this many frames for filtering.
    # During hotstart, unmatched/duplicate tracks are pruned.  Set to 0
    # to disable hotstart filtering entirely.
    video_hotstart_delay = scfg.Value(
        0,
        help='Frames to hold outputs for hotstart filtering (default 15 in SAM3, 0=disable)'
    )
    # Maximum number of frames to process per video chunk.  SAM3 video
    # predictor keeps per-frame features in GPU memory; processing very
    # long or high-resolution videos in one shot can cause OOM.  The
    # video is split into overlapping chunks of this size and the chunks
    # are processed sequentially.  Set to 0 for no chunking.
    video_chunk_size = scfg.Value(
        100,
        help='Max frames per video propagation chunk (0=no chunking)'
    )
    # Starting value for IDs of tracks the refiner creates (for detections
    # that don't match any input seed track when add_new_objects=True).
    # The refiner skips any IDs already used by input tracks, and remaps
    # later input tracks whose IDs would collide with already-assigned
    # refiner IDs, so the final output is always collision-free regardless
    # of this starting value.
    new_track_id_start = scfg.Value(
        1,
        help='Starting ID for refiner-created tracks (default 1)'
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


def _mask_bbox(mask):
    """Get [x1, y1, x2, y2] bounding box from a binary mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows):
        return [0, 0, 0, 0]
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [float(x1), float(y1), float(x2 + 1), float(y2 + 1)]


def _set_mask_on_detection(det, mask, bbox):
    """Set a binary mask (cropped to the detection bbox) on a detection."""
    binary_mask = _ensure_binary_mask(mask)
    x1 = max(0, int(bbox.min_x()))
    y1 = max(0, int(bbox.min_y()))
    x2 = min(binary_mask.shape[1], int(bbox.max_x()))
    y2 = min(binary_mask.shape[0], int(bbox.max_y()))
    if x2 <= x1 or y2 <= y1:
        return
    cropped = binary_mask[y1:y2, x1:x2]
    if cropped.size == 0:
        return
    pil_img = PILImage.fromarray(cropped)
    vital_img = ImageContainer(VitalPIL.from_pil(pil_img))
    det.mask = vital_img


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
        # Track-ID allocation state. ``_next_track_id`` is the next
        # candidate for a refiner-created track; ``_allocate_next_id``
        # skips over IDs that have already been used by either another
        # refiner-created track or an input seed track, so the output is
        # always collision-free. Input IDs that collide with IDs already
        # handed out to refiner-created tracks are remapped once via
        # ``_input_id_remap`` and reused on subsequent frames.
        self._next_track_id = 1
        self._assigned_ids = set()
        self._known_input_ids = set()
        self._input_id_remap = {}

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
        self._next_track_id = int(self._config.new_track_id_start)
        self._assigned_ids.clear()
        self._known_input_ids.clear()
        self._input_id_remap.clear()

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

            # Override SAM3 video predictor thresholds.
            # The built-in defaults (score_threshold_detection=0.5,
            # new_det_thresh=0.7, hotstart_delay=15) are tuned for
            # general video and are too aggressive for many domains.
            vp = self._video_predictor
            det_thresh = float(self._config.video_detection_threshold)
            new_det_thresh = float(self._config.video_new_det_threshold)
            hotstart_delay = int(self._config.video_hotstart_delay)

            if hasattr(vp, 'score_threshold_detection'):
                vp.score_threshold_detection = det_thresh
            if hasattr(vp, 'new_det_thresh'):
                vp.new_det_thresh = new_det_thresh
            if hasattr(vp, 'hotstart_delay'):
                vp.hotstart_delay = hotstart_delay
            # Enable detection NMS to suppress overlapping detections.
            # The base class default is 0.0 (disabled) but SAM3's own
            # model_builder sets 0.1 when constructing the video model.
            if hasattr(vp, 'det_nms_thresh') and vp.det_nms_thresh <= 0:
                vp.det_nms_thresh = 0.1

    def _run_video_propagation(self):
        """
        Run SAM3 video predictor on accumulated frames and prompts.
        Splits long videos into chunks to avoid GPU OOM.
        Returns a dict of (obj_id, frame_idx) -> (binary_mask, box_xyxy).
        """
        import torch
        import sys

        if not self._pil_frames:
            return {}
        if not self._frame_prompts and not self._text_prompt_frames:
            return {}

        self._ensure_video_predictor()
        if self._video_predictor is None:
            return {}

        chunk_size = int(self._config.video_chunk_size) if hasattr(self._config, 'video_chunk_size') else 100
        total_frames = len(self._pil_frames)

        if chunk_size <= 0 or total_frames <= chunk_size:
            # Process all at once
            return self._run_video_propagation_chunk(
                self._pil_frames, 0, self._frame_prompts,
                self._text_prompt_frames)

        # Process in overlapping chunks with ID reconciliation.
        # Each chunk assigns its own object IDs independently.  We use
        # mask IoU in the overlap region to match new-chunk IDs to the
        # IDs already established in all_results, so that the same
        # physical object keeps the same track ID across chunks.
        print(f"[SAM3] Processing {total_frames} frames in chunks of {chunk_size}")
        all_results = {}
        next_global_id = 0  # monotonically increasing global ID counter
        overlap = max(10, chunk_size // 10)
        start = 0

        while start < total_frames:
            end = min(start + chunk_size, total_frames)
            chunk_frames = self._pil_frames[start:end]

            # Remap prompts to chunk-local indices
            chunk_box_prompts = {}
            for fidx, prompts in self._frame_prompts.items():
                if start <= fidx < end:
                    chunk_box_prompts[fidx - start] = prompts

            chunk_text_prompts = {}
            for fidx, text in self._text_prompt_frames.items():
                if start <= fidx < end:
                    chunk_text_prompts[fidx - start] = text

            # If no text prompt falls in this chunk, add one on the
            # first frame of the chunk so detection stays active.
            if not chunk_text_prompts and self._text_prompt_frames:
                first_text = next(iter(self._text_prompt_frames.values()))
                chunk_text_prompts[0] = first_text

            chunk_results = self._run_video_propagation_chunk(
                chunk_frames, start, chunk_box_prompts, chunk_text_prompts)

            if not all_results:
                # First chunk — adopt IDs directly, offset to global range
                chunk_oids = set(oid for oid, _ in chunk_results.keys())
                id_map = {}
                for cid in sorted(chunk_oids):
                    id_map[cid] = next_global_id
                    next_global_id += 1
                for (cid, local_fidx), val in chunk_results.items():
                    gid = id_map[cid]
                    all_results[(gid, local_fidx + start)] = val
            else:
                # Build ID mapping by matching masks in the overlap region.
                id_map = self._match_chunk_ids(
                    all_results, chunk_results, start, overlap,
                    self._iou_threshold)

                # Assign new global IDs for unmatched chunk objects
                for cid in set(oid for oid, _ in chunk_results.keys()):
                    if cid not in id_map:
                        id_map[cid] = next_global_id
                        next_global_id += 1

                # Merge, skipping overlap frames that already have data
                overlap_end = start + overlap
                for (cid, local_fidx), val in chunk_results.items():
                    global_fidx = local_fidx + start
                    gid = id_map[cid]
                    key = (gid, global_fidx)
                    # In overlap region, keep prior chunk's results
                    if global_fidx < overlap_end and key in all_results:
                        continue
                    all_results[key] = val

            start += chunk_size - overlap

        return all_results

    @staticmethod
    def _match_chunk_ids(all_results, chunk_results, chunk_start,
                         overlap, iou_thresh):
        """
        Match object IDs from a new chunk to existing global IDs using
        mask IoU in the overlap region.

        Returns a dict mapping chunk_obj_id -> global_obj_id for matched
        objects.  Unmatched chunk objects are not included.
        """
        # Collect masks per object in the overlap frames from both sides
        overlap_end = chunk_start + overlap

        # Existing global results in the overlap region: gid -> list of masks
        global_masks = {}
        for (gid, gfidx), rtup in all_results.items():
            if chunk_start <= gfidx < overlap_end:
                global_masks.setdefault(gid, {})[gfidx] = rtup[0]

        # New chunk results in the overlap region: cid -> list of masks
        chunk_masks = {}
        for (cid, local_fidx), rtup in chunk_results.items():
            gfidx = local_fidx + chunk_start
            if chunk_start <= gfidx < overlap_end:
                chunk_masks.setdefault(cid, {})[gfidx] = rtup[0]

        if not global_masks or not chunk_masks:
            return {}

        # Compute average IoU between each (chunk_id, global_id) pair
        # across shared overlap frames
        id_map = {}
        used_gids = set()

        # Score all pairs
        pairs = []
        for cid, c_frames in chunk_masks.items():
            for gid, g_frames in global_masks.items():
                shared = set(c_frames.keys()) & set(g_frames.keys())
                if not shared:
                    continue
                total_iou = 0.0
                for fidx in shared:
                    total_iou += compute_iou(
                        _mask_bbox(c_frames[fidx]),
                        _mask_bbox(g_frames[fidx]))
                avg_iou = total_iou / len(shared)
                if avg_iou > iou_thresh:
                    pairs.append((avg_iou, cid, gid))

        # Greedy matching: best IoU first, each ID used at most once
        pairs.sort(reverse=True)
        used_cids = set()
        for iou_val, cid, gid in pairs:
            if cid in used_cids or gid in used_gids:
                continue
            id_map[cid] = gid
            used_cids.add(cid)
            used_gids.add(gid)

        return id_map

    def _run_video_propagation_chunk(self, pil_frames, global_offset,
                                     frame_prompts, text_prompt_frames):
        """
        Run SAM3 video propagation on a single chunk of frames.
        frame_prompts and text_prompt_frames use chunk-local indices.
        Returns results keyed by (obj_id, local_frame_idx).
        """
        import torch
        import contextlib

        torch.cuda.empty_cache()

        # Build inference context: inference_mode + fp16 autocast + SDPA
        # fallback for pre-Ampere GPUs (e.g. Turing RTX 5000).  The SAM 3.1
        # adapter already wraps calls this way, but the raw SAM 3.0 video
        # predictor does not.  Running in fp32 without this causes OOM on
        # 16 GB GPUs even with small frame counts.
        cm = contextlib.ExitStack()
        cm.enter_context(torch.inference_mode())
        try:
            cm.enter_context(get_autocast_context(
                str(self._model_manager.device)))
        except Exception:
            pass
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            cm.enter_context(sdpa_kernel(
                [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
            ))
        except Exception:
            pass

        with cm:
            state = self._video_predictor.init_state(
                pil_frames, offload_video_to_cpu=True,
            )

            from viame.pytorch.sam3_utilities import _Sam3p1VideoPredictorAdapter
            is_sam31 = isinstance(self._video_predictor,
                                  _Sam3p1VideoPredictorAdapter)

            if is_sam31:
                # SAM 3.1 multiplex ``add_prompt`` unconditionally calls
                # ``reset_state`` on every invocation (see
                # ``sam3_multiplex_tracking.py``), so any earlier box/text
                # prompt is wiped by the next call. Collapse everything into
                # a single add_prompt on one seed frame: text_str + all
                # accumulated seed boxes (across all seed frames) fused onto
                # the earliest seeded frame. The tracker's text-driven
                # detector continues running on later frames during
                # propagation, so we don't lose the per-frame detection
                # signal.
                text_query = None
                for _, tq in text_prompt_frames.items():
                    text_query = tq
                    break

                if frame_prompts:
                    seed_frame = min(frame_prompts.keys())
                    seed_boxes = []
                    for fidx in sorted(frame_prompts.keys()):
                        for _obj_id, box_rel_xywh in frame_prompts[fidx]:
                            seed_boxes.append(box_rel_xywh)
                    add_kwargs = dict(
                        frame_idx=seed_frame,
                        boxes_xywh=seed_boxes,
                        box_labels=[1] * len(seed_boxes),
                    )
                    if text_query is not None:
                        add_kwargs['text_str'] = text_query
                    self._video_predictor.add_prompt(state, **add_kwargs)
                elif text_query is not None:
                    # Text-only seed
                    seed_frame = 0
                    for fidx in text_prompt_frames.keys():
                        seed_frame = fidx
                        break
                    self._video_predictor.add_prompt(
                        state,
                        frame_idx=seed_frame,
                        text_str=text_query,
                    )
            else:
                # SAM 3.0 video predictor: add_prompt does not reset state,
                # so we can accumulate per-frame seed boxes and add the
                # global text prompt once at the end.
                for frame_idx, prompts in frame_prompts.items():
                    for obj_id, box_rel_xywh in prompts:
                        self._video_predictor.add_prompt(
                            state,
                            frame_idx=frame_idx,
                            boxes_xywh=[box_rel_xywh],
                            box_labels=[1],
                            obj_id=obj_id,
                        )

                for frame_idx, text_query in text_prompt_frames.items():
                    self._video_predictor.add_prompt(
                        state,
                        frame_idx=frame_idx,
                        text_str=text_query,
                    )
                    break

            # Suppress tqdm progress bars from SAM3's propagation
            import tqdm
            _orig_init = tqdm.tqdm.__init__
            def _quiet_init(self_tqdm, *args, **kwargs):
                kwargs['disable'] = True
                _orig_init(self_tqdm, *args, **kwargs)
            tqdm.tqdm.__init__ = _quiet_init

            try:
                results = {}
                for frame_idx, frame_results in self._video_predictor.propagate_in_video(state):
                    self._collect_frame_results(results, frame_idx, frame_results)
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
        probs = np.array(frame_results.get('out_probs', []))

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
            score = float(probs[i]) if i < len(probs) else 1.0
            results[key] = (masks[i], [ax1, ay1, ax2, ay2], score)

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

        # Resolve each input track's ID to the ID we'll actually use. New
        # input IDs get registered; any that collide with an ID already
        # handed out to a refiner-created track get remapped to the next
        # unused ID. Subsequent frames reuse the mapping.
        for track in tracks.tracks():
            self._get_or_remap_input_id(track.id)

        # Extract current frame's track states, keyed by the (possibly
        # remapped) resolved ID so downstream output uses that ID too.
        track_states = {}  # resolved_id -> (track, state, detection)
        for track in tracks.tracks():
            resolved_id = self._input_id_remap[track.id]
            for state in track:
                if state.frame_id == frame_id:
                    detection = state.detection()
                    track_states[resolved_id] = (track, state, detection)
                    break

        if self._propagate_tracked:
            return self._refine_with_video_predictor(
                ts, frame_id, img_np, tracks, track_states
            )
        else:
            return self._refine_per_frame(
                ts, frame_id, img_np, tracks, track_states
            )

    # ------------------------------------------------------------------
    # Track-ID allocation helpers
    # ------------------------------------------------------------------

    def _allocate_next_id(self):
        """Return the next unused track ID and mark it as assigned."""
        while (self._next_track_id in self._assigned_ids
               or self._next_track_id in self._known_input_ids):
            self._next_track_id += 1
        result = self._next_track_id
        self._assigned_ids.add(result)
        self._next_track_id += 1
        return result

    def _get_or_remap_input_id(self, original_id):
        """
        Return the ID to use for an input track. If the input's original
        ID collides with an ID already handed out to a refiner-created
        track, remap it to a fresh unused ID. The remap is cached so the
        same input track keeps the same resolved ID across frames.
        """
        if original_id in self._input_id_remap:
            return self._input_id_remap[original_id]
        if original_id in self._assigned_ids:
            new_id = self._allocate_next_id()
            self._input_id_remap[original_id] = new_id
            self._known_input_ids.add(new_id)
            return new_id
        self._input_id_remap[original_id] = original_id
        self._known_input_ids.add(original_id)
        return original_id

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

        # Store the text query for the video predictor.  SAM3's
        # add_prompt(text_str=...) applies globally to ALL frames
        # and resets the inference state each time it is called, so
        # we must only call it once (on the first frame).  The video
        # predictor's detector will run on every frame automatically
        # when a text prompt is set.
        if self._add_new_objects and self._text_query_list:
            if local_frame_idx == 0:
                text_query = ', '.join(self._text_query_list)
                self._text_prompt_frames[0] = text_query

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

        try:
            self._run_finalize_propagation()
        except Exception as e:
            import sys, traceback
            sys.stderr.write(f"[SAM3 Refiner] ERROR in propagation: {e}\n")
            traceback.print_exc(file=sys.stderr)

        # Split tracks that have large spatial jumps (identity switches)
        self._split_jumping_tracks()

        # Stitch tracks that end near where another begins (same object
        # that got different IDs across chunk boundaries)
        self._stitch_tracks()

        output_tracks = []
        for obj_id, history in self._propagated_tracks.items():
            if len(history) > 0:
                output_tracks.append(Track(obj_id, list(history)))

        return ObjectTrackSet(output_tracks)

    def _split_jumping_tracks(self):
        """
        Post-process propagated tracks to split any that jump
        unreasonably far between consecutive frames.  A jump is
        detected when the bounding-box center moves more than the
        diagonal of the box (i.e. more than its own size).
        """
        new_tracks = {}
        next_id = max(self._propagated_tracks.keys(), default=-1) + 1

        for oid, states in list(self._propagated_tracks.items()):
            states.sort(key=lambda s: s.frame_id)
            if len(states) <= 1:
                new_tracks[oid] = states
                continue

            segments = [[states[0]]]
            for i in range(1, len(states)):
                prev_det = segments[-1][-1].detection()
                curr_det = states[i].detection()
                pb = prev_det.bounding_box
                cb = curr_det.bounding_box

                # Center of previous and current boxes
                pcx = (pb.min_x() + pb.max_x()) / 2
                pcy = (pb.min_y() + pb.max_y()) / 2
                ccx = (cb.min_x() + cb.max_x()) / 2
                ccy = (cb.min_y() + cb.max_y()) / 2

                dist = ((ccx - pcx) ** 2 + (ccy - pcy) ** 2) ** 0.5

                # Max allowed jump: larger of the two box diagonals
                pw = pb.max_x() - pb.min_x()
                ph = pb.max_y() - pb.min_y()
                cw = cb.max_x() - cb.min_x()
                ch = cb.max_y() - cb.min_y()
                max_diag = max((pw**2 + ph**2) ** 0.5,
                               (cw**2 + ch**2) ** 0.5)

                # Threshold: 2.5x diagonal with a floor of 100px
                # (small objects need room) and a ceiling of 200px
                # (large objects shouldn't link across the frame).
                threshold = min(max(max_diag * 2.5, 100), 200)

                if dist > threshold:
                    # Start a new segment
                    segments.append([states[i]])
                else:
                    segments[-1].append(states[i])

            # Keep the first (longest existing ID) segment under the
            # original ID; assign new IDs to split-off segments
            best_idx = max(range(len(segments)), key=lambda j: len(segments[j]))
            for j, seg in enumerate(segments):
                if j == best_idx:
                    new_tracks[oid] = seg
                elif len(seg) >= 2:
                    new_tracks[next_id] = seg
                    next_id += 1
                # Drop single-frame split-off fragments

        self._propagated_tracks = new_tracks

    @staticmethod
    def _track_velocity(states, from_end=True, n_frames=5):
        """
        Estimate velocity (vx, vy) in pixels/frame from the last (or
        first) *n_frames* states of a track using linear regression of
        the bounding-box centres.
        """
        if len(states) < 2:
            return 0.0, 0.0
        if from_end:
            seg = states[-n_frames:]
        else:
            seg = states[:n_frames]
        frames = [s.frame_id for s in seg]
        cxs = [(s.detection().bounding_box.min_x() +
                s.detection().bounding_box.max_x()) / 2 for s in seg]
        cys = [(s.detection().bounding_box.min_y() +
                s.detection().bounding_box.max_y()) / 2 for s in seg]
        n = len(frames)
        if n < 2:
            return 0.0, 0.0
        fm = sum(frames) / n
        xm = sum(cxs) / n
        ym = sum(cys) / n
        denom = sum((f - fm) ** 2 for f in frames)
        if denom < 1e-9:
            return 0.0, 0.0
        vx = sum((f - fm) * (x - xm) for f, x in zip(frames, cxs)) / denom
        vy = sum((f - fm) * (y - ym) for f, y in zip(frames, cys)) / denom
        return vx, vy

    def _stitch_tracks(self, max_frame_gap=5):
        """
        Merge tracks where one ends near where another begins.

        Uses velocity-based position prediction: extrapolate the ending
        track's motion to estimate where it would be at the starting
        track's first frame, then compare that predicted position to
        the actual start.  This allows fast-moving objects to be linked
        across gaps that would be too large for a raw-distance check.

        Also falls back to raw distance for slow/stationary objects.
        """
        track_info = {}
        for oid, states in self._propagated_tracks.items():
            if not states:
                continue
            states.sort(key=lambda s: s.frame_id)
            sdet = states[0].detection()
            edet = states[-1].detection()
            sb = sdet.bounding_box
            eb = edet.bounding_box
            evx, evy = self._track_velocity(states, from_end=True)
            svx, svy = self._track_velocity(states, from_end=False)
            track_info[oid] = {
                'start_frame': states[0].frame_id,
                'end_frame': states[-1].frame_id,
                'start_cx': (sb.min_x() + sb.max_x()) / 2,
                'start_cy': (sb.min_y() + sb.max_y()) / 2,
                'start_diag': ((sb.max_x()-sb.min_x())**2 + (sb.max_y()-sb.min_y())**2) ** 0.5,
                'end_cx': (eb.min_x() + eb.max_x()) / 2,
                'end_cy': (eb.min_y() + eb.max_y()) / 2,
                'end_diag': ((eb.max_x()-eb.min_x())**2 + (eb.max_y()-eb.min_y())**2) ** 0.5,
                'end_vx': evx, 'end_vy': evy,
                'start_vx': svx, 'start_vy': svy,
            }

        merged = True
        while merged:
            merged = False
            candidates = []
            for oid_end, ie in track_info.items():
                for oid_start, ist in track_info.items():
                    if oid_end == oid_start:
                        continue
                    frame_gap = ist['start_frame'] - ie['end_frame']
                    if frame_gap < 1 or frame_gap > max_frame_gap:
                        continue

                    # Raw distance between endpoints
                    raw_dist = ((ie['end_cx'] - ist['start_cx'])**2 +
                                (ie['end_cy'] - ist['start_cy'])**2) ** 0.5

                    # Predicted position by extrapolating end velocity
                    pred_cx = ie['end_cx'] + ie['end_vx'] * frame_gap
                    pred_cy = ie['end_cy'] + ie['end_vy'] * frame_gap
                    pred_dist = ((pred_cx - ist['start_cx'])**2 +
                                 (pred_cy - ist['start_cy'])**2) ** 0.5

                    # Also try back-projecting the start velocity
                    bpred_cx = ist['start_cx'] - ist['start_vx'] * frame_gap
                    bpred_cy = ist['start_cy'] - ist['start_vy'] * frame_gap
                    bpred_dist = ((bpred_cx - ie['end_cx'])**2 +
                                  (bpred_cy - ie['end_cy'])**2) ** 0.5

                    # Use the best (smallest) of raw, forward-predicted,
                    # and backward-predicted distances
                    best_dist = min(raw_dist, pred_dist, bpred_dist)

                    max_diag = max(ie['end_diag'], ist['start_diag'], 10)
                    threshold = min(max(max_diag * 2.5, 100), 200)
                    if best_dist < threshold:
                        candidates.append((best_dist, oid_end, oid_start))

            if not candidates:
                break

            candidates.sort()
            used = set()
            for dist_val, oid_end, oid_start in candidates:
                if oid_end in used or oid_start in used:
                    continue
                self._propagated_tracks[oid_end].extend(
                    self._propagated_tracks[oid_start])
                self._propagated_tracks[oid_end].sort(
                    key=lambda s: s.frame_id)
                del self._propagated_tracks[oid_start]

                # Update track_info for merged track
                states = self._propagated_tracks[oid_end]
                edet = states[-1].detection()
                eb = edet.bounding_box
                evx, evy = self._track_velocity(states, from_end=True)
                track_info[oid_end]['end_frame'] = states[-1].frame_id
                track_info[oid_end]['end_cx'] = (eb.min_x() + eb.max_x()) / 2
                track_info[oid_end]['end_cy'] = (eb.min_y() + eb.max_y()) / 2
                track_info[oid_end]['end_diag'] = (
                    (eb.max_x()-eb.min_x())**2 + (eb.max_y()-eb.min_y())**2) ** 0.5
                track_info[oid_end]['end_vx'] = evx
                track_info[oid_end]['end_vy'] = evy
                del track_info[oid_start]

                used.add(oid_end)
                used.add(oid_start)
                merged = True

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
        for (oid, fidx), result_tuple in all_results.items():
            mask, box_xyxy = result_tuple[0], result_tuple[1]
            score = result_tuple[2] if len(result_tuple) > 2 else 1.0

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

            confidence = float(score)
            dot = DetectedObjectType(class_name, confidence)
            det = DetectedObject(bbox, confidence, dot)

            # Set the full binary mask (cropped to bbox) on the
            # detection.  The CSV writer's mask_to_poly_points path
            # will convert this to multi-contour polygons with proper
            # (poly)/(hole) tags, handling disjoint mask regions that
            # a single flattened polygon cannot represent.
            _set_mask_on_detection(det, mask, bbox)

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
                    tid = self._allocate_next_id()
                    output_tracks.append(Track(tid, [new_state]))

        # Pass through unprocessed input tracks. Rebuild the Track with
        # the resolved ID when the input's original ID got remapped.
        if not self._resegment_existing:
            for tid, (track, state, det) in track_states.items():
                if tid not in processed_track_ids:
                    output_tracks.append(
                        track if track.id == tid else Track(tid, list(track))
                    )
                    processed_track_ids.add(tid)

        for track in tracks.tracks():
            tid = self._input_id_remap.get(track.id, track.id)
            if tid not in processed_track_ids and tid not in track_states:
                output_tracks.append(
                    track if track.id == tid else Track(tid, list(track))
                )

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
            'grounding_model_id': '',
            'device': 'cuda',
            'overwrite_existing': 'True',
            'output_type': 'polygon',
            'polygon_simplification': '0.01',
            'text_query': '',
            'detection_threshold': '0.3',
            'text_threshold': '0.25',
            'iou_threshold': '0.5',
            'add_new_objects': 'False',
            'max_new_objects': '50',
            'min_mask_area': '10',
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
        model_config.device = self._kwiver_config['device']

        # Load Grounding DINO if configured (for text-based detection)
        gid = self._kwiver_config.get('grounding_model_id', '')
        if gid and gid.lower() not in ('', 'none', 'false'):
            model_config.grounding_model_id = gid
        else:
            model_config.grounding_model_id = None

        self._model_manager.init_models(model_config, use_video_predictor=False)

        # Parse config values
        self._overwrite_existing = parse_bool(self._kwiver_config['overwrite_existing'])
        self._output_type = self._kwiver_config['output_type']
        self._polygon_simplification = float(self._kwiver_config['polygon_simplification'])
        self._add_new_objects = parse_bool(self._kwiver_config['add_new_objects'])
        self._max_new_objects = int(self._kwiver_config['max_new_objects'])
        self._min_mask_area = int(self._kwiver_config['min_mask_area'])
        self._detection_threshold = float(self._kwiver_config['detection_threshold'])
        self._text_threshold = float(self._kwiver_config['text_threshold'])
        self._iou_threshold = float(self._kwiver_config['iou_threshold'])

        tq = self._kwiver_config.get('text_query', '')
        self._text_query_list = [q.strip() for q in tq.split(',') if q.strip()] if tq else []

        return True

    def check_configuration(self, cfg):
        """Check if the configuration is valid."""
        return True

    def refine(self, image_data, detections):
        """
        Refine detections by adding segmentation masks.  When
        ``add_new_objects`` is enabled and a ``text_query`` is set,
        also detects new objects via Grounding DINO before segmenting.

        Args:
            image_data: Image container
            detections: DetectedObjectSet to refine

        Returns:
            DetectedObjectSet: Refined detections with masks
        """
        import torch

        img_np = image_to_rgb_numpy(image_data)

        # Detect new objects with text query if configured
        if self._add_new_objects and self._text_query_list:
            new_dets = self._model_manager.detect_with_text(
                img_np, self._text_query_list,
                self._detection_threshold, self._text_threshold,
            )
            # Suppress new detections that overlap with existing ones
            suppress_boxes = []
            for det in detections:
                bb = det.bounding_box
                suppress_boxes.append(
                    [bb.min_x(), bb.min_y(), bb.max_x(), bb.max_y()])

            new_count = 0
            for box, score, class_name in new_dets:
                if new_count >= self._max_new_objects:
                    break
                overlaps = any(
                    compute_iou(box, sb) > self._iou_threshold
                    for sb in suppress_boxes)
                if not overlaps:
                    bbox = BoundingBoxD(box[0], box[1], box[2], box[3])
                    dot = DetectedObjectType(class_name, score)
                    new_det = DetectedObject(bbox, score, dot)
                    detections.add(new_det)
                    suppress_boxes.append(list(box))
                    new_count += 1

        if len(detections) == 0:
            return DetectedObjectSet()

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
            mask_area = np.sum(mask)
            if mask_area < self._min_mask_area:
                continue

            # Set mask on detection for multi-contour polygon output
            _set_mask_on_detection(det, mask, det.bounding_box)

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
