#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved
# BSD 3-Clause License. See either the root top-level LICENSE file or
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

"""
Interactive Segmentation Service

A persistent process that keeps segmentation algorithms loaded and handles
inference requests via stdin/stdout JSON protocol. Designed to be spawned by both
Desktop (Electron) and Web (Girder) platforms for fast interactive segmentation.

This service uses KWIVER vital algorithms configured via config files:
- SegmentViaPoints: For point-based segmentation
- PerformTextQuery: For text-based detection/segmentation (optional)

Usage:
    python -m viame.core.interactive_segmentation --config /path/to/config.pipe
    python -m viame.core.interactive_segmentation --config /path/to/config.pipe --plugin-path /path/to/plugins

Protocol:
    Input (JSON per line on stdin):
    {
        "id": "unique-request-id",
        "command": "predict",
        "image_path": "/path/to/frame.png",
        "points": [[x1, y1], [x2, y2], ...],
        "point_labels": [1, 1, 0, ...],
    }

    Output (JSON per line on stdout):
    {
        "id": "unique-request-id",
        "success": true,
        "polygon": [[x1, y1], [x2, y2], ...],
        "bounds": [x_min, y_min, x_max, y_max],
        "score": 0.95,
    }

    Commands:
    - "predict": Run point-based segmentation on an image
    - "text_query": Run text-based detection/segmentation (if configured)
    - "set_image": Pre-load an image for multiple predictions
    - "clear_image": Clear the cached image
    - "shutdown": Gracefully terminate the service
"""

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@contextlib.contextmanager
def suppress_stdout():
    """
    Context manager to redirect stdout to stderr.

    This prevents library warnings/prints from corrupting
    the JSON protocol on stdout.
    """
    original_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = original_stdout


class InteractiveSegmentationService:
    """
    Interactive Segmentation Service using KWIVER vital algorithms.

    Handles stdin/stdout JSON protocol communication and delegates
    to configured vital algorithms for inference.
    """

    def __init__(
        self,
        segment_via_points_algo,
        perform_text_query_algo=None,
        image_io_algo=None,
        hole_policy: str = "allow",
        multipolygon_policy: str = "allow",
        max_polygon_points: int = 25,
        adaptive_simplify: bool = False,
        max_polygon_points_limit: int = 100,
        plugin_paths: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the service with configured algorithms.

        Args:
            segment_via_points_algo: Configured SegmentViaPoints algorithm instance
            perform_text_query_algo: Optional configured PerformTextQuery algorithm
            image_io_algo: Optional configured ImageIO algorithm for loading images
            hole_policy: How to handle holes in masks ('allow' or 'remove')
            multipolygon_policy: How to handle multiple polygons ('allow', 'convex_hull', 'largest')
            max_polygon_points: Maximum number of points in output polygons
            max_polygon_points_limit: Ceiling the point budget grows to for
                point-click masks too complex for max_polygon_points
            adaptive_simplify: Use adaptive polygon simplification
            plugin_paths: Extra plugin paths, forwarded to the embedded stereo
                warper used by stereo_segment.
            device: Device override, forwarded to the embedded stereo warper.
        """
        self._segment_algo = segment_via_points_algo
        self._text_query_algo = perform_text_query_algo
        self._image_io_algo = image_io_algo
        self._hole_policy = hole_policy
        self._multipolygon_policy = multipolygon_policy
        self._max_polygon_points = max_polygon_points
        self._adaptive_simplify = adaptive_simplify
        self._max_polygon_points_limit = max(max_polygon_points, max_polygon_points_limit)
        self._prompt_instances = None
        self._prompt_instances_key = None
        self._prompt_instances_like = None
        self._current_image_path: Optional[str] = None
        self._current_image_container = None
        # Embedded interactive-stereo warper for stereo_segment (lazy). Reuses
        # the configured stereo backend (epipolar or dense disparity); does NOT
        # load SAM, so the segmentation model is never loaded twice.
        self._plugin_paths = plugin_paths or []
        self._device = device
        self._stereo_warper = None

    def _log(self, message: str) -> None:
        """Log to stderr (stdout is reserved for JSON responses)."""
        print(f"[SegmentationService] {message}", file=sys.stderr, flush=True)

    def has_text_query(self) -> bool:
        return self._text_query_algo is not None

    def set_text_query_algo(self, algo) -> None:
        self._text_query_algo = algo

    def _send_response(self, response: Dict[str, Any]) -> None:
        """Send JSON response to stdout."""
        print(json.dumps(response), flush=True)

    def _send_error(self, request_id: Optional[str], error: str) -> None:
        """Send error response."""
        self._send_response({
            "id": request_id,
            "success": False,
            "error": error,
        })

    _VIDEO_EXTENSIONS = {'.avi', '.mp4', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpg', '.mpeg', '.m4v'}

    def _is_video_file(self, path: str) -> bool:
        """Check if a path is a video file based on extension."""
        ext = os.path.splitext(path)[1].lower()
        return ext in self._VIDEO_EXTENSIONS

    def _load_image(self, image_path: str, frame_time: float = None):
        """Load an image (or video frame) and return a vital ImageContainer."""
        if self._is_video_file(image_path) and frame_time is not None:
            return self._load_video_frame(image_path, frame_time)

        # Use KWIVER ImageIO algorithm if available (preferred - handles memory layout properly)
        if self._image_io_algo is not None:
            return self._image_io_algo.load(image_path)

        # Fallback to VitalPIL conversion
        from kwiver.vital.types import ImageContainer
        from kwiver.vital.util import VitalPIL
        from PIL import Image as PILImage

        pil_img = PILImage.open(image_path).convert("RGB")
        vital_img = VitalPIL.from_pil(pil_img)
        return ImageContainer(vital_img)

    def _load_video_frame(self, video_path: str, frame_time: float):
        """Extract a single frame from a video at the given time (seconds)."""
        from kwiver.vital.algo import VideoInput
        from kwiver.vital.types import Timestamp

        # Cache the video reader for repeated access to the same video. Build
        # the reader locally and only commit it to self AFTER a successful
        # open(), so a failed open (e.g. a missing/moved file) does not leave a
        # half-initialized reader that breaks every subsequent load.
        if (getattr(self, '_video_reader', None) is None
                or getattr(self, '_video_reader_path', None) != video_path):
            reader = VideoInput.create("vidl_ffmpeg")
            cfg = reader.get_configuration()
            cfg.set_value("time_source", "start_at_0")
            reader.set_configuration(cfg)
            reader.open(video_path)
            # Determine video FPS by reading the first frame
            ts = Timestamp()
            reader.next_frame(ts, 0)
            self._video_reader = reader
            self._video_reader_path = video_path
            self._video_fps = reader.frame_rate()

        # Convert time to frame number using the video's native FPS.
        # The ffmpeg reader uses 1-based frame numbering (frame 0 is
        # a pre-first-frame state), so add 1.
        target_frame = round(frame_time * self._video_fps) + 1
        target_frame = max(1, target_frame)

        ts = Timestamp()
        self._video_reader.seek_frame(ts, target_frame, 0)
        image = self._video_reader.frame_image()

        if image is None:
            raise RuntimeError(f"Could not read frame at t={frame_time:.3f}s "
                               f"(frame {target_frame}) from {video_path}")

        self._log(f"Loaded video t={frame_time:.3f}s (frame {target_frame}) "
                  f"from {os.path.basename(video_path)}")
        return image

    def _simplify_ring(self, ring, grow):
        from viame.core.segmentation_utils import simplify_polygon_within_error

        limit = self._max_polygon_points_limit if grow else self._max_polygon_points
        return simplify_polygon_within_error(
            ring, self._max_polygon_points, limit, adaptive=self._adaptive_simplify)

    def _detections_to_response(self, detected_objects, keep_points=None, instances=False) -> List[Dict[str, Any]]:
        """Convert DetectedObjectSet to response dictionaries.

        `polygon` is the single polygon the configured policies leave; the
        `polygons` list always carries every component of the mask, and any
        component holding one of `keep_points` (the positive prompts) survives
        the small-component filter. For a buffer of mask `instances` the
        bounds span every polygon, and a ring too complex for
        max_polygon_points takes up to max_polygon_points_limit."""
        from viame.core.segmentation_utils import mask_to_polygon, mask_to_polygons

        results = []

        for det_obj in detected_objects:
            bbox = det_obj.bounding_box
            bounds = [bbox.min_x(), bbox.min_y(), bbox.max_x(), bbox.max_y()]
            score = det_obj.confidence

            # Get polygon from mask if available
            polygon = None
            polygons_data = None
            rle_mask = None
            mask_shape = None

            if det_obj.mask is not None:
                mask = det_obj.mask.image().asarray()
                if mask is not None and mask.size > 0:
                    # Convert to binary mask
                    if mask.ndim == 3:
                        mask = mask[:, :, 0]
                    mask = (mask > 0).astype(np.uint8)

                    polygon, poly_bounds = mask_to_polygon(
                        mask, self._hole_policy, self._multipolygon_policy
                    )

                    # Get multi-polygon data with holes
                    offset_x, offset_y = bbox.min_x(), bbox.min_y()
                    raw_polygons, mp_bounds = mask_to_polygons(
                        mask, self._hole_policy, "allow",
                        keep_points=[[x - offset_x, y - offset_y] for x, y in (keep_points or [])],
                    )

                    if polygon:
                        polygon = self._simplify_ring(polygon, instances)

                    if raw_polygons:
                        for poly_data in raw_polygons:
                            poly_data["exterior"] = self._simplify_ring(poly_data["exterior"], instances)
                            poly_data["holes"] = [
                                self._simplify_ring(hole, instances) for hole in poly_data["holes"]]

                    # Offset polygon to original image coordinates (mask is cropped to bbox)
                    if polygon:
                        polygon = [[x + offset_x, y + offset_y] for x, y in polygon]

                    # Offset multi-polygon data
                    if raw_polygons:
                        for poly_data in raw_polygons:
                            poly_data["exterior"] = [
                                [x + offset_x, y + offset_y]
                                for x, y in poly_data["exterior"]
                            ]
                            poly_data["holes"] = [
                                [[x + offset_x, y + offset_y] for x, y in hole]
                                for hole in poly_data["holes"]
                            ]
                        polygons_data = raw_polygons

                    # Use polygon-derived bounds instead of detection bbox
                    if instances and raw_polygons and mp_bounds != [0, 0, 0, 0]:
                        poly_bounds = mp_bounds
                    if polygon and poly_bounds and poly_bounds != [0, 0, 0, 0]:
                        bounds = [
                            poly_bounds[0] + offset_x, poly_bounds[1] + offset_y,
                            poly_bounds[2] + offset_x, poly_bounds[3] + offset_y,
                        ]
                    elif raw_polygons and mp_bounds and mp_bounds != [0, 0, 0, 0]:
                        bounds = [
                            mp_bounds[0] + offset_x, mp_bounds[1] + offset_y,
                            mp_bounds[2] + offset_x, mp_bounds[3] + offset_y,
                        ]

                    # Create RLE mask for efficient transfer
                    flat_mask = mask.flatten().astype(np.uint8)
                    rle_mask = []
                    if len(flat_mask) > 0:
                        current_val = flat_mask[0]
                        count = 1
                        for val in flat_mask[1:]:
                            if val == current_val:
                                count += 1
                            else:
                                rle_mask.append([int(current_val), count])
                                current_val = val
                                count = 1
                        rle_mask.append([int(current_val), count])
                    mask_shape = list(mask.shape)

            result = {
                "polygon": polygon,
                "bounds": bounds,
                "score": score,
            }

            if polygons_data is not None:
                result["polygons"] = polygons_data

            if rle_mask is not None:
                result["rle_mask"] = rle_mask
                result["mask_shape"] = mask_shape

            # Get class label if available
            if det_obj.type is not None:
                result["label"] = det_obj.type.get_most_likely_class()

            results.append(result)

        return results

    def handle_predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a predict command using SegmentViaPoints algorithm."""
        from kwiver.vital.types import Point2d

        image_path = request.get("image_path")
        points = request.get("points", [])
        point_labels = request.get("point_labels", [])
        frame_time = request.get("frame_time")
        line = request.get("line")

        if not image_path:
            raise ValueError("image_path is required")
        if not points:
            raise ValueError("At least one point is required")
        if len(points) != len(point_labels):
            raise ValueError("points and point_labels must have same length")

        # Load image if different from cached (include time in cache key for videos)
        cache_key = f"{image_path}:t={frame_time}" if frame_time is not None else image_path
        if self._current_image_path != cache_key:
            self._log(f"Loading image: {image_path}" + (f" t={frame_time:.3f}s" if frame_time is not None else ""))
            self._current_image_container = self._load_image(image_path, frame_time)
            self._current_image_path = cache_key

        # Convert points to vital Point2d objects using x,y constructor
        vital_points = [Point2d(float(p[0]), float(p[1])) for p in points]
        vital_labels = [int(label) for label in point_labels]
        positives = [[float(p[0]), float(p[1])] for p, l in zip(points, vital_labels) if l == 1]
        instances = not (line and len(line) >= 2)

        if not positives:
            raise ValueError("Add a foreground point first; background points only trim existing masks")

        # Run segmentation (suppress stdout to prevent library warnings corrupting JSON)
        with suppress_stdout():
            if instances:
                detected_objects = self._segment_instances(cache_key, points, vital_labels)
            else:
                detected_objects = self._segment_algo.segment(
                    self._current_image_container,
                    vital_points,
                    vital_labels
                )
                detected_objects = self._fit_to_line(
                    detected_objects, line, vital_points, vital_labels)

            # Convert results
            results = self._detections_to_response(
                detected_objects, keep_points=positives, instances=instances)

        if results:
            # Return the best result (first one)
            response = results[0]
            response["success"] = True
            return response
        else:
            return {
                "success": True,
                "polygon": None,
                "bounds": None,
                "score": 0.0,
            }

    def _full_mask(self, det, dims):
        """A detection's cropped mask placed on a full-frame canvas."""
        if det is None or det.mask is None:
            return None
        crop = det.mask.image().asarray()
        if crop is None or crop.size == 0:
            return None
        if crop.ndim == 3:
            crop = crop[:, :, 0]
        box = det.bounding_box
        x0, y0 = int(box.min_x()), int(box.min_y())
        full = np.zeros(dims, dtype=bool)
        h = min(crop.shape[0], dims[0] - y0)
        w = min(crop.shape[1], dims[1] - x0)
        if h > 0 and w > 0:
            full[y0:y0 + h, x0:x0 + w] = crop[:h, :w] > 0
        return full

    def _detection_from_mask(self, mask, like):
        """A detection set holding `mask`, scored and typed like `like`."""
        from kwiver.vital.types import (
            DetectedObject, DetectedObjectSet, BoundingBoxD, ImageContainer, Image)

        result = DetectedObjectSet()
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return result
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        bbox = BoundingBoxD(x0, y0, x1, y1)
        det = (DetectedObject(bbox, like.confidence, like.type)
               if like.type is not None else DetectedObject(bbox, like.confidence))
        crop = np.ascontiguousarray(mask[y0:y1 + 1, x0:x1 + 1].astype(np.uint8))
        det.mask = ImageContainer(Image(crop))
        result.add(det)
        return result

    def _segment_instances(self, cache_key, points, labels):
        """Point clicks as a buffer of per-object mask instances (see
        PromptInstances), returned as the one detection covering them all."""
        from kwiver.vital.types import Point2d, DetectedObjectSet
        from viame.core.segmentation_utils import PromptInstances

        image = self._current_image_container
        dims = (image.height(), image.width())
        last = {}

        def predict(pos, neg):
            objs = self._segment_algo.segment(
                image,
                [Point2d(float(x), float(y)) for x, y in list(pos) + list(neg)],
                [1] * len(pos) + [0] * len(neg))
            det = next(iter(objs), None) if objs is not None else None
            if det is not None:
                last["det"] = det
            return self._full_mask(det, dims)

        if self._prompt_instances_key != cache_key or self._prompt_instances.shape != dims:
            self._prompt_instances = PromptInstances(dims, None)
            self._prompt_instances_key = cache_key
            self._prompt_instances_like = None
        buffer = self._prompt_instances
        buffer.predict = predict
        try:
            buffer.sync(list(zip(points, labels)))
        except Exception:
            self._prompt_instances_key = None
            raise
        finally:
            buffer.predict = None
        self._log(f"Point prompts held as {len(buffer.instances)} mask instance(s)")
        if "det" in last:
            self._prompt_instances_like = last["det"]
        if self._prompt_instances_like is None:
            return DetectedObjectSet()
        return self._detection_from_mask(buffer.mask(), self._prompt_instances_like)

    def _fit_to_line(self, detected_objects, line, vital_points, vital_labels):
        """Keep a mask prompted from a head/tail line in scale with that line:
        when it comes out far larger, retry with background prompts ringing
        the line, then clip whatever remains to a band around the line."""
        from kwiver.vital.types import (
            Point2d, DetectedObject, DetectedObjectSet, BoundingBoxD,
            ImageContainer, Image)
        from viame.core.segmentation_utils import (
            mask_oversized_for_line, mask_undersized_for_line,
            line_background_points, clip_mask_to_line)

        def first(objects):
            return next(iter(objects), None) if objects is not None else None

        def oversized(det):
            box = det.bounding_box
            return mask_oversized_for_line(
                [box.min_x(), box.min_y(), box.max_x(), box.max_y()], line)

        det = first(detected_objects)
        if det is None or not oversized(det):
            return detected_objects

        image = self._current_image_container
        background = line_background_points(line, (image.width(), image.height()))
        if background:
            self._log("Mask out of scale with its line; retrying with background prompts")
            retried = self._segment_algo.segment(
                image,
                list(vital_points) + [Point2d(x, y) for x, y in background],
                list(vital_labels) + [0] * len(background))
            retry = first(retried)
            if retry is not None:
                if not oversized(retry):
                    return retried
                det = retry

        self._log("Mask still out of scale with its line; clipping to the line")
        result = DetectedObjectSet()
        if det.mask is None:
            return result
        box = det.bounding_box
        clipped = clip_mask_to_line(
            det.mask.image().asarray(), (box.min_x(), box.min_y()), line)
        if clipped is None:
            return result
        mask, (x0, y0) = clipped
        bounds = [x0, y0, x0 + mask.shape[1] - 1, y0 + mask.shape[0] - 1]
        if mask_undersized_for_line(bounds, line):
            # Only a scrap of the mask lay along the line: report no mask so
            # the caller keeps its own line-derived box.
            return result
        bbox = BoundingBoxD(*bounds)
        fitted = (DetectedObject(bbox, det.confidence, det.type)
                  if det.type is not None else DetectedObject(bbox, det.confidence))
        fitted.mask = ImageContainer(Image(np.ascontiguousarray(mask)))
        result.add(fitted)
        return result

    def handle_polygon_keypoints(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Head/tail keypoints for a polygon, derived the way the keypoint
        pipelines derive them from a mask (add_keypoints_from_mask)."""
        from viame.core.segmentation_utils import polygons_to_keypoints

        polygons = request.get("polygons")
        if polygons is None:
            polygon = request.get("polygon")
            if not polygon or len(polygon) < 3:
                raise ValueError("polygon with at least three points is required")
            polygons = [{"exterior": polygon, "holes": []}]
        if not polygons:
            raise ValueError("at least one polygon is required")
        with suppress_stdout():
            keypoints = polygons_to_keypoints(polygons)
        if keypoints is None:
            return {"success": False, "error": "Could not derive head/tail from the polygon"}
        head, tail = keypoints
        return {"success": True, "head": head, "tail": tail}

    def handle_text_query(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a text_query command using PerformTextQuery algorithm."""
        if self._text_query_algo is None:
            raise ValueError("Text query not configured")

        from kwiver.vital.types import Timestamp

        image_path = request.get("image_path")
        text = request.get("text", "")
        frame_time = request.get("frame_time")

        if not image_path:
            raise ValueError("image_path is required")
        if not text:
            raise ValueError("text query is required")

        # Load image (or video frame at the given time)
        image_container = self._load_image(image_path, frame_time)

        # Create timestamp
        timestamp = Timestamp()
        timestamp.set_frame(0)

        # Run text query
        track_sets = self._text_query_algo.perform_query(
            text,
            [image_container],
            [timestamp],
            []
        )

        # Extract detections from track set
        detections = []
        if track_sets and len(track_sets) > 0:
            track_set = track_sets[0]
            for track in track_set.tracks():
                for state in track:
                    det_obj = state.detection()
                    bbox = det_obj.bounding_box
                    bounds = [bbox.min_x(), bbox.min_y(), bbox.max_x(), bbox.max_y()]
                    score = det_obj.confidence

                    detection = {
                        "bounds": bounds,
                        "box": bounds,
                        "score": score,
                        "track_id": track.id,
                    }

                    if det_obj.type is not None:
                        detection["label"] = det_obj.type.get_most_likely_class()

                    # Get polygon from mask if available
                    if det_obj.mask is not None:
                        mask = det_obj.mask.image().asarray()
                        if mask is not None and mask.size > 0:
                            from viame.core.segmentation_utils import (
                                mask_to_polygon,
                                simplify_polygon_to_max_points,
                                adaptive_simplify_polygon,
                            )
                            if mask.ndim == 3:
                                mask = mask[:, :, 0]
                            mask = (mask > 0).astype(np.uint8)
                            polygon, _ = mask_to_polygon(
                                mask, self._hole_policy, self._multipolygon_policy
                            )

                            # Offset polygon by bbox origin (mask is relative to bbox)
                            if polygon:
                                ox, oy = bounds[0], bounds[1]
                                polygon = [[p[0] + ox, p[1] + oy] for p in polygon]

                            # Simplify polygon if needed
                            if polygon and len(polygon) > self._max_polygon_points:
                                if self._adaptive_simplify:
                                    polygon = adaptive_simplify_polygon(
                                        polygon, self._max_polygon_points, min_points=4
                                    )
                                else:
                                    polygon = simplify_polygon_to_max_points(
                                        polygon, self._max_polygon_points
                                    )

                            detection["polygon"] = polygon

                    detections.append(detection)

        return {
            "success": True,
            "detections": detections,
        }

    def handle_set_image(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-load an image for multiple predictions."""
        image_path = request.get("image_path")
        frame_time = request.get("frame_time")
        if not image_path:
            raise ValueError("image_path is required")

        cache_key = f"{image_path}:t={frame_time}" if frame_time is not None else image_path
        self._log(f"Pre-loading image: {image_path}" + (f" t={frame_time:.3f}s" if frame_time is not None else ""))
        self._current_image_container = self._load_image(image_path, frame_time)
        self._current_image_path = cache_key

        return {
            "success": True,
            "message": f"Image loaded: {image_path}",
        }

    def handle_clear_image(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Clear the cached image."""
        self._current_image_container = None
        self._current_image_path = None
        return {
            "success": True,
            "message": "Image cache cleared",
        }

    def warmup(self) -> None:
        """Eagerly load the point-segmentation model so the first click is fast.

        Called when the user enters point-segmentation mode (mode entry), so the
        segmentation model loads then rather than on the first click. The caller
        should suppress stdout, since model init may print.

        The text-query model is deliberately NOT loaded here: it is large and
        commonly a different (heavier) backend than the segmenter, and may go
        unused for a whole segmentation session. We only verify it is configured
        and defer its load to the first text_query request (perform_query lazily
        calls ``_ensure_model``). Point-segmentation warmup failure is non-fatal
        (it falls back to loading on first use)."""
        seg = self._segment_algo
        if seg is not None and hasattr(seg, "_ensure_model"):
            try:
                seg._ensure_model()
            except Exception as e:
                self._log(f"Warning: segmenter model warmup failed: {e}")

        # Text query: presence check only -- do not load the model here.
        if self._text_query_algo is not None:
            self._log("Text query configured; its model loads on the first "
                      "text query.")

    def set_stereo_warper(self, warper) -> None:
        """Inject a shared, already-built InteractiveStereoService.

        Used by the unified interactive service so stereo_segment reuses the
        same stereo backend instance that interactive-stereo mode loaded,
        instead of constructing a second one (the stereo model is never loaded
        twice). When unset, _get_stereo_warper() builds its own on first use.
        """
        self._stereo_warper = warper

    def _get_stereo_warper(self, calibration_file: Optional[str] = None):
        """Lazily build an embedded interactive-stereo warper.

        Reuses interactive_stereo's configured backend -- epipolar template
        matching (model-free) or dense disparity (ComputeStereoDepthMap, e.g.
        Foundation-Stereo) -- so stereo_segment supports every warping type via
        the same vital base classes, not just the model-free path. SAM is never
        instantiated here, so the (large) segmentation model is not loaded twice.
        """
        if self._stereo_warper is None:
            from viame.core.interactive_stereo import (
                InteractiveStereoService,
                load_algorithm_from_config,
                find_viame_config,
            )
            stereo_config = find_viame_config()
            if not stereo_config:
                raise ValueError(
                    "Could not locate the interactive stereo config "
                    "(interactive_stereo_default.conf); is VIAME_INSTALL set?")
            stereo_algo, matcher, svc_cfg = load_algorithm_from_config(
                stereo_config, self._plugin_paths)
            self._stereo_warper = InteractiveStereoService(
                compute_stereo_depth_map_algo=stereo_algo,
                epipolar_matcher=matcher,
                **svc_cfg,
            )
            self._log(
                "Embedded stereo warper created "
                f"({'epipolar' if matcher is not None else 'dense'} mode)")
        if not self._stereo_warper._enabled:
            self._stereo_warper.handle_enable({"calibration_file": calibration_file})
        return self._stereo_warper

    @staticmethod
    def _polygons_area(polygons) -> float:
        """Area enclosed by {"exterior", "holes"} polygons, holes excluded."""
        def ring(points):
            xy = np.asarray(points, dtype=float)
            if xy.ndim != 2 or len(xy) < 3:
                return 0.0
            return 0.5 * abs(float(np.dot(xy[:, 0], np.roll(xy[:, 1], -1))
                                   - np.dot(xy[:, 1], np.roll(xy[:, 0], -1))))

        return sum(ring(p.get("exterior") or []) - sum(ring(h) for h in p.get("holes") or [])
                   for p in polygons or [])

    def handle_stereo_segment(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Stereo point-segmentation orchestration.

        Given a click and the already-segmented polygon on the source camera,
        this:
          1. warps a segmentation seed to the other camera using the configured
             stereo backend (epipolar or dense); when point sampling is enabled
             the seed is the coordinate-wise median of N points sampled inside
             the source polygon and warped across (noise reduction for bad
             point mappings);
          2. runs SAM on the other camera with that seed to get its polygon;
          3. optionally derives a head/tail line for each polygon and the
             stereo length measurement.

        The click may come from either camera. `source_camera` ('left' or
        'right') says which; without it the source is matched against the
        stereo pair the warper already has loaded, and taken as left otherwise.

        Without a click (an existing mask being mapped across) the seeds are
        points inside the source mask. A result whose area is out of scale with
        the source is refused rather than returned.

        Request: {
            points, point_labels,            # the source-camera click, if any
            polygon,                         # source-camera polygon
            polygons,                        # all of its parts, with holes
            source_image_path, other_image_path,
            calibration_file, frame_time
        }
        """
        points = request.get("points") or []
        point_labels = request.get("point_labels") or [1] * len(points)
        source_polygon = request.get("polygon")
        source_polygons = request.get("polygons") or (
            [{"exterior": source_polygon, "holes": []}] if source_polygon else [])
        if not source_polygon and source_polygons:
            source_polygon = source_polygons[0]["exterior"]
        source_image = request.get("source_image_path")
        other_image = request.get("other_image_path")
        frame_time = request.get("frame_time")

        if not other_image:
            raise ValueError("other_image_path is required")

        warper = self._get_stereo_warper(request.get("calibration_file"))

        side = request.get("source_camera")
        if side not in ("left", "right"):
            swapped = (source_image is not None
                       and source_image == warper._current_right_path
                       and other_image == warper._current_left_path)
            side = "right" if swapped else "left"
        from_right = side == "right"

        set_resp = warper.handle_set_frame({
            "left_image_path": other_image if from_right else source_image,
            "right_image_path": source_image if from_right else other_image,
            "frame_time": frame_time,
        })
        if not set_resp.get("disparity_ready", False):
            # Dense mode computes disparity asynchronously; wait for it.
            warper._disparity_event.wait(timeout=120.0)

        # 1. Warp a segmentation seed to the other camera.
        warp = warper.handle_transfer_segmentation_point({
            "points": points,
            "labels": point_labels,
            "polygon": source_polygon,
            "polygons": source_polygons,
            "source_camera": side,
        })
        seed_points = warp.get("transferred_points") or []
        seed_labels = warp.get("point_labels") or [1] * len(seed_points)
        # num_matched defaults to the seed count for backends that don't report
        # it; epipolar matching reports 0 when no point actually matched (it
        # substitutes the source coordinate as a fallback), which must not be
        # treated as a successful transfer.
        num_matched = warp.get("num_matched", len(seed_points))
        if not seed_points or num_matched == 0:
            return {
                "success": False,
                "error": "No stereo match found on the other camera "
                         "(epipolar matching failed); the object may be "
                         "occluded, out of frame, or the calibration is off",
            }

        # 2. Segment the other camera with SAM using the warped seed.
        seg = self.handle_predict({
            "image_path": other_image,
            "points": seed_points,
            "point_labels": seed_labels,
            "frame_time": frame_time,
        })
        other_polygon = seg.get("polygon")
        if not other_polygon:
            return {
                "success": False,
                "error": "Segmentation produced no region on the other camera "
                         "for the matched stereo point",
                "seed_points": seed_points,
                "seed_labels": seed_labels,
            }

        other_polygons = seg.get("polygons") or [{"exterior": other_polygon, "holes": []}]
        reason = warper.size_mismatch(
            self._polygons_area(source_polygons), self._polygons_area(other_polygons))
        if reason:
            return {
                "success": False,
                "error": reason,
                "size_mismatch": True,
                "seed_points": seed_points,
                "seed_labels": seed_labels,
            }

        result = {
            "success": True,
            "polygon": other_polygon,
            "polygons": other_polygons,
            "bounds": seg.get("bounds"),
            "score": seg.get("score"),
            "seed_points": seed_points,
            "seed_labels": seed_labels,
        }

        # 3. Optional head/tail lines + stereo measurement.
        if source_polygon and other_polygon:
            measure = warper.handle_measure_from_polygons({
                "polygon_left": other_polygon if from_right else source_polygon,
                "polygon_right": source_polygon if from_right else other_polygon,
            })
            if measure.get("generate_line"):
                result["generate_line"] = True
                result["line_source"] = measure.get("line_right" if from_right else "line_left")
                result["line_other"] = measure.get("line_left" if from_right else "line_right")
                if "measurement" in measure:
                    result["measurement"] = measure["measurement"]

        return result

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Route request to appropriate handler."""
        command = request.get("command")

        handlers = {
            "predict": self.handle_predict,
            "set_image": self.handle_set_image,
            "clear_image": self.handle_clear_image,
            "stereo_segment": self.handle_stereo_segment,
            "polygon_keypoints": self.handle_polygon_keypoints,
        }

        # Add text_query handler if algorithm is configured
        if self._text_query_algo is not None:
            handlers["text_query"] = self.handle_text_query

        handler = handlers.get(command)
        if not handler:
            # Provide helpful error for text_query when not configured
            if command == "text_query":
                raise ValueError(
                    "text_query command requires a perform_text_query algorithm to be "
                    "configured. SAM2 does not support text queries - install the SAM3 "
                    "add-on so interactive_text_query_default.conf pulls in "
                    "interactive_text_query_sam3.conf for text-based detection."
                )
            raise ValueError(f"Unknown command: {command}")

        return handler(request)

    def run(self) -> None:
        """Main loop: read JSON requests from stdin, write responses to stdout."""
        self._log("model initialized successfully")
        self._log("Service started, waiting for requests...")

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            request_id = None
            try:
                request = json.loads(line)
                request_id = request.get("id")

                # Handle shutdown command
                if request.get("command") == "shutdown":
                    self._log("Shutdown requested")
                    self._send_response({
                        "id": request_id,
                        "success": True,
                        "message": "Shutting down",
                    })
                    break

                # Process request
                response = self.handle_request(request)
                response["id"] = request_id
                self._send_response(response)

            except json.JSONDecodeError as e:
                self._send_error(request_id, f"Invalid JSON: {e}")
            except Exception as e:
                self._log(f"Error processing request: {e}")
                import traceback
                traceback.print_exc(file=sys.stderr)
                self._send_error(request_id, str(e))

        self._log("Service shutting down")


def _merge_configs(config_path, device: str = None):
    """Read one or more config files into a single block: a lone segmenter
    config pulls in its text-query sibling, relative model paths resolve
    against the config directories, and ``device`` overrides the SAM keys."""
    import kwiver.vital.config as vital_config

    if isinstance(config_path, (str, os.PathLike)):
        config_paths = [str(config_path)]
    else:
        config_paths = [str(p) for p in config_path]

    # Auto-discover a sibling interactive_text_query_default.conf when the
    # caller supplied only one config and that config has no text-query
    # backend. This keeps the segmenter and text-query config files
    # mutually independent (neither ``include``s the other) while still
    # letting "point the service at the segmenter default" bring up both
    # features when both backends are available. Add-ons override that
    # default sibling to point at their backend (e.g. the SAM3 add-on).
    if len(config_paths) == 1:
        probe = vital_config.read_config_file(config_paths[0])
        if not probe.has_value("perform_text_query:type"):
            sibling = Path(config_paths[0]).parent / "interactive_text_query_default.conf"
            if sibling.exists():
                config_paths.append(str(sibling))

    cfg = vital_config.read_config_file(config_paths[0])
    for extra in config_paths[1:]:
        cfg.merge_config(vital_config.read_config_file(extra))

    # Resolve relative paths in config values. Each key's directory is the
    # *first* config whose file set that key — we approximate by resolving
    # against each config dir in order and keeping the first match. This
    # matters when the segmenter and text query configs live in different
    # directories (e.g. user supplies one from an add-on and one from core).
    config_dirs = [Path(p).parent for p in config_paths]
    path_keys = ['checkpoint', 'model_config', 'grounding_model_id', 'cfg', 'weights']
    for key in cfg.available_values():
        for path_key in path_keys:
            if path_key in key:
                value = cfg.get_value(key)
                if value and not os.path.isabs(value) and not value.startswith('$'):
                    for d in config_dirs:
                        resolved = d / value
                        if resolved.exists():
                            cfg.set_value(key, str(resolved))
                            break

    # Override device setting if provided
    if device:
        # Try common device config keys used by SAM algorithms
        device_keys = [
            "segment_via_points:sam2:device",
            "segment_via_points:sam3:device",
            "perform_text_query:sam2:device",
            "perform_text_query:sam3:device",
        ]
        for key in device_keys:
            cfg.set_value(key, device)

    return cfg


def _text_query_algo_from(cfg):
    from kwiver.vital.algo import PerformTextQuery

    if not cfg.has_value("perform_text_query:type"):
        return None
    impl_name = cfg.get_value("perform_text_query:type")
    algo = PerformTextQuery.create(impl_name)
    algo.set_configuration(cfg.subblock("perform_text_query:" + impl_name))
    return algo


def load_text_query_algo_from_config(config_path, device: str = None):
    """Build only the perform_text_query algorithm, or None when no config
    (including the auto-discovered sibling) defines one. Lets a service that
    started before a text-query add-on was installed pick it up later."""
    return _text_query_algo_from(_merge_configs(config_path, device))


def load_algorithms_from_config(config_path, plugin_paths: List[str] = None, device: str = None):
    """
    Load and configure algorithms from one or more KWIVER config files.

    The segmenter and text query configurations are independent files; pass
    them both to get both algorithms. If a single config path is given and
    it does not define ``perform_text_query:type``, a sibling
    ``interactive_text_query_default.conf`` in the same directory is auto-
    loaded when present. Later configs in the list override keys from
    earlier ones.

    Args:
        config_path: Path to a config file, or a list of paths. Later
            entries merge on top of earlier ones.
        plugin_paths: Optional list of additional plugin paths to load
        device: Optional device override (cuda, cpu, auto)

    Returns:
        Tuple of (segment_via_points_algo, perform_text_query_algo, image_io_algo, service_config)
    """
    from kwiver.vital.algo import SegmentViaPoints, ImageIO
    from kwiver.vital.modules import modules as vital_modules

    # Load plugin modules
    vital_modules.load_known_modules()

    if plugin_paths:
        for path in plugin_paths:
            if os.path.isdir(path):
                vital_modules.load_module(path)

    cfg = _merge_configs(config_path, device)

    # Create segment_via_points algorithm
    segment_algo = None
    if cfg.has_value("segment_via_points:type"):
        impl_name = cfg.get_value("segment_via_points:type")
        segment_algo = SegmentViaPoints.create(impl_name)
        segment_algo.set_configuration(cfg.subblock("segment_via_points:" + impl_name))

    # Create perform_text_query algorithm (optional)
    text_query_algo = _text_query_algo_from(cfg)

    # Create image_io algorithm for loading images (optional but recommended)
    image_io_algo = None
    if cfg.has_value("image_io:type"):
        impl_name = cfg.get_value("image_io:type")
        image_io_algo = ImageIO.create(impl_name)
        image_io_algo.set_configuration(cfg.subblock("image_io:" + impl_name))

    # Extract service configuration
    service_config = {
        "hole_policy": cfg.get_value("service:hole_policy") if cfg.has_value("service:hole_policy") else "allow",
        "multipolygon_policy": cfg.get_value("service:multipolygon_policy") if cfg.has_value("service:multipolygon_policy") else "allow",
        "max_polygon_points": int(cfg.get_value("service:max_polygon_points")) if cfg.has_value("service:max_polygon_points") else 25,
        "adaptive_simplify": cfg.get_value("service:adaptive_simplify").lower() in ('true', '1', 'yes') if cfg.has_value("service:adaptive_simplify") else False,
        "max_polygon_points_limit": int(cfg.get_value("service:max_polygon_points_limit")) if cfg.has_value("service:max_polygon_points_limit") else 100,
    }

    return segment_algo, text_query_algo, image_io_algo, service_config


def find_viame_config(model_type: Optional[str] = None) -> Optional[str]:
    """
    Find the segmentation config file in the VIAME install.

    Args:
        model_type: 'sam2' or 'sam3' for that add-on's config. With none
            given, SAM2 is preferred for point segmentation (SAM3 may be
            installed only for text queries), then SAM3, then the default.

    Returns:
        Path to config file if found, None otherwise
    """
    viame_install = os.environ.get("VIAME_INSTALL")
    if not viame_install:
        return None

    pipelines_dir = Path(viame_install) / "configs" / "pipelines"

    # Map model type to interactive config file
    config_files = {
        "sam2": "interactive_segmenter_sam2.conf",
        "sam3": "interactive_segmenter_sam3.conf",
    }

    if model_type is None:
        candidates = [config_files["sam2"], config_files["sam3"],
                      "interactive_segmenter_default.conf"]
    else:
        candidates = [config_files.get(model_type)]

    for config_name in candidates:
        if config_name and (pipelines_dir / config_name).exists():
            return str(pipelines_dir / config_name)

    return None


def create_default_config(output_path: str, model_type: str = "sam2"):
    """
    Create a default config file for the segmentation service.

    This generates a config file that includes the shared VIAME segmenter
    config files (common_sam2_segmenter.conf or common_sam3_segmenter.conf).

    Args:
        output_path: Path to write the config file
        model_type: Type of model ('sam2' or 'sam3')
    """
    if model_type == "sam2":
        config = """# Interactive Segmentation Service Configuration
# This config file sets up the SegmentViaPoints algorithm using SAM2.
#
# Include the shared SAM2 segmenter config for model paths and defaults.
# Uncomment the include line if running from VIAME install:
# include common_sam2_segmenter.conf

# Image loader algorithm (uses KWIVER native image loading)
image_io:type = vxl

# Point-based segmentation algorithm
segment_via_points:type = sam2
segment_via_points:sam2:checkpoint =
segment_via_points:sam2:cfg = configs/sam2.1/sam2.1_hiera_b+.yaml
segment_via_points:sam2:device = cuda

# Service settings
service:hole_policy = allow
service:multipolygon_policy = allow
service:max_polygon_points = 25
service:max_polygon_points_limit = 100
service:adaptive_simplify = false
"""
    else:
        config = """# Interactive Segmentation Service Configuration
# This config file sets up both SegmentViaPoints and PerformTextQuery
# algorithms using SAM3.
#
# Include the shared SAM3 segmenter config for model paths and defaults.
# Uncomment the include line if running from VIAME install:
# include common_sam3_segmenter.conf

# Image loader algorithm (uses KWIVER native image loading)
image_io:type = vxl

# Point-based segmentation algorithm
segment_via_points:type = sam3
segment_via_points:sam3:checkpoint =
segment_via_points:sam3:model_config =
segment_via_points:sam3:device = cuda

# Text-based query algorithm (optional)
perform_text_query:type = sam3
perform_text_query:sam3:checkpoint =
perform_text_query:sam3:model_config =
perform_text_query:sam3:device = cuda
perform_text_query:sam3:detection_threshold = 0.3
perform_text_query:sam3:max_detections = 10

# Service settings
service:hole_policy = allow
service:multipolygon_policy = allow
service:max_polygon_points = 25
service:max_polygon_points_limit = 100
service:adaptive_simplify = false
"""

    with open(output_path, 'w') as f:
        f.write(config)

    print(f"Created default config: {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Segmentation Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use a config file
    python -m viame.core.interactive_segmentation --config /path/to/config.pipe

    # Generate a default config file
    python -m viame.core.interactive_segmentation --generate-config sam2.pipe --model sam2

    # With additional plugin paths
    python -m viame.core.interactive_segmentation --config config.pipe --plugin-path /path/to/plugins
        """
    )
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="Path to a KWIVER config file. May be specified more than once "
             "to compose independent files (e.g. segmenter + text query); "
             "keys in later files override earlier ones. If a single file "
             "without a perform_text_query section is given, a sibling "
             "interactive_text_query_default.conf is auto-loaded when "
             "present.",
    )
    parser.add_argument(
        "--generate-config",
        default=None,
        metavar="OUTPUT_PATH",
        help="Generate a default config file and exit",
    )
    parser.add_argument(
        "--model",
        default="sam2",
        choices=["sam2", "sam3"],
        help="Model type for generated config (default: sam2)",
    )
    parser.add_argument(
        "--plugin-path",
        action="append",
        default=[],
        help="Additional plugin paths to load (can be specified multiple times)",
    )
    parser.add_argument(
        "--viame-path",
        default=None,
        help="Path to VIAME install directory (sets VIAME_INSTALL env var)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on (cuda, cpu, auto)",
    )
    args = parser.parse_args()

    # Set VIAME_INSTALL from --viame-path if provided
    if args.viame_path:
        os.environ["VIAME_INSTALL"] = args.viame_path

    # Handle config generation
    if args.generate_config:
        create_default_config(args.generate_config, args.model)
        return

    # Auto-detect config if not provided
    if not args.config:
        auto = find_viame_config(args.model)
        if auto:
            args.config = [auto]
            print(f"[SegmentationService] Using auto-detected config: {auto}", file=sys.stderr)

    # Require config file
    if not args.config:
        parser.error("--config is required (or use --generate-config to create one)")

    for c in args.config:
        if not Path(c).exists():
            print(f"Error: Config file not found: {c}", file=sys.stderr)
            sys.exit(1)

    try:
        # Suppress stdout during initialization to prevent library warnings
        # from corrupting the JSON protocol
        with suppress_stdout():
            segment_algo, text_query_algo, image_io_algo, service_config = load_algorithms_from_config(
                args.config, args.plugin_path, args.device
            )

        if segment_algo is None:
            print("Error: No segment_via_points algorithm configured", file=sys.stderr)
            sys.exit(1)

        # Create and run service
        service = InteractiveSegmentationService(
            segment_via_points_algo=segment_algo,
            perform_text_query_algo=text_query_algo,
            image_io_algo=image_io_algo,
            plugin_paths=args.plugin_path,
            device=args.device,
            **service_config
        )

        # Eagerly warm up only the segmentation model so the first click is
        # fast. The text-query model is intentionally left to load lazily on
        # the first text_query request -- it is large, often a different/heavier
        # backend, and may be unused for the whole session.
        with suppress_stdout():
            if hasattr(segment_algo, "_ensure_model"):
                try:
                    segment_algo._ensure_model()
                except Exception as e:
                    print(f"[SegmentationService] Warning: segmenter warmup failed: {e}",
                          file=sys.stderr)

        service.run()

    except KeyboardInterrupt:
        print("[SegmentationService] Interrupted", file=sys.stderr)
    except Exception as e:
        print(f"[SegmentationService] Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
