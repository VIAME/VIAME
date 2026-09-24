#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved
# BSD 3-Clause License. See either the root top-level LICENSE file or
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

"""
Interactive Stereo Service

A persistent process that keeps stereo depth algorithms loaded and handles
disparity computation requests via stdin/stdout JSON protocol. Designed for
interactive stereo annotation where lines drawn on the left image are automatically
transferred to the right image using disparity mapping.

This service uses KWIVER vital algorithms configured via config files:
- ComputeStereoDepthMap: For stereo disparity/depth computation

Unlike SAM, this service proactively computes disparity maps when the user navigates
to a new frame, so the disparity is ready when they draw annotations.

Usage:
    python -m viame.core.interactive_stereo --config /path/to/config.pipe
    python -m viame.core.interactive_stereo --config /path/to/config.pipe --plugin-path /path/to/plugins

Protocol:
    Input (JSON per line on stdin):
    {
        "id": "unique-request-id",
        "command": "set_frame",
        "left_image_path": "/path/to/left.png",
        "right_image_path": "/path/to/right.png"
    }

    Output (JSON per line on stdout):
    {
        "id": "unique-request-id",
        "success": true,
        "message": "Disparity computation started"
    }

    Commands:
    - "enable": Load the algorithm and enable the service (requires calibration);
      an optional "config" path selects the stereo config to load
    - "disable": Unload the algorithm and disable the service
    - "set_frame": Start computing disparity for stereo pair (proactive)
    - "cancel": Cancel current disparity computation
    - "get_status": Get current status (enabled, computing, ready)
    - "transfer_line": Transfer a line from left to right image using disparity
    - "measure_curve": Measure a curved fish using rectified dense disparity
    - "transfer_points": Transfer points between stereo images (left to right by default)
    - "shutdown": Gracefully terminate the service
"""

import argparse
import json
import os
import sys
import threading
import queue
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import cv2

# Compiled C++ stereo measurement bindings. The stereo length/measurement math
# lives solely in viame::core::compute_stereo_measurement (no Python duplicate),
# so this module is a hard dependency.
from viame.core import _measurement as _cpp_measurement
from viame import image_kernels


class EpipolarTemplateMatcher:
    """
    Per-point stereo matching along epipolar curves using template matching.

    Mirrors the epipolar_template_matching method in measurement_utilities.cxx.
    Uses camera calibration to compute epipolar curves in unrectified images,
    then matches template patches from the source image along the curve in
    the target image using Normalized Cross-Correlation (NCC).

    When dino_top_k > 0 and the dino_matcher module is available, uses
    two-stage matching: DINOv2 selects the top-K semantically similar candidates,
    then NCC picks the precise match from that filtered set. This reduces false
    matches on repetitive textures.
    """

    def __init__(
        self,
        template_size=13,
        template_matching_threshold=0.5,
        epipolar_min_disparity=2.0,
        epipolar_max_disparity=300.0,
        epipolar_num_samples=5000,
        dino_model_name="dinov2_vitb14",
        dino_top_k=0,
        dino_weights_path="",
    ):
        self._template_size = template_size
        self._threshold = template_matching_threshold
        self._min_disparity = epipolar_min_disparity
        self._max_disparity = epipolar_max_disparity
        self._num_samples = epipolar_num_samples
        self._K_left = None
        self._K_left_inv = None
        self._K_right = None
        self._R = None
        self._T = None
        self._min_depth = 0.0
        self._max_depth = 0.0
        self._calibrated = False

        # DINO top-K + NCC two-stage matching
        self._dino_model_name = dino_model_name
        self._dino_top_k = dino_top_k
        self._dino_weights_path = dino_weights_path
        self._dino_matcher = None
        self._dino_available = False
        self._dino_images_set = False

        if dino_top_k > 0:
            self._init_dino()

    def _log(self, msg):
        print(f"[EpipolarMatcher] {msg}", file=sys.stderr, flush=True)

    def _init_dino(self):
        # A config that asks for DINO must not silently degrade to plain NCC.
        if self._dino_weights_path and not os.path.isfile(self._dino_weights_path):
            raise RuntimeError(
                f"DINO weights not found: {self._dino_weights_path} "
                "(is the DINO add-on installed?)")
        try:
            from viame.pytorch import dino_matcher
            self._dino_matcher = dino_matcher
            dino_matcher.init_matcher(
                model_name=self._dino_model_name, device="cuda", threshold=0.0,
                weights_path=self._dino_weights_path)
        except Exception as e:
            raise RuntimeError(f"DINO matcher failed to initialize: {e}") from e
        self._dino_available = True
        self._log(f"DINO matcher initialized: model={self._dino_model_name}, "
                  f"top_k={self._dino_top_k}")

    def set_images(self, left_bgr, right_bgr):
        """Set BGR images for DINO feature extraction (call when frame changes)."""
        if not self._dino_available:
            return
        try:
            self._dino_matcher.set_images(left_bgr, right_bgr)
            self._dino_images_set = True
        except Exception as e:
            self._log(f"DINO set_images failed: {e}")
            self._dino_images_set = False

    def load_calibration(self, filepath):
        """
        Load stereo calibration (K_left, K_right, R, T) from a file.

        Normalized on viame::core::read_stereo_rig (via the viame.core._measurement
        bindings) -- the same loader the measurement pipeline processes use --
        which supports .json, .yml/.yaml, .npz, .mat and OpenCV calibration
        directories.
        """
        self._log(f"Loading calibration from: {filepath}")

        cal = _cpp_measurement.load_stereo_calibration(filepath)
        self._K_left = np.array(cal['k_left'], dtype=np.float64).reshape(3, 3)
        self._K_right = np.array(cal['k_right'], dtype=np.float64).reshape(3, 3)
        self._R = np.array(cal['rotation'], dtype=np.float64).reshape(3, 3)
        self._T = np.array(cal['translation'], dtype=np.float64).flatten()

        self._K_left_inv = np.linalg.inv(self._K_left)

        # Convert disparity range to depth range: depth = fx * baseline / disparity
        fx_l = self._K_left[0, 0]
        baseline = np.linalg.norm(self._T)
        if (self._min_disparity > 0 and self._max_disparity > 0
                and fx_l > 0 and baseline > 0):
            self._min_depth = fx_l * baseline / self._max_disparity
            self._max_depth = fx_l * baseline / self._min_disparity
        else:
            self._min_depth = 1000.0
            self._max_depth = 100000.0

        self._calibrated = True

        self._log(f"Calibration loaded: fx_l={fx_l:.1f}, "
                  f"baseline={baseline:.4f}, "
                  f"depth_range=[{self._min_depth:.1f}, {self._max_depth:.1f}]")

    @property
    def calibrated(self):
        return self._calibrated

    def _compute_epipolar_points(self, source_point):
        """Compute epipolar curve in right image for a left image point."""
        pt_h = np.array([source_point[0], source_point[1], 1.0])
        normalized = self._K_left_inv @ pt_h
        ray_dir = normalized / np.linalg.norm(normalized)

        num = self._num_samples
        depth_step = (self._max_depth - self._min_depth) / max(num - 1, 1)

        points = []
        prev_px = prev_py = None

        for i in range(num):
            depth = self._min_depth + i * depth_step
            p3d = ray_dir * depth
            p3d_right = self._R @ p3d + self._T

            if p3d_right[2] <= 0:
                continue

            inv_z = 1.0 / p3d_right[2]
            px = self._K_right[0, 0] * p3d_right[0] * inv_z + self._K_right[0, 2]
            py = self._K_right[1, 1] * p3d_right[1] * inv_z + self._K_right[1, 2]

            ipx, ipy = int(round(px)), int(round(py))
            if ipx == prev_px and ipy == prev_py:
                continue
            prev_px, prev_py = ipx, ipy
            points.append((px, py))

        return points

    def match_point(self, left_gray, right_gray, source_point):
        """Find corresponding point in right image via epipolar template matching.

        When DINO top-K is enabled and available, first filters epipolar candidates
        by DINOv2 semantic similarity, then runs NCC on the filtered set.
        """
        if not self._calibrated:
            self._log(f"match_point({source_point}): not calibrated")
            return None

        half = self._template_size // 2
        x_src = int(round(source_point[0]))
        y_src = int(round(source_point[1]))

        h_l, w_l = left_gray.shape[:2]
        if (x_src < half or x_src >= w_l - half
                or y_src < half or y_src >= h_l - half):
            self._log(f"match_point({source_point}): source too close to edge "
                      f"(image {w_l}x{h_l}, margin={half})")
            return None

        template = left_gray[
            y_src - half:y_src + half + 1,
            x_src - half:x_src + half + 1
        ].astype(np.float32)

        epipolar_pts = self._compute_epipolar_points(source_point)
        if not epipolar_pts:
            self._log(f"match_point({source_point}): no epipolar points computed "
                      f"(depth_range=[{self._min_depth:.1f}, {self._max_depth:.1f}])")
            return None

        # DINO top-K filtering: reduce candidate set before NCC
        if self._dino_available and self._dino_images_set and self._dino_top_k > 0:
            epi_xs = [p[0] for p in epipolar_pts]
            epi_ys = [p[1] for p in epipolar_pts]
            try:
                topk_indices = self._dino_matcher.get_top_k_indices(
                    float(source_point[0]), float(source_point[1]),
                    epi_xs, epi_ys, k=self._dino_top_k)
                if topk_indices:
                    epipolar_pts = [epipolar_pts[i] for i in topk_indices]
            except Exception as e:
                self._log(f"DINO top-K failed ({e}), using full set")

        h_r, w_r = right_gray.shape[:2]
        best_score = -1.0
        best_point = None
        n_in_bounds = 0

        for ep_x, ep_y in epipolar_pts:
            x_tgt = int(round(ep_x))
            y_tgt = int(round(ep_y))

            if (x_tgt < half or x_tgt >= w_r - half
                    or y_tgt < half or y_tgt >= h_r - half):
                continue

            n_in_bounds += 1
            target_patch = right_gray[
                y_tgt - half:y_tgt + half + 1,
                x_tgt - half:x_tgt + half + 1
            ].astype(np.float32)

            result = cv2.matchTemplate(
                target_patch, template, cv2.TM_CCOEFF_NORMED)
            score = float(result[0, 0])

            if score > best_score:
                best_score = score
                best_point = (ep_x, ep_y)

        if best_score < self._threshold:
            ep_first = epipolar_pts[0] if epipolar_pts else None
            ep_last = epipolar_pts[-1] if epipolar_pts else None
            self._log(f"match_point({source_point}): best_score={best_score:.3f} "
                      f"< threshold={self._threshold}, "
                      f"epipolar_pts={len(epipolar_pts)}, "
                      f"in_bounds={n_in_bounds}, "
                      f"right_img={w_r}x{h_r}, "
                      f"ep_range=[{ep_first} .. {ep_last}]")
            return None

        return best_point

    def compute_measurement(self, left_p1, right_p1, left_p2, right_p2):
        """Full stereo measurement for a line: length, 3D midpoint (x, y, z),
        range (midpoint distance to the left camera) and RMS reprojection error.

        Computed by viame::core::compute_stereo_measurement via the
        viame.core._measurement bindings (single source of truth shared with
        the C++ measurement pipeline). Returns a dict in calibration units, or
        None if the matcher is not calibrated.
        """
        if not self._calibrated:
            return None

        # Pass plain Python lists (not numpy arrays): the C++ bindings take
        # flat std::vector<double> to avoid pybind11's numpy<->Eigen caster,
        # which is incompatible with numpy 2.0.
        return dict(
            _cpp_measurement.compute_stereo_measurement_from_calibration(
                np.asarray(self._K_left, dtype=np.float64).ravel().tolist(),
                np.asarray(self._K_right, dtype=np.float64).ravel().tolist(),
                np.asarray(self._R, dtype=np.float64).ravel().tolist(),
                np.asarray(self._T, dtype=np.float64).ravel().tolist(),
                [float(left_p1[0]), float(left_p1[1])],
                [float(right_p1[0]), float(right_p1[1])],
                [float(left_p2[0]), float(left_p2[1])],
                [float(right_p2[0]), float(right_p2[1])]))


# compute_measurements keys the dense grid honours, with the service defaults.
# alpha=0 crops to the region valid in both images, which collapses to a
# sliver when the baseline has a large vertical component. Keypoints sit on
# the object's silhouette, where the network blends the object with what lies
# behind it; a high percentile of the neighbourhood keeps the nearer surface.
DENSE_GRID_DEFAULTS = {
    "rectification_alpha": "-1.0",
    "refine_keypoints_disparity_window": "3",
    "refine_keypoints_disparity_percentile": "0.9",
    "refine_keypoints_disparity_min_valid_fraction": "0.0",
    "refine_keypoints_disparity_use_circle": "false",
    "refine_disparity_segment": "true",
    "disparity_segment_samples": "11",
    "disparity_segment_max_outliers": "3",
    "disparity_segment_max_error": "10.0",
}


class DenseStereoRectifier:
    """The rectified grid a dense disparity backend works on, built by the same
    C++ (map_keypoints_to_camera) the measurement pipelines use, so
    interactive results match batch results. Sized lazily from the first
    frame."""

    def __init__(self, calibration_path: str, options: Optional[Dict[str, str]] = None):
        self._path = calibration_path
        self._options = dict(DENSE_GRID_DEFAULTS, **(options or {}))
        self._grid = None
        self._size = None

    def prepare(self, width: int, height: int) -> None:
        if self._size == (width, height):
            return
        self._grid = _cpp_measurement.DenseStereoGrid(self._path, width, height, self._options)
        self._size = (width, height)

    @property
    def ready(self) -> bool:
        return self._grid is not None

    @property
    def segment_fit(self) -> bool:
        return self._options["refine_disparity_segment"].strip().lower() in ("true", "1", "yes", "on")

    def intrinsics(self) -> Dict[str, float]:
        return self._grid.intrinsics()

    def rectify_image(self, image: np.ndarray, right: bool) -> np.ndarray:
        return self._grid.rectify_image(np.ascontiguousarray(image, dtype=np.uint8), right)

    def rectify_points(self, points, right: bool) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return self._grid.rectify_points(pts, right) if len(pts) else pts

    def unrectify_points(self, points, right: bool) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return self._grid.unrectify_points(pts, right) if len(pts) else pts

    def match_grid_points(self, disparity: np.ndarray, grid_points) -> np.ndarray:
        """Right-grid matches of left-grid points; NaN where none is valid."""
        pts = np.asarray(grid_points, dtype=np.float64).reshape(-1, 2)
        if not len(pts):
            return pts
        return self._grid.match_grid_points(np.ascontiguousarray(disparity, dtype=np.float32), pts)

    def fit_segment(self, disparity: np.ndarray, left_head, left_tail):
        """Right endpoints (original coordinates) from the disparity profile
        along the segment, or None when the fit is rejected."""
        return self._grid.fit_segment(
            np.ascontiguousarray(disparity, dtype=np.float32),
            [float(left_head[0]), float(left_head[1])],
            [float(left_tail[0]), float(left_tail[1])])


class InteractiveStereoService:
    """
    Interactive Stereo Service using KWIVER vital algorithms.

    Handles stdin/stdout JSON protocol communication and delegates
    to configured vital algorithms for disparity computation.
    """

    def __init__(
        self,
        compute_stereo_depth_map_algo=None,
        dense_grid_options: Optional[Dict[str, str]] = None,
        epipolar_matcher: Optional[EpipolarTemplateMatcher] = None,
        scale: float = 1.0,
        segmentation_generate_line: bool = False,
        segmentation_point_sampling: bool = False,
        segmentation_point_samples: int = 5,
        max_transfer_size_ratio: float = 2.5,
        send_response=None,
    ):
        """
        Initialize the service with configured algorithms.

        Args:
            compute_stereo_depth_map_algo: Configured ComputeStereoDepthMap algorithm instance
                (for dense disparity mode). Mutually exclusive with epipolar_matcher.
            epipolar_matcher: Configured EpipolarTemplateMatcher instance
                (for per-point epipolar template matching mode).
            scale: Scale factor for input images (<=1.0). Lower = faster but less accurate.
            segmentation_generate_line: When True, the stereo point segmentation
                flow derives a head/tail line from the polygon on each camera and
                generates the length measurement.
            segmentation_point_sampling: When True, the other camera's segmentation
                seeds are segmentation_point_samples warped points
                sampled inside the source polygon (noise reduction).
            segmentation_point_samples: number of points to sample when
                segmentation_point_sampling is enabled.
            max_transfer_size_ratio: a shape whose area on the other camera is
                more than this many times larger or smaller than the original
                is refused instead of mapped; 0 disables the check.
        """
        self._stereo_algo = compute_stereo_depth_map_algo
        self._dense_grid_options = dense_grid_options or {}
        self._epipolar_matcher = epipolar_matcher
        self._use_epipolar = epipolar_matcher is not None
        self._scale = scale
        # Optional writer for outgoing JSON (sync responses, deferred transfers,
        # and async disparity events). When hosted by the unified interactive
        # service this routes through its single lock-guarded stdout writer;
        # otherwise it falls back to printing directly (standalone use).
        self._send_response_cb = send_response

        self._seg_generate_line = bool(segmentation_generate_line)
        self._seg_point_sampling = bool(segmentation_point_sampling)
        self._seg_point_samples = max(1, int(segmentation_point_samples))
        self._max_transfer_size_ratio = float(max_transfer_size_ratio)

        self._enabled = False

        # Calibration parameters
        self._calibration = None
        self._focal_length = 0.0
        self._baseline = 0.0
        self._principal_x = 0.0
        self._principal_y = 0.0

        # Current frame state
        self._current_left_path: Optional[str] = None
        self._current_right_path: Optional[str] = None
        self._current_disparity: Optional[np.ndarray] = None
        self._current_frame_time = None
        # Dense backends see rectified images; None keeps the raw grid.
        self._rectifier: Optional[DenseStereoRectifier] = None
        self._disparity_ready = False

        # Images for epipolar template matching mode
        self._left_gray: Optional[np.ndarray] = None
        self._right_gray: Optional[np.ndarray] = None

        # Disparity cache - stores last N computed disparities
        # Key: (left_path, right_path), Value: disparity array
        self._disparity_cache: Dict[Tuple[str, str], np.ndarray] = {}
        self._cache_order: List[Tuple[str, str]] = []  # Track insertion order for LRU
        self._max_cache_size = 4  # Keep last 4 disparity maps

        # Background computation
        self._compute_thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()
        self._disparity_event = threading.Event()
        self._compute_lock = threading.Lock()
        self._compute_queue = queue.Queue()

    def _log(self, message: str) -> None:
        """Log to stderr (stdout is reserved for JSON responses)."""
        print(f"[InteractiveStereo] {message}", file=sys.stderr, flush=True)

    def _send_response(self, response: Dict[str, Any]) -> None:
        """Send JSON response to stdout (or via the injected writer)."""
        if self._send_response_cb is not None:
            self._send_response_cb(response)
        else:
            print(json.dumps(response), flush=True)

    def _send_error(self, request_id: Optional[str], error: str) -> None:
        """Send error response."""
        self._send_response({
            "id": request_id,
            "success": False,
            "error": error,
        })

    def _add_to_cache(self, left_path: str, right_path: str, disparity: np.ndarray,
                      frame_time=None) -> None:
        """Add a disparity map to the cache with LRU eviction."""
        cache_key = (left_path, right_path, frame_time)

        # If already in cache, move to end of order list
        if cache_key in self._disparity_cache:
            self._cache_order.remove(cache_key)
            self._cache_order.append(cache_key)
            return

        # Evict oldest if at capacity
        while len(self._cache_order) >= self._max_cache_size:
            oldest_key = self._cache_order.pop(0)
            if oldest_key in self._disparity_cache:
                del self._disparity_cache[oldest_key]
                self._log(f"Evicted disparity from cache: {oldest_key[0]}")

        # Add to cache
        self._disparity_cache[cache_key] = disparity
        self._cache_order.append(cache_key)
        self._log(f"Added disparity to cache. Cache size: {len(self._cache_order)}")

    def _get_from_cache(self, left_path: str, right_path: str,
                        frame_time=None) -> Optional[np.ndarray]:
        """Get a disparity map from the cache if available."""
        cache_key = (left_path, right_path, frame_time)
        disparity = self._disparity_cache.get(cache_key)
        if disparity is not None:
            # Move to end of order list (most recently used)
            self._cache_order.remove(cache_key)
            self._cache_order.append(cache_key)
            self._log(f"Cache hit for: {left_path}")
        return disparity

    def _load_calibration(self, calibration_data: Dict[str, Any]) -> None:
        """Load stereo calibration from JSON data."""
        self._calibration = calibration_data

        # Extract focal length from left camera
        self._focal_length = float(calibration_data.get('fx_left', 0.0))
        self._principal_x = float(calibration_data.get('cx_left', 0.0))
        self._principal_y = float(calibration_data.get('cy_left', 0.0))

        # Compute baseline from translation vector
        T = calibration_data.get('T', [0.0, 0.0, 0.0])
        if isinstance(T, list) and len(T) >= 3:
            self._baseline = abs(T[0])
            if self._baseline < 1e-6:
                self._baseline = np.sqrt(T[0]**2 + T[1]**2 + T[2]**2)
        else:
            self._baseline = 0.0

        self._log(f"Loaded calibration: focal_length={self._focal_length}, "
                  f"baseline={self._baseline}, principal=({self._principal_x}, {self._principal_y})")

    # ------------------------------------------------ dense grid mapping
    def _to_grid(self, points, right: bool = False) -> np.ndarray:
        """Original image coordinates -> the disparity grid (rectified when
        a calibration file is loaded, otherwise the raw image)."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if self._rectifier is not None and self._rectifier.ready:
            return self._rectifier.rectify_points(pts, right)
        return pts

    def _from_grid(self, points, right: bool = True) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if self._rectifier is not None and self._rectifier.ready:
            return self._rectifier.unrectify_points(pts, right)
        return pts

    # Keypoints sit on the object's silhouette, where the network blends the
    # object with what lies behind it. The object is the nearer surface, so a
    # high percentile of the neighbourhood recovers its disparity.
    _DISPARITY_WINDOW = 3
    _DISPARITY_PERCENTILE = 90

    def _grid_disparity(self, grid_points, disparity=None) -> np.ndarray:
        """Disparity at grid coordinates (clamped to the grid); 0 where the
        neighbourhood holds no valid value."""
        disparity = self._current_disparity if disparity is None else disparity
        h, w = disparity.shape[:2]
        pts = np.asarray(grid_points, dtype=np.float64).reshape(-1, 2)
        xs = np.clip(np.rint(pts[:, 0]), 0, w - 1).astype(int)
        ys = np.clip(np.rint(pts[:, 1]), 0, h - 1).astype(int)
        out = np.zeros(len(pts))
        r = self._DISPARITY_WINDOW
        for i, (x, y) in enumerate(zip(xs, ys)):
            window = disparity[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
            valid = window[np.isfinite(window) & (window > 0)]
            if valid.size:
                out[i] = float(np.percentile(valid, self._DISPARITY_PERCENTILE))
        return out

    def _match_grid(self, grid_points, disparity=None, right_to_left: bool = False):
        """Grid matches on the other camera and their disparities (0 where
        the neighbourhood holds no valid value)."""
        disparity = self._current_disparity if disparity is None else disparity
        grid = np.asarray(grid_points, dtype=np.float64).reshape(-1, 2)
        if self._rectifier is not None and self._rectifier.ready:
            matched = self._rectifier.match_grid_points(disparity, grid)
            disp = np.where(np.isfinite(matched[:, 0]), grid[:, 0] - matched[:, 0], 0.0)
        else:
            disp = self._grid_disparity(grid, disparity)
        sign = 1.0 if right_to_left else -1.0
        return grid + np.column_stack([sign * disp, np.zeros_like(disp)]), disp

    def _dense_transfer(self, points, right_to_left: bool = False):
        """Corresponding points on the other camera via the disparity grid.
        Returns (matched original coordinates, disparities)."""
        grid = self._to_grid(points, right=right_to_left)
        disparity = self._right_reference_disparity() if right_to_left else None
        matched, disp = self._match_grid(grid, disparity, right_to_left)
        return self._from_grid(matched, right=not right_to_left), disp

    def _grid_calibration(self):
        """(fx, fy, cx_left, cx_right, cy, baseline) of the disparity grid."""
        rect = self._rectifier
        if rect is not None and rect.ready:
            i = rect.intrinsics()
            return (i["fx"], i["fy"], i["cx_left"], i["cx_right"], i["cy"], i["baseline"])
        if self._focal_length <= 0 or self._baseline <= 0:
            return None
        fx = float(self._focal_length)
        return (fx, fx, float(self._principal_x), float(self._principal_x),
                float(self._principal_y), float(self._baseline))

    def _dense_measurement(self, lp1, rp1, lp2, rp2):
        """Full stereo measurement (dense mode) via the C++ implementation.

        The endpoints (original image coordinates, corresponded on both
        cameras) are moved onto the rectified grid, where the rig is an
        idealized pair (R == I, T == [-baseline, 0, 0]) with the intrinsics of
        the rectifying projections. Without a calibration file the raw image is
        the grid and the scalar calibration stands in.
        """
        cal = self._grid_calibration()
        if cal is None:
            return None
        fx, fy, cx_l, cx_r, cy, baseline = cal
        left = self._to_grid([lp1, lp2], right=False)
        right = self._to_grid([rp1, rp2], right=True)
        if (left[:, 0] - right[:, 0] <= 0).any():
            return None
        # Flat row-major lists (see compute_measurement for why lists, not numpy)
        k_left = [fx, 0.0, cx_l, 0.0, fy, cy, 0.0, 0.0, 1.0]
        k_right = [fx, 0.0, cx_r, 0.0, fy, cy, 0.0, 0.0, 1.0]
        rotation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        translation = [-baseline, 0.0, 0.0]

        return dict(
            _cpp_measurement.compute_stereo_measurement_from_calibration(
                k_left, k_right, rotation, translation,
                left[0].tolist(), right[0].tolist(),
                left[1].tolist(), right[1].tolist()))

    _VIDEO_EXTENSIONS = {'.avi', '.mp4', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpg', '.mpeg', '.m4v'}

    def _is_video_file(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in self._VIDEO_EXTENSIONS

    def _load_image(self, image_path: str, frame_time: float = None):
        """Load an image (or video frame at given time) and return a vital ImageContainer."""
        if self._is_video_file(image_path) and frame_time is not None:
            return self._load_video_frame(image_path, frame_time)

        from viame.types import ImageContainer, Image
        from viame.segmentation.segmentation_utils import load_image

        imdata = load_image(image_path)
        return ImageContainer(Image(imdata))

    def _load_video_frame(self, video_path: str, frame_time: float):
        """Extract a single frame from a video at the given time (seconds)."""
        from viame.algo import VideoInput
        from viame.types import Timestamp

        # Cache video readers keyed by path (may have left + right videos)
        if not hasattr(self, '_video_readers'):
            self._video_readers = {}

        if video_path not in self._video_readers:
            vi = VideoInput.create("vidl_ffmpeg")
            cfg = vi.get_configuration()
            cfg.set_value("time_source", "start_at_0")
            vi.set_configuration(cfg)
            vi.open(video_path)
            # Read first frame to get FPS
            ts = Timestamp()
            vi.next_frame(ts, 0)
            fps = vi.frame_rate()
            self._video_readers[video_path] = (vi, fps)

        vi, fps = self._video_readers[video_path]
        target_frame = round(frame_time * fps) + 1  # 1-based ffmpeg numbering
        target_frame = max(1, target_frame)

        ts = Timestamp()
        vi.seek_frame(ts, target_frame, 0)
        image = vi.frame_image()

        if image is None:
            raise RuntimeError(f"Could not read frame at t={frame_time:.3f}s from {video_path}")

        return image

    def _compute_disparity_sync(
        self,
        left_path: str,
        right_path: str,
        frame_time: float = None,
    ) -> Optional[np.ndarray]:
        """
        Compute disparity for stereo pair synchronously using the configured algorithm.
        Returns disparity map or None if cancelled.
        """
        if self._cancel_event.is_set():
            return None

        # Load images as ImageContainers (with video support)
        left_container = self._load_image(left_path, frame_time)
        right_container = self._load_image(right_path, frame_time)

        if left_container is None or right_container is None:
            raise RuntimeError(
                f"Failed to load stereo images: left={left_path}, right={right_path}"
            )

        left_img = left_container.image()
        right_img = right_container.image()
        left_size = (left_img.width(), left_img.height())
        right_size = (right_img.width(), right_img.height())

        if left_size != right_size:
            raise RuntimeError(
                f"Left/right image size mismatch: left={left_size}, right={right_size} "
                f"(left_path={left_path}, right_path={right_path})"
            )

        if self._cancel_event.is_set():
            return None

        if self._rectifier is not None:
            from viame.types import Image, ImageContainer
            self._rectifier.prepare(*left_size)
            intrinsics = self._rectifier.intrinsics()
            self._focal_length = intrinsics["fx"]
            self._principal_x = intrinsics["cx_left"]
            self._principal_y = intrinsics["cy"]
            self._baseline = intrinsics["baseline"]
            left_container = ImageContainer(Image(
                self._rectifier.rectify_image(left_img.asarray(), False)))
            right_container = ImageContainer(Image(
                self._rectifier.rectify_image(right_img.asarray(), True)))

        # Call the algorithm's compute method
        result_container = self._stereo_algo.compute(left_container, right_container)

        if result_container is None:
            raise RuntimeError(
                "Stereo algorithm returned None — check algorithm configuration "
                f"(input sizes: {left_size})"
            )

        if self._cancel_event.is_set():
            return None

        # Convert result to numpy array
        result_image = result_container.image()
        disp_npy = result_image.asarray()

        # The algorithm returns disparity scaled by 256 as uint16
        # Convert back to float disparity values
        if disp_npy.dtype == np.uint16:
            disp_npy = disp_npy.astype(np.float32) / 256.0

        # Ensure 2D (H, W) — KWIVER images may be (H, W, 1) for single-channel
        if disp_npy.ndim == 3 and disp_npy.shape[2] == 1:
            disp_npy = disp_npy[:, :, 0]

        return disp_npy

    def _background_compute_worker(self) -> None:
        """Background thread that processes disparity computation requests."""
        while True:
            try:
                task = self._compute_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if task is None:
                break

            request_id, left_path, right_path, frame_time = task

            try:
                self._cancel_event.clear()
                self._log(f"Starting disparity computation for {left_path}")

                disparity = self._compute_disparity_sync(left_path, right_path, frame_time)

                if disparity is not None and not self._cancel_event.is_set():
                    with self._compute_lock:
                        # Add to cache for future use
                        self._add_to_cache(left_path, right_path, disparity, frame_time)

                        # Only update if this is still the current frame
                        if (self._current_left_path == left_path and
                                self._current_right_path == right_path and
                                self._current_frame_time == frame_time):
                            self._current_disparity = disparity
                            self._disparity_ready = True
                            self._disparity_event.set()
                            self._log(f"Disparity ready for {left_path}")
                            # Send async notification
                            self._send_response({
                                "id": request_id,
                                "type": "disparity_ready",
                                "success": True,
                                "left_path": left_path,
                            })
                else:
                    self._log("Disparity computation cancelled")

            except Exception as e:
                self._log(f"Error computing disparity: {e}")
                import traceback
                traceback.print_exc(file=sys.stderr)
                self._send_response({
                    "id": request_id,
                    "type": "disparity_error",
                    "success": False,
                    "error": str(e),
                })

            self._compute_queue.task_done()

    def _start_background_worker(self) -> None:
        """Start the background computation worker thread."""
        if self._compute_thread is None or not self._compute_thread.is_alive():
            self._compute_thread = threading.Thread(
                target=self._background_compute_worker,
                daemon=True
            )
            self._compute_thread.start()

    def _cancel_computation(self) -> None:
        """Cancel any ongoing disparity computation."""
        self._cancel_event.set()
        # Clear the queue
        while not self._compute_queue.empty():
            try:
                self._compute_queue.get_nowait()
                self._compute_queue.task_done()
            except queue.Empty:
                break

    def handle_enable(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Enable the service."""
        if self._enabled:
            return {
                "success": True,
                "message": "Already enabled",
            }

        # Load calibration from file path (used by epipolar template matching)
        calibration_file = request.get("calibration_file")
        if calibration_file and self._epipolar_matcher is not None:
            self._epipolar_matcher.load_calibration(calibration_file)

        # Load calibration from JSON data (used by dense disparity mode)
        calibration = request.get("calibration")
        if calibration:
            self._load_calibration(calibration)
        if calibration_file and not self._use_epipolar:
            self._rectifier = DenseStereoRectifier(calibration_file, self._dense_grid_options)
            self._calibration = self._calibration or {"file": calibration_file}
            self._log(f"Dense stereo will rectify with {calibration_file}")

        self._enabled = True
        self._start_background_worker()
        self._log("Interactive stereo service enabled")

        return {
            "success": True,
            "message": "Interactive stereo enabled",
        }

    def handle_disable(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Disable the service."""
        if not self._enabled:
            return {
                "success": True,
                "message": "Already disabled",
            }

        self._cancel_computation()
        self._current_disparity = None
        self._right_reference_key = None
        self._right_reference_map = None
        self._disparity_ready = False
        self._disparity_event.clear()
        self._current_left_path = None
        self._current_right_path = None
        self._enabled = False

        self._log("Interactive stereo service disabled")
        return {
            "success": True,
            "message": "Interactive stereo disabled",
        }

    def handle_set_calibration(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Update calibration parameters."""
        calibration = request.get("calibration")
        if not calibration:
            raise ValueError("calibration is required")

        self._load_calibration(calibration)
        return {
            "success": True,
            "message": "Calibration updated",
        }

    def handle_set_frame(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set the current frame and start computing disparity proactively.
        Cancels any previous computation. Checks cache first.
        """
        if not self._enabled:
            raise ValueError("Service not enabled. Call enable first.")

        left_path = request.get("left_image_path")
        right_path = request.get("right_image_path")
        request_id = request.get("id")
        frame_time = request.get("frame_time")

        if not left_path or not right_path:
            raise ValueError("left_image_path and right_image_path are required")

        if not os.path.exists(left_path):
            raise ValueError(f"Left image not found: {left_path}")
        if not os.path.exists(right_path):
            raise ValueError(f"Right image not found: {right_path}")

        # Check if already computing this frame (video frames share a path)
        with self._compute_lock:
            if (self._current_left_path == left_path and
                    self._current_right_path == right_path and
                    self._current_frame_time == frame_time):
                if self._disparity_ready:
                    return {
                        "success": True,
                        "message": "Disparity already computed",
                        "disparity_ready": True,
                    }
                else:
                    return {
                        "success": True,
                        "message": "Disparity computation already in progress",
                        "disparity_ready": False,
                    }

            self._current_frame_time = frame_time
            if self._use_epipolar:
                # Epipolar mode: update state, will load images below
                self._cancel_computation()
                self._current_left_path = left_path
                self._current_right_path = right_path
            else:
                # Dense mode: check if we have this disparity cached
                cached_disparity = self._get_from_cache(left_path, right_path, frame_time)
                if cached_disparity is not None:
                    self._log(f"Using cached disparity for: {left_path}")
                    self._current_left_path = left_path
                    self._current_right_path = right_path
                    self._current_disparity = cached_disparity
                    self._disparity_ready = True
                    self._disparity_event.set()
                    return {
                        "success": True,
                        "message": "Disparity loaded from cache",
                        "disparity_ready": True,
                    }

                # Cancel current computation and start new one
                self._cancel_computation()
                self._current_left_path = left_path
                self._current_right_path = right_path
                self._current_disparity = None
                self._disparity_ready = False
                self._disparity_event.clear()

        if self._use_epipolar:
            # Load images for template matching (outside lock - I/O bound)
            # Use _load_image for video support, then convert to OpenCV
            if self._is_video_file(left_path) and self._current_frame_time is not None:
                left_container = self._load_image(left_path, self._current_frame_time)
                right_container = self._load_image(right_path, self._current_frame_time)
                left_arr = np.array(left_container.image().asarray())
                right_arr = np.array(right_container.image().asarray())
                # Convert RGB to grayscale
                left_gray = image_kernels.to_gray(left_arr) if left_arr.ndim == 3 else left_arr
                right_gray = image_kernels.to_gray(right_arr) if right_arr.ndim == 3 else right_arr
                # BGR for DINO
                if self._epipolar_matcher._dino_available:
                    left_bgr = image_kernels.swap_channels(left_arr) if left_arr.ndim == 3 else left_arr
                    right_bgr = image_kernels.swap_channels(right_arr) if right_arr.ndim == 3 else right_arr
                    self._epipolar_matcher.set_images(left_bgr, right_bgr)
            else:
                left_gray = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
                right_gray = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
                if left_gray is None or right_gray is None:
                    raise ValueError(
                        f"Failed to load images: left={left_path}, right={right_path}")

                # Load BGR images for DINO feature extraction if enabled
                if self._epipolar_matcher._dino_available:
                    left_bgr = cv2.imread(left_path, cv2.IMREAD_COLOR)
                    right_bgr = cv2.imread(right_path, cv2.IMREAD_COLOR)
                    if left_bgr is not None and right_bgr is not None:
                        self._epipolar_matcher.set_images(left_bgr, right_bgr)

            with self._compute_lock:
                self._left_gray = left_gray
                self._right_gray = right_gray
                self._disparity_ready = True
                self._disparity_event.set()

            self._log(f"Images loaded for template matching: {left_path}")
            return {
                "success": True,
                "message": "Images loaded for template matching",
                "disparity_ready": True,
            }

        # Queue the new dense disparity computation
        self._compute_queue.put((request_id, left_path, right_path, self._current_frame_time))

        return {
            "success": True,
            "message": "Disparity computation started",
            "disparity_ready": False,
        }

    def handle_cancel(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Cancel current disparity computation."""
        self._cancel_computation()
        return {
            "success": True,
            "message": "Computation cancelled",
        }

    def handle_get_status(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get current service status."""
        with self._compute_lock:
            return {
                "success": True,
                "enabled": self._enabled,
                "disparity_ready": self._disparity_ready,
                "current_left_path": self._current_left_path,
                "current_right_path": self._current_right_path,
                "has_calibration": self._calibration is not None,
            }

    def _do_transfer_line(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute line transfer with lock already held or disparity known ready."""
        with self._compute_lock:
            line = request.get("line")
            if not line or len(line) != 2:
                raise ValueError("line must be a list of two [x, y] points")

            p1 = line[0]
            p2 = line[1]

            if self._use_epipolar:
                right_p1 = self._epipolar_matcher.match_point(
                    self._left_gray, self._right_gray, p1)
                right_p2 = self._epipolar_matcher.match_point(
                    self._left_gray, self._right_gray, p2)

                if right_p1 is None or right_p2 is None:
                    return {
                        "success": False,
                        "error": "Template matching failed for one or both line endpoints",
                    }

                result = {
                    "success": True,
                    "transferred_line": [
                        [float(right_p1[0]), float(right_p1[1])],
                        [float(right_p2[0]), float(right_p2[1])],
                    ],
                    "original_line": line,
                }

                # Triangulate the endpoints for the full stereo measurement
                # (length, 3D midpoint, range, RMS).
                measurement = self._epipolar_matcher.compute_measurement(
                    p1, right_p1, p2, right_p2)
                if measurement is not None:
                    result["length"] = measurement["length"]
                    result["measurement"] = measurement

                return result

            matched, disp = self._dense_transfer([p1, p2])
            if self._rectifier is not None and self._rectifier.ready and self._rectifier.segment_fit:
                # Fit the disparity profile along the body rather than trusting
                # two edge pixels; fall back to the per-point matches.
                fitted = self._rectifier.fit_segment(self._current_disparity, p1, p2)
                if fitted is not None:
                    matched = np.asarray(fitted, dtype=float)
                    grid_left = self._to_grid([p1, p2])
                    grid_right = self._to_grid(matched, right=True)
                    disp = grid_left[:, 0] - grid_right[:, 0]
            disp1, disp2 = float(disp[0]), float(disp[1])
            transferred_line = matched.tolist()

            depth_info = None
            measurement = None
            if self._focal_length > 0 and self._baseline > 0:
                depth1 = (self._focal_length * self._baseline) / max(disp1, 1e-6) if disp1 > 0 else None
                depth2 = (self._focal_length * self._baseline) / max(disp2, 1e-6) if disp2 > 0 else None
                depth_info = {
                    "depth_point1": depth1,
                    "depth_point2": depth2,
                    "disparity_point1": disp1,
                    "disparity_point2": disp2,
                }
                # Triangulate via the rectified pinhole model for the full
                # stereo measurement (length, 3D midpoint, range).
                measurement = self._dense_measurement(p1, matched[0], p2, matched[1])

            result = {
                "success": True,
                "transferred_line": transferred_line,
                "original_line": line,
                "depth_info": depth_info,
            }
            if measurement is not None:
                result["length"] = measurement["length"]
                result["measurement"] = measurement
            return result

    def size_mismatch(self, source_area: float, mapped_area: float) -> Optional[str]:
        """Why a shape mapped to the other camera is refused for its size, or None."""
        limit = self._max_transfer_size_ratio
        if limit <= 0 or source_area <= 0 or mapped_area <= 0:
            return None
        ratio = mapped_area / source_area
        if 1.0 / limit <= ratio <= limit:
            return None
        self._log(f"Mapped shape is {ratio:.2g}x the original area (limit {limit:.2g}x); not mapped")
        return "Failed to map to the other camera"

    @staticmethod
    def _hull_area(points) -> float:
        """Convex hull area of a point set, 0 when it is close to a line."""
        import cv2
        pts = np.asarray(points, dtype=np.float32)
        if len(pts) < 3:
            return 0.0
        area = float(cv2.contourArea(cv2.convexHull(pts)))
        extent = pts.max(axis=0) - pts.min(axis=0)
        return area if area >= 0.03 * float(extent @ extent) else 0.0

    def _do_transfer_points(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Point transfer that refuses a point set whose size changes too much."""
        response = self._transfer_points(request)
        mapped = response.get("transferred_points") or []
        if response.get("success") and all(p is not None for p in mapped):
            reason = self.size_mismatch(
                self._hull_area(request["points"]), self._hull_area(mapped))
            if reason:
                return {"success": False, "error": reason, "size_mismatch": True}
        return response

    def _transfer_points(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute points transfer with lock already held or disparity known ready."""
        with self._compute_lock:
            points = request.get("points")
            if not points:
                raise ValueError("points is required")

            if request.get('strict') or request.get('source_camera', 'left') != 'left':
                return self._transfer_points_checked(request)

            if self._use_epipolar:
                transferred_points = []
                disparity_values = []
                num_matched = 0

                for p in points:
                    matched = self._epipolar_matcher.match_point(
                        self._left_gray, self._right_gray, p)
                    if matched is not None:
                        transferred_points.append(
                            [float(matched[0]), float(matched[1])])
                        disparity_values.append(float(p[0] - matched[0]))
                        num_matched += 1
                    else:
                        # No epipolar match: fall back to the source coordinate
                        # so plain point transfer still returns something, but
                        # record that this point did not actually match so
                        # callers (e.g. stereo segmentation) can fail loudly.
                        transferred_points.append([float(p[0]), float(p[1])])
                        disparity_values.append(0.0)

                return {
                    "success": True,
                    "transferred_points": transferred_points,
                    "original_points": points,
                    "disparity_values": disparity_values,
                    "num_matched": num_matched,
                }

            matched, disp = self._dense_transfer(points)
            return {
                "success": True,
                "transferred_points": matched.tolist(),
                "original_points": points,
                "disparity_values": disp.tolist(),
            }

    def _grid_images(self, left_path, right_path, frame_time):
        """The stereo pair as the dense backend sees it (rectified if possible)."""
        left = self._load_image(left_path, frame_time).image().asarray()
        right = self._load_image(right_path, frame_time).image().asarray()
        if self._rectifier is not None:
            self._rectifier.prepare(left.shape[1], left.shape[0])
            left = self._rectifier.rectify_image(left, False)
            right = self._rectifier.rectify_image(right, True)
        return left, right

    def _right_reference_disparity(self):
        """Cache one reverse map per stereo frame; caller holds _compute_lock."""
        from viame.types import Image, ImageContainer
        key = (self._current_left_path, self._current_right_path,
               getattr(self, '_current_frame_time', None))
        if getattr(self, '_right_reference_key', None) == key:
            return self._right_reference_map
        config = self._stereo_algo.get_configuration()
        if config.has_value('output_mode') and config.get_value('output_mode') != 'disparity':
            raise ValueError('Point transfer requires disparity output')
        left, right = self._grid_images(key[0], key[1], key[2])
        result = self._stereo_algo.compute(
            ImageContainer(Image(np.ascontiguousarray(right[:, ::-1]))),
            ImageContainer(Image(np.ascontiguousarray(left[:, ::-1]))))
        if result is None:
            raise ValueError('Reverse disparity computation failed')
        disparity = result.image().asarray()
        if disparity.ndim == 3 and disparity.shape[2] == 1:
            disparity = disparity[:, :, 0]
        scale = 256.0 if disparity.dtype == np.uint16 else 1.0
        disparity = disparity[:, ::-1].astype(float) / scale
        if disparity.shape != self._current_disparity.shape:
            raise ValueError('Reverse disparity does not match the source image grid')
        self._right_reference_key, self._right_reference_map = key, disparity
        return disparity

    def _transfer_points_checked(self, request):
        """Direction-aware point transfer. Invalid matches never become annotations."""
        import copy
        points = np.asarray(request['points'], dtype=float)
        side = request.get('source_camera', 'left')
        if side not in ('left', 'right'):
            raise ValueError('source_camera must be left or right')
        if points.ndim != 2 or points.shape[1] != 2 or not 1 <= len(points) <= 4096 or not np.isfinite(points).all():
            raise ValueError('points must contain 1..4096 finite [x,y] positions')
        if request.get('left_image_path') is not None:
            if (request['left_image_path'] != self._current_left_path or
                    request.get('right_image_path') != self._current_right_path or
                    request.get('frame_time') != getattr(self, '_current_frame_time', None)):
                raise ValueError('Point transfer request no longer matches the current frame')
        if self._use_epipolar:
            matcher = self._epipolar_matcher
            source, target = self._left_gray, self._right_gray
            if side == 'right':
                matcher = copy.copy(matcher)
                matcher._K_left, matcher._K_right = matcher._K_right, matcher._K_left
                matcher._K_left_inv = np.linalg.inv(matcher._K_left)
                matcher._R = matcher._R.T
                matcher._T = -matcher._R @ matcher._T
                # Forward DINO image features cannot be used for the reversed pair.
                matcher._dino_available = False
                source, target = target, source
            matched = [matcher.match_point(source, target, p) for p in points]
            h, w = target.shape[:2]
            hs, ws = source.shape[:2]
            valid = [bool(p is not None and np.isfinite(p).all() and
                          0 <= p[0] < w and 0 <= p[1] < h and
                          0 <= original[0] < ws and 0 <= original[1] < hs)
                     for original, p in zip(points, matched)]
        else:
            if self._current_disparity is None:
                raise ValueError('Disparity not ready')
            disparity = self._current_disparity if side == 'left' else self._right_reference_disparity()
            grid = self._to_grid(points, right=(side == 'right'))
            matched_grid, d = self._match_grid(grid, disparity, side == 'right')
            h, w = disparity.shape
            inside = lambda g: (g[:, 0] >= 0) & (g[:, 0] < w) & (g[:, 1] >= 0) & (g[:, 1] < h)
            valid = (np.isfinite(d) & (d > 0) & inside(grid) & inside(matched_grid)).tolist()
            matched = self._from_grid(np.nan_to_num(matched_grid), right=(side == 'left'))
        values = [float((p[0] - q[0]) if side == 'left' else (q[0] - p[0])) if ok else 0.0
                  for p, q, ok in zip(points, matched, valid)]
        return dict(success=all(valid), original_points=points.tolist(),
                    transferred_points=[np.asarray(q, dtype=float).tolist() if ok else None
                                        for q, ok in zip(matched, valid)],
                    valid_matches=[bool(v) for v in valid], disparity_values=values,
                    num_matched=sum(valid),
                    error=None if all(valid) else 'No valid correspondence for one or more points')

    def _deferred_transfer(self, request_id, handler, request):
        """Wait for disparity in a background thread, then send the response."""
        try:
            if not self._disparity_event.wait(timeout=120):
                self._send_response({
                    "id": request_id,
                    "success": False,
                    "error": "Disparity computation timed out",
                })
                return
            response = handler(request)
            response["id"] = request_id
            self._send_response(response)
        except Exception as e:
            self._send_response({
                "id": request_id,
                "success": False,
                "error": str(e),
            })

    def handle_transfer_line(self, request: Dict[str, Any]):
        """
        Transfer a line from left image to right image using disparity.

        Given a line defined by two points on the left image, compute the
        corresponding points on the right image using the disparity map.
        For horizontal stereo, x_right = x_left - disparity.

        If disparity is not yet ready, defers the response until it is.
        """
        if not self._enabled:
            raise ValueError("Service not enabled. Call enable first.")

        # Check readiness under the lock, then release before calling
        # _do_transfer_line (which acquires its own lock). Using a non-
        # reentrant Lock twice from the same thread would deadlock.
        with self._compute_lock:
            if self._use_epipolar:
                ready = self._disparity_ready and self._left_gray is not None
            else:
                ready = self._disparity_ready and self._current_disparity is not None

        if ready:
            return self._do_transfer_line(request)

        # Disparity not ready — wait in background thread so main loop stays responsive
        request_id = request.get("id")
        threading.Thread(
            target=self._deferred_transfer,
            args=(request_id, self._do_transfer_line, request),
            daemon=True,
        ).start()
        return None

    def handle_transfer_points(self, request: Dict[str, Any]):
        """
        Transfer points using dense disparity or epipolar matching.

        Optional source_camera is 'left' (default) or 'right'. With strict=True,
        invalid matches return null coordinates and valid_matches=False rather
        than legacy fallback positions. Right-camera requests always use this
        checked path. Dense right-to-left transfer computes a cached reverse map.
        Optional left_image_path, right_image_path and frame_time identify the
        expected frame and reject requests superseded during deferred processing.

        If disparity is not yet ready, defers the response until it is.
        """
        if not self._enabled:
            raise ValueError("Service not enabled. Call enable first.")

        with self._compute_lock:
            if self._use_epipolar:
                ready = self._disparity_ready and self._left_gray is not None
            else:
                ready = self._disparity_ready and self._current_disparity is not None

        if ready:
            return self._do_transfer_points(request)

        # Disparity not ready — wait in background thread so main loop stays responsive
        request_id = request.get("id")
        threading.Thread(
            target=self._deferred_transfer,
            args=(request_id, self._do_transfer_points, request),
            daemon=True,
        ).start()
        return None

    def handle_measure_line(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Compute the 3D length of a line given its endpoints on BOTH images.

        Two-point lines triangulate supplied endpoints without matching.
        Multi-point lines re-match their centerlines on this frame's disparity,
        deferring the response while it computes, and fall back on the lines
        themselves wherever the disparity cannot place a sample or never comes.
        Optional left_image_path, right_image_path and frame_time identify the
        requested frame so deferred work cannot use a neighbouring video frame.
        """
        if not self._enabled:
            raise ValueError("Service not enabled. Call enable first.")

        left_line = request.get("left_line")
        right_line = request.get("right_line")
        if not left_line or not right_line or min(len(left_line), len(right_line)) < 2:
            raise ValueError("left_line and right_line need at least two points")
        if len(left_line) > 2 or len(right_line) > 2:
            pending = dict(request)
            with self._compute_lock:
                # Legacy callers omit frame identity: snapshot it before waiting.
                if pending.get("left_image_path") is None:
                    pending.update(left_image_path=self._current_left_path,
                                   right_image_path=self._current_right_path,
                                   frame_time=self._current_frame_time)
                ready = self._disparity_ready and (
                    self._left_gray is not None and self._right_gray is not None
                    if self._use_epipolar else self._current_disparity is not None)
            if ready:
                return self._measure_centerline_request(pending)
            threading.Thread(
                target=self._deferred_measure, args=(pending.get("id"), pending), daemon=True,
            ).start()
            return None

        lp1, lp2 = left_line[0], left_line[1]
        rp1, rp2 = right_line[0], right_line[1]

        if self._use_epipolar:
            if not self._epipolar_matcher.calibrated:
                return {"success": False, "error": "Calibration not loaded"}
            measurement = self._epipolar_matcher.compute_measurement(
                lp1, rp1, lp2, rp2)
        else:
            measurement = self._dense_measurement(lp1, rp1, lp2, rp2)

        if measurement is None:
            return {
                "success": False,
                "error": "Could not compute length "
                         "(missing calibration or invalid points)",
            }

        return {
            "success": True,
            "length": measurement["length"],
            "measurement": measurement,
        }

    _DISPARITY_WAIT_SECONDS = 120

    def _deferred_measure(self, request_id, request):
        """Wait for disparity, but measure from the lines alone if it never comes."""
        self._disparity_event.wait(timeout=self._DISPARITY_WAIT_SECONDS)
        try:
            response = self._measure_centerline_request(request)
        except Exception as e:
            response = {"success": False, "error": str(e)}
        response["id"] = request_id
        self._send_response(response)

    def _measure_centerline_request(self, request):
        return self._measure_edited_centerline(
            request["left_line"], request["right_line"], request)

    def _segment_measurement(self, lp1, rp1, lp2, rp2):
        if self._use_epipolar:
            if not self._epipolar_matcher.calibrated:
                return None
            return self._epipolar_matcher.compute_measurement(lp1, rp1, lp2, rp2)
        return self._dense_measurement(lp1, rp1, lp2, rp2)

    def _centerline_measurement(self, left, right):
        """Sum of the piecewise stereo lengths of corresponded polylines."""
        pieces = [self._segment_measurement(left[i], right[i], left[i + 1], right[i + 1])
                  for i in range(len(left) - 1)]
        chord = self._segment_measurement(left[0], right[0], left[-1], right[-1])
        if (chord is None or
                any(p is None or not np.isfinite(p['length']) or p['length'] <= 0 for p in pieces)):
            return {"success": False, "error": "Invalid reconstructed centerline"}
        length = float(sum(p['length'] for p in pieces))
        measurement = dict(chord, length=length, curved_length=length,
                           straight_length=chord['length'],
                           curvature_ratio=length / chord['length'] if chord['length'] > 0 else 1.0,
                           stereo_rms=max(p.get('stereo_rms', 0) for p in pieces))
        return {"success": True, "length": length, "measurement": measurement}

    @staticmethod
    def _paired_points(left_line, right_line, samples):
        """Points on right_line at the vertex-relative positions samples hold
        on left_line: equal vertex counts correspond one to one."""
        left = np.asarray(left_line, dtype=float)
        right = np.asarray(right_line, dtype=float)
        along = lambda points: np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        u = np.interp(along(samples), along(left), np.arange(len(left)))
        return np.column_stack([np.interp(u, np.arange(len(right)), right[:, i]) for i in range(2)])

    @staticmethod
    def _along_curve(curve, reference):
        """Points on curve at the arc-length fractions of reference's points."""
        def fractions(points):
            d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            return d / d[-1] if d[-1] > 0 else np.linspace(0, 1, len(points))
        f = fractions(curve)
        return np.column_stack([np.interp(fractions(reference), f, curve[:, i]) for i in range(2)])

    def _measure_edited_centerline(self, left_line, right_line, request):
        """Re-match edited centerlines on the disparity; vertex indices are not
        stereo matches. A sample the disparity cannot place, or places off the
        right curve, takes the paired vertex position (equal counts) or the
        same arc-length fraction of the right curve instead, and without any
        disparity the whole line measures that way."""
        from viame.core.curved_measurement import resample_polyline
        from scipy.spatial import cKDTree
        left = resample_polyline(left_line, 32)
        right = resample_polyline(right_line, 512)
        fallback = (self._paired_points(left_line, right_line, left)
                    if len(left_line) == len(right_line) else self._along_curve(right, left))
        with self._compute_lock:
            if (request.get("left_image_path") != self._current_left_path or
                    request.get("right_image_path") != self._current_right_path or
                    request.get("frame_time") != self._current_frame_time):
                raise ValueError("Line measurement request no longer matches the current frame")
            placed = np.zeros(len(left), dtype=bool)
            if not self._disparity_ready or (
                    self._left_gray is None or self._right_gray is None
                    if self._use_epipolar else self._current_disparity is None):
                matched = fallback
            elif self._use_epipolar:
                matches = [self._epipolar_matcher.match_point(self._left_gray, self._right_gray, p)
                           for p in left]
                placed = np.array([m is not None for m in matches])
                matched = np.asarray([f if m is None else m for m, f in zip(matches, fallback)], dtype=float)
            else:
                disparity = self._current_disparity
                grid = self._to_grid(left)
                matched_grid, disp = self._match_grid(grid, disparity)
                h, w = disparity.shape
                placed = (np.isfinite(disp) & (disp > 0) & np.isfinite(matched_grid).all(axis=1) &
                          (matched_grid[:, 0] >= 0) & (matched_grid[:, 0] < w) &
                          (matched_grid[:, 1] >= 0) & (matched_grid[:, 1] < h))
                matched = np.where(placed[:, None], self._from_grid(np.nan_to_num(matched_grid)), fallback)
            distance = np.where(np.isfinite(matched).all(axis=1),
                                cKDTree(right).query(np.nan_to_num(matched))[0], np.inf)
            off_curve = distance > 5
            matched = np.where(off_curve[:, None], fallback, matched)
            result = self._centerline_measurement(left, matched)
        if result["success"]:
            result.update(sampled_points=left.tolist(), matched_points=matched.tolist())
            disputed = int((placed & off_curve).sum())
            if disputed and disputed >= 0.25 * placed.sum():
                result["warning"] = (
                    f"Stereo matches disagree with the drawn line on {disputed} of {int(placed.sum())} "
                    f"samples (up to {np.max(distance[placed & off_curve]):.0f} px); the length may be inaccurate")
        return result

    def handle_measure_curve(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Opt-in curved measurement on the current rectified disparity grid.

        Bidirectional mode runs a second inference on horizontally flipped,
        swapped images, then unflips the output into right-reference disparity.
        Explicit frame paths prevent accidentally measuring stale frame data.
        """
        from viame.core.curved_measurement import request_measurement
        if not self._enabled or self._use_epipolar:
            raise ValueError("measure_curve requires an enabled dense stereo backend")
        with self._compute_lock:
            if not self._disparity_ready or self._current_disparity is None:
                raise ValueError("Wait for disparity_ready before measure_curve")
            if (request.get('left_image_path') != self._current_left_path or
                    request.get('right_image_path') != self._current_right_path or
                    request.get('frame_time') != getattr(self, '_current_frame_time', None)):
                raise ValueError("Curve request must identify the current stereo frame")
            config = self._stereo_algo.get_configuration()
            if config.has_value('output_mode') and config.get_value('output_mode') != 'disparity':
                raise ValueError("measure_curve requires disparity output, not depth")
            reverse = None
            if request.get('options', {}).get('mode', 'left') == 'bidirectional':
                from viame.types import Image, ImageContainer
                frame_time = getattr(self, '_current_frame_time', None)
                left, right = self._grid_images(
                    self._current_left_path, self._current_right_path, frame_time)
                result = self._stereo_algo.compute(
                    ImageContainer(Image(np.ascontiguousarray(right[:, ::-1]))),
                    ImageContainer(Image(np.ascontiguousarray(left[:, ::-1]))))
                if result is None:
                    raise ValueError("Reverse stereo inference failed")
                reverse = result.image().asarray()
                if reverse.ndim == 3 and reverse.shape[2] == 1:
                    reverse = reverse[:, :, 0]
                scale = 256.0 if reverse.dtype == np.uint16 else 1.0
                reverse = reverse[:, ::-1].astype(float) / scale
            return request_measurement(request, self._current_disparity, reverse)

    def handle_aggregate_lengths(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate per-detection lengths along a track into a single value.

        Delegates to viame::core::aggregate_lengths (the same C++ helper used by
        the pair_stereo_tracks pipeline process). Needs only the list of lengths.

        Request: { "lengths": [..], "method": "average"|"average_iqr"|"median",
                   "iqr_factor": 1.5 }
        """
        lengths = request.get("lengths")
        if not lengths:
            return {"success": False, "error": "lengths is required"}

        method = request.get("method", "average")
        iqr_factor = float(request.get("iqr_factor", 1.5))

        avg = _cpp_measurement.aggregate_lengths(
            [float(x) for x in lengths], method, iqr_factor)

        if avg is None or avg < 0:
            return {"success": False, "error": "No valid lengths to aggregate"}

        return {"success": True, "avg_length": float(avg)}

    @staticmethod
    def _polygon_to_keypoints(polygon):
        """Head/tail keypoints for a polygon, shared with the segmentation service."""
        from viame.segmentation.segmentation_utils import polygon_to_keypoints
        return polygon_to_keypoints(polygon)

    def _warp_one_point(self, p):
        """Warp a single point from the source to the other camera using whichever
        stereo backend is configured -- epipolar template matching (model-free) or
        the dense disparity algorithm (e.g. Foundation-Stereo). Returns [x, y] or
        None."""
        if self._use_epipolar:
            if self._left_gray is None or self._right_gray is None:
                return None
            matched = self._epipolar_matcher.match_point(
                self._left_gray, self._right_gray, p)
            return [float(matched[0]), float(matched[1])] if matched is not None else None
        if self._current_disparity is not None:
            matched, _ = self._dense_transfer([p])
            return matched[0].tolist()
        return None

    @staticmethod
    def _interior_points(polygon, n):
        """Up to n points well inside a polygon ({"exterior", "holes"}), spread
        apart: the deepest interior point first, then farthest-point picks
        among the pixels at least a third as deep."""
        exterior = np.asarray(polygon.get("exterior") or [], dtype=np.float64)
        if exterior.ndim != 2 or exterior.shape[0] < 3:
            return []
        origin = np.floor(exterior.min(axis=0)) - 1
        size = (np.ceil(exterior.max(axis=0)) - origin + 2).astype(int)
        mask = np.zeros((size[1], size[0]), dtype=np.uint8)
        cv2.fillPoly(mask, [np.rint(exterior - origin).astype(np.int32)], 1)
        for hole in polygon.get("holes") or []:
            if len(hole) >= 3:
                cv2.fillPoly(mask, [np.rint(np.asarray(hole) - origin).astype(np.int32)], 0)
        depth = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        if depth.max() <= 0:
            return []
        ys, xs = np.where(depth >= depth.max() / 3.0)
        candidates = np.column_stack([xs, ys]).astype(np.float64)
        chosen = [candidates[np.argmax(depth[ys, xs])]]
        while len(chosen) < min(n, len(candidates)):
            gaps = np.min([np.hypot(*(candidates - c).T) for c in chosen], axis=0)
            if gaps.max() <= 0:
                break
            chosen.append(candidates[np.argmax(gaps)])
        return [[float(x + origin[0]), float(y + origin[1])] for x, y in chosen]

    def handle_transfer_segmentation_point(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Warp segmentation seed point(s) from the source camera to the other.

        With point sampling enabled, or when no click is given, the seeds are
        points spread well inside each source polygon, warped via the configured
        stereo backend, keeping those that shift the way the rest of their
        polygon does. Otherwise the provided click points are warped directly.

        Request: { "points": [[x,y]..], "labels": [..], "polygon": [[x,y]..],
                   "polygons": [{"exterior": [..], "holes": [[..]]}..],
                   "source_camera": "left" (default) | "right" }
        """
        if not self._enabled:
            raise ValueError("Service not enabled. Call enable first.")

        points = request.get("points") or []
        labels = request.get("labels") or [1] * len(points)
        polygon = request.get("polygon")
        polygons = request.get("polygons") or (
            [{"exterior": polygon, "holes": []}] if polygon else [])
        from_right = request.get("source_camera", "left") == "right"

        def warp(source_points):
            """Valid matches only, as (index, [x, y]) pairs."""
            if from_right:
                response = self._transfer_points(
                    {"points": source_points, "source_camera": "right"})
                return [(i, q) for i, q in enumerate(response["transferred_points"]) if q is not None]
            with self._compute_lock:
                matched = [self._warp_one_point(p) for p in source_points]
            return [(i, q) for i, q in enumerate(matched) if q is not None]

        # Seeds come from inside the source mask when sampling is on, and
        # whenever there is a mask but no click to warp.
        if polygons and (self._seg_point_sampling or not points):
            per_polygon = max(2, -(-self._seg_point_samples // len(polygons)))
            seeds = []
            for poly in polygons:
                samples = self._interior_points(poly, per_polygon)
                matched = warp(samples) if samples else []
                if not matched:
                    continue
                # One object moves as one between the cameras: a sample that
                # shifted unlike the rest matched something else.
                shifts = np.asarray([np.subtract(q, samples[i]) for i, q in matched])
                extent = np.ptp(np.asarray(poly["exterior"], dtype=np.float64), axis=0)
                tolerance = max(4.0, 0.1 * float(np.hypot(*extent)))
                agree = np.hypot(*(shifts - np.median(shifts, axis=0)).T) <= tolerance
                seeds += [[float(q[0]), float(q[1])] for (_, q), ok in zip(matched, agree) if ok]
            if seeds:
                self._log(f"Segmentation seeds: {len(seeds)} interior point(s) of "
                          f"{len(polygons)} source polygon(s)")
                return {
                    "success": True,
                    "transferred_points": seeds,
                    "point_labels": [1] * len(seeds),
                    "sampled": len(seeds),
                    "num_matched": len(seeds),
                }
            self._log("No interior point of the source mask matched; falling back to direct warp")

        if from_right:
            matched = warp(points) if points else []
            return {
                "success": bool(matched),
                "transferred_points": [q for _, q in matched],
                "point_labels": [labels[i] for i, _ in matched],
                "num_matched": len(matched),
            }

        # Direct warp of the supplied click points
        response = self._transfer_points({"points": points})
        response["point_labels"] = labels
        # Dense mode always produces a disparity per point; treat each
        # returned point as a match unless the epipolar path reported otherwise.
        response.setdefault("num_matched", len(response.get("transferred_points") or []))
        return response

    def handle_measure_from_polygons(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a head/tail line from each camera's segmentation polygon and the
        corresponding stereo measurement, when segmentation_generate_line is enabled.
        The lines use the add_keypoints_from_mask oriented-bbox
        algorithm (the measurement pipeline default); the measurement reuses
        compute_stereo_measurement. When disabled, reports generate_line=False so the
        caller does nothing extra.

        Request: { "polygon_left": [[x,y]..], "polygon_right": [[x,y]..] }
        """
        if not self._enabled:
            raise ValueError("Service not enabled. Call enable first.")
        if not self._seg_generate_line:
            return {"success": True, "generate_line": False}

        poly_left = request.get("polygon_left")
        poly_right = request.get("polygon_right")
        if not poly_left or not poly_right:
            return {"success": True, "generate_line": False}

        kl = self._polygon_to_keypoints(poly_left)
        kr = self._polygon_to_keypoints(poly_right)
        if kl is None or kr is None:
            return {"success": True, "generate_line": False}

        (left_head, left_tail) = kl
        (right_head, right_tail) = kr
        result = {
            "success": True,
            "generate_line": True,
            "line_left": [left_head, left_tail],
            "line_right": [right_head, right_tail],
        }

        if self._use_epipolar and self._epipolar_matcher.calibrated:
            measurement = self._epipolar_matcher.compute_measurement(
                left_head, right_head, left_tail, right_tail)
            if measurement is not None:
                result["measurement"] = measurement

        return result

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Route request to appropriate handler."""
        command = request.get("command")

        handlers = {
            "enable": self.handle_enable,
            "disable": self.handle_disable,
            "set_calibration": self.handle_set_calibration,
            "set_frame": self.handle_set_frame,
            "cancel": self.handle_cancel,
            "get_status": self.handle_get_status,
            "transfer_line": self.handle_transfer_line,
            "transfer_points": self.handle_transfer_points,
            "measure_line": self.handle_measure_line,
            "measure_curve": self.handle_measure_curve,
            "aggregate_lengths": self.handle_aggregate_lengths,
        }

        handler = handlers.get(command)
        if not handler:
            raise ValueError(f"Unknown command: {command}")

        return handler(request)

    def run(self) -> None:
        """Main loop: read JSON requests from stdin, write responses to stdout."""
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
                    self._cancel_computation()
                    if self._compute_queue:
                        self._compute_queue.put(None)  # Signal worker to exit
                    self._send_response({
                        "id": request_id,
                        "success": True,
                        "message": "Shutting down",
                    })
                    break

                # Process request
                response = self.handle_request(request)
                if response is not None:
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


def load_algorithm_from_config(config_path: str, plugin_paths: List[str] = None):
    """
    Load and configure stereo algorithms from a KWIVER config file.

    Supports two modes:
    - Epipolar template matching: when ``matching_method`` key is present
    - Dense disparity (ComputeStereoDepthMap): when ``compute_stereo_depth_map:type`` is present

    Args:
        config_path: Path to the config file
        plugin_paths: Optional list of additional plugin paths to load

    Returns:
        Tuple of (compute_stereo_depth_map_algo, epipolar_matcher, service_config)
    """
    import viame.config as vital_config
    from viame.modules import modules as vital_modules

    # Load plugin modules
    vital_modules.load_known_modules()

    if plugin_paths:
        for path in plugin_paths:
            if os.path.isdir(path):
                vital_modules.load_module(path)

    # Read config file using vital's built-in loader (supports includes)
    config_dir = os.path.dirname(os.path.abspath(config_path))
    # Newer bindings type search_paths as an opaque ConfigKeys vector.
    search_paths = getattr(vital_config, "ConfigKeys", list)()
    search_paths.append(config_dir)
    cfg = vital_config.read_config_file(config_path, search_paths)

    # Check for epipolar template matching mode
    epipolar_matcher = None
    if cfg.has_value("matching_method"):
        method = cfg.get_value("matching_method")
        if method == "epipolar_template_matching":
            def _cfg_float(key, default):
                return float(cfg.get_value(key)) if cfg.has_value(key) else default
            def _cfg_int(key, default):
                return int(cfg.get_value(key)) if cfg.has_value(key) else default

            epipolar_matcher = EpipolarTemplateMatcher(
                template_size=_cfg_int("template_size", 13),
                template_matching_threshold=_cfg_float(
                    "template_matching_threshold", 0.5),
                epipolar_min_disparity=_cfg_float("epipolar_min_disparity", 2.0),
                epipolar_max_disparity=_cfg_float(
                    "epipolar_max_disparity", 300.0),
                epipolar_num_samples=_cfg_int("epipolar_num_samples", 5000),
                dino_model_name=cfg.get_value("dino_model_name")
                    if cfg.has_value("dino_model_name") else "dinov2_vitb14",
                dino_top_k=_cfg_int("dino_top_k", 0),
                dino_weights_path=cfg.get_value("dino_weights_path")
                    if cfg.has_value("dino_weights_path") else "",
            )

    # Check for dense disparity algorithm
    stereo_algo = None
    if cfg.has_value("compute_stereo_depth_map:type"):
        from viame.algo import ComputeStereoDepthMap
        impl_name = cfg.get_value("compute_stereo_depth_map:type")
        stereo_algo = ComputeStereoDepthMap.create(impl_name)
        stereo_algo.set_configuration(cfg.subblock("compute_stereo_depth_map:" + impl_name))

    def _cfg_bool(key, default):
        if not cfg.has_value(key):
            return default
        return str(cfg.get_value(key)).strip().lower() in ("true", "1", "yes", "on")

    # Extract service configuration
    dense_grid_options = {
        key: str(cfg.get_value(key)) for key in DENSE_GRID_DEFAULTS if cfg.has_value(key)
    }
    service_config = {
        "scale": float(cfg.get_value("service:scale")) if cfg.has_value("service:scale") else 1.0,
        "dense_grid_options": dense_grid_options,
        "segmentation_generate_line": _cfg_bool("segmentation_generate_line", False),
        "segmentation_point_sampling": _cfg_bool("segmentation_point_sampling", False),
        "segmentation_point_samples": int(cfg.get_value("segmentation_point_samples"))
            if cfg.has_value("segmentation_point_samples") else 5,
        "max_transfer_size_ratio": float(cfg.get_value("max_transfer_size_ratio"))
            if cfg.has_value("max_transfer_size_ratio") else 2.5,
    }

    return stereo_algo, epipolar_matcher, service_config


def find_viame_config() -> Optional[str]:
    """
    Find the default stereo config file in VIAME install.

    Returns:
        Path to config file if found, None otherwise
    """
    viame_install = os.environ.get("VIAME_INSTALL")
    if not viame_install:
        return None

    pipelines_dir = Path(viame_install) / "configs" / "pipelines"
    config_path = pipelines_dir / "interactive_stereo_default.conf"
    if config_path.exists():
        return str(config_path)

    return None


def create_default_config(output_path: str):
    """
    Create a default config file for the interactive stereo service.

    This generates a config file that uses epipolar template matching.

    Args:
        output_path: Path to write the config file
    """
    config = """# Interactive Stereo Service Configuration
# This config uses epipolar template matching (same approach as
# stereo_measure_current_annots_template.pipe). No GPU required.

matching_method = epipolar_template_matching

# Template matching parameters
template_size = 13
template_matching_threshold = 0.5

# Epipolar search range (disparity-based, in pixels)
epipolar_min_disparity = 2
epipolar_max_disparity = 300
epipolar_num_samples = 5000

# DINO + NCC two-stage matching (optional, requires Python + PyTorch + GPU).
# DINOv2 features select the top-K semantically similar candidates, then NCC
# picks the precise match. Reduces false matches on repetitive textures.
# Set dino_top_k to 0 (default) to disable, or 100 for recommended setting.
# dino_top_k = 100
# dino_model_name = dinov2_vitb14

# Service settings
service:scale = 1.0
"""

    with open(output_path, 'w') as f:
        f.write(config)

    print(f"Created default config: {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Stereo Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use a config file
    python -m viame.core.interactive_stereo --config /path/to/config.pipe

    # Generate a default config file
    python -m viame.core.interactive_stereo --generate-config stereo.conf

    # With additional plugin paths
    python -m viame.core.interactive_stereo --config config.pipe --plugin-path /path/to/plugins
        """
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to KWIVER config file",
    )
    parser.add_argument(
        "--generate-config",
        default=None,
        metavar="OUTPUT_PATH",
        help="Generate a default config file and exit",
    )
    parser.add_argument(
        "--plugin-path",
        action="append",
        default=[],
        help="Additional plugin paths to load (can be specified multiple times)",
    )
    args = parser.parse_args()

    # Handle config generation
    if args.generate_config:
        create_default_config(args.generate_config)
        return

    # Require config file
    if not args.config:
        parser.error("--config is required (or use --generate-config to create one)")

    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    try:
        # Load algorithm from config
        stereo_algo, epipolar_matcher, service_config = load_algorithm_from_config(
            args.config, args.plugin_path
        )

        if stereo_algo is None and epipolar_matcher is None:
            print("Error: No stereo algorithm or matching method configured",
                  file=sys.stderr)
            sys.exit(1)

        # Create and run service
        service = InteractiveStereoService(
            compute_stereo_depth_map_algo=stereo_algo,
            epipolar_matcher=epipolar_matcher,
            **service_config
        )
        service.run()

    except KeyboardInterrupt:
        print("[InteractiveStereo] Interrupted", file=sys.stderr)
    except Exception as e:
        print(f"[InteractiveStereo] Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
