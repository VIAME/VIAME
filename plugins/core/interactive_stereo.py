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
    - "enable": Load the algorithm and enable the service (requires calibration)
    - "disable": Unload the algorithm and disable the service
    - "set_frame": Start computing disparity for stereo pair (proactive)
    - "cancel": Cancel current disparity computation
    - "get_status": Get current status (enabled, computing, ready)
    - "transfer_line": Transfer a line from left to right image using disparity
    - "transfer_points": Transfer multiple points from left to right image
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


class EpipolarTemplateMatcher:
    """
    Per-point stereo matching along epipolar curves using template matching.

    Mirrors the epipolar_template_matching method in measurement_utilities.cxx.
    Uses camera calibration to compute epipolar curves in unrectified images,
    then matches template patches from the source image along the curve in
    the target image using Normalized Cross-Correlation (NCC).

    When dino_top_k > 0 and the dino3_matcher module is available, uses
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
        self._dino_matcher = None
        self._dino_available = False
        self._dino_images_set = False

        if dino_top_k > 0:
            self._init_dino()

    def _log(self, msg):
        print(f"[EpipolarMatcher] {msg}", file=sys.stderr, flush=True)

    def _init_dino(self):
        """Try to import and initialize the DINO matcher module."""
        try:
            from viame.pytorch import dino3_matcher
            self._dino_matcher = dino3_matcher
            dino3_matcher.init_matcher(
                model_name=self._dino_model_name, device="cuda", threshold=0.0)
            self._dino_available = True
            self._log(f"DINO matcher initialized: model={self._dino_model_name}, "
                      f"top_k={self._dino_top_k}")
        except Exception as e:
            self._log(f"DINO matcher not available ({e}), using NCC only")
            self._dino_available = False
            self._dino_top_k = 0

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
        Load stereo calibration from a file using KWIVER's StereoCalibration.

        Supports .npz, .json, and .mat formats via StereoCalibration.from_file().
        """
        from viame.opencv.stereo_algos import StereoCalibration

        self._log(f"Loading calibration from: {filepath}")
        cal = StereoCalibration.from_file(filepath)

        # Extract intrinsic matrices (3x3)
        self._K_left, self._K_right = cal.intrinsic_matrices()
        self._K_left = np.array(self._K_left, dtype=np.float64)
        self._K_right = np.array(self._K_right, dtype=np.float64)
        self._K_left_inv = np.linalg.inv(self._K_left)

        # Extract extrinsic R, T from the right camera
        om = cal.data['right']['extrinsic']['om']
        self._R = cv2.Rodrigues(np.array(om, dtype=np.float64))[0]
        self._T = np.array(
            cal.data['right']['extrinsic']['T'], dtype=np.float64).flatten()

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


class InteractiveStereoService:
    """
    Interactive Stereo Service using KWIVER vital algorithms.

    Handles stdin/stdout JSON protocol communication and delegates
    to configured vital algorithms for disparity computation.
    """

    def __init__(
        self,
        compute_stereo_depth_map_algo=None,
        epipolar_matcher: Optional[EpipolarTemplateMatcher] = None,
        scale: float = 1.0,
    ):
        """
        Initialize the service with configured algorithms.

        Args:
            compute_stereo_depth_map_algo: Configured ComputeStereoDepthMap algorithm instance
                (for dense disparity mode). Mutually exclusive with epipolar_matcher.
            epipolar_matcher: Configured EpipolarTemplateMatcher instance
                (for per-point epipolar template matching mode).
            scale: Scale factor for input images (<=1.0). Lower = faster but less accurate.
        """
        self._stereo_algo = compute_stereo_depth_map_algo
        self._epipolar_matcher = epipolar_matcher
        self._use_epipolar = epipolar_matcher is not None
        self._scale = scale
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
        """Send JSON response to stdout."""
        print(json.dumps(response), flush=True)

    def _send_error(self, request_id: Optional[str], error: str) -> None:
        """Send error response."""
        self._send_response({
            "id": request_id,
            "success": False,
            "error": error,
        })

    def _add_to_cache(self, left_path: str, right_path: str, disparity: np.ndarray) -> None:
        """Add a disparity map to the cache with LRU eviction."""
        cache_key = (left_path, right_path)

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

    def _get_from_cache(self, left_path: str, right_path: str) -> Optional[np.ndarray]:
        """Get a disparity map from the cache if available."""
        cache_key = (left_path, right_path)
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

    def _load_image(self, image_path: str):
        """Load an image and return a vital ImageContainer."""
        from kwiver.vital.types import ImageContainer, Image

        from viame.core.segmentation_utils import load_image

        imdata = load_image(image_path)
        return ImageContainer(Image(imdata))

    def _compute_disparity_sync(
        self,
        left_path: str,
        right_path: str,
    ) -> Optional[np.ndarray]:
        """
        Compute disparity for stereo pair synchronously using the configured algorithm.
        Returns disparity map or None if cancelled.
        """
        if self._cancel_event.is_set():
            return None

        # Load images as ImageContainers
        left_container = self._load_image(left_path)
        right_container = self._load_image(right_path)

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

            request_id, left_path, right_path = task

            try:
                self._cancel_event.clear()
                self._log(f"Starting disparity computation for {left_path}")

                disparity = self._compute_disparity_sync(left_path, right_path)

                if disparity is not None and not self._cancel_event.is_set():
                    with self._compute_lock:
                        # Add to cache for future use
                        self._add_to_cache(left_path, right_path, disparity)

                        # Only update if this is still the current frame
                        if (self._current_left_path == left_path and
                                self._current_right_path == right_path):
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

        if not left_path or not right_path:
            raise ValueError("left_image_path and right_image_path are required")

        if not os.path.exists(left_path):
            raise ValueError(f"Left image not found: {left_path}")
        if not os.path.exists(right_path):
            raise ValueError(f"Right image not found: {right_path}")

        # Check if already computing this frame
        with self._compute_lock:
            if (self._current_left_path == left_path and
                    self._current_right_path == right_path):
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

            if self._use_epipolar:
                # Epipolar mode: update state, will load images below
                self._cancel_computation()
                self._current_left_path = left_path
                self._current_right_path = right_path
            else:
                # Dense mode: check if we have this disparity cached
                cached_disparity = self._get_from_cache(left_path, right_path)
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
        self._compute_queue.put((request_id, left_path, right_path))

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

                return {
                    "success": True,
                    "transferred_line": [
                        [float(right_p1[0]), float(right_p1[1])],
                        [float(right_p2[0]), float(right_p2[1])],
                    ],
                    "original_line": line,
                }

            H, W = self._current_disparity.shape[:2]

            def clamp_point(p):
                x = max(0, min(W - 1, int(round(p[0]))))
                y = max(0, min(H - 1, int(round(p[1]))))
                return x, y

            x1, y1 = clamp_point(p1)
            x2, y2 = clamp_point(p2)

            disp1 = float(self._current_disparity[y1, x1])
            disp2 = float(self._current_disparity[y2, x2])

            x1_right = p1[0] - disp1
            x2_right = p2[0] - disp2

            transferred_line = [
                [float(x1_right), float(p1[1])],
                [float(x2_right), float(p2[1])],
            ]

            depth_info = None
            if self._focal_length > 0 and self._baseline > 0:
                depth1 = (self._focal_length * self._baseline) / max(disp1, 1e-6) if disp1 > 0 else None
                depth2 = (self._focal_length * self._baseline) / max(disp2, 1e-6) if disp2 > 0 else None
                depth_info = {
                    "depth_point1": depth1,
                    "depth_point2": depth2,
                    "disparity_point1": disp1,
                    "disparity_point2": disp2,
                }

            return {
                "success": True,
                "transferred_line": transferred_line,
                "original_line": line,
                "depth_info": depth_info,
            }

    def _do_transfer_points(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute points transfer with lock already held or disparity known ready."""
        with self._compute_lock:
            points = request.get("points")
            if not points:
                raise ValueError("points is required")

            if self._use_epipolar:
                transferred_points = []
                disparity_values = []

                for p in points:
                    matched = self._epipolar_matcher.match_point(
                        self._left_gray, self._right_gray, p)
                    if matched is not None:
                        transferred_points.append(
                            [float(matched[0]), float(matched[1])])
                        disparity_values.append(float(p[0] - matched[0]))
                    else:
                        transferred_points.append([float(p[0]), float(p[1])])
                        disparity_values.append(0.0)

                return {
                    "success": True,
                    "transferred_points": transferred_points,
                    "original_points": points,
                    "disparity_values": disparity_values,
                }

            H, W = self._current_disparity.shape[:2]
            transferred_points = []
            disparity_values = []

            for p in points:
                x = max(0, min(W - 1, int(round(p[0]))))
                y = max(0, min(H - 1, int(round(p[1]))))

                disp = float(self._current_disparity[y, x])
                x_right = p[0] - disp

                transferred_points.append([float(x_right), float(p[1])])
                disparity_values.append(disp)

            return {
                "success": True,
                "transferred_points": transferred_points,
                "original_points": points,
                "disparity_values": disparity_values,
            }

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
        Transfer multiple points from left image to right image using disparity.

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
    from kwiver.vital.config import config as vital_config
    from kwiver.vital.modules import modules as vital_modules

    # Load plugin modules
    vital_modules.load_known_modules()

    if plugin_paths:
        for path in plugin_paths:
            if os.path.isdir(path):
                vital_modules.load_module(path)

    # Read config file using vital's built-in loader (supports includes)
    config_dir = os.path.dirname(os.path.abspath(config_path))
    cfg = vital_config.read_config_file(config_path, [config_dir])

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
            )

    # Check for dense disparity algorithm
    stereo_algo = None
    if cfg.has_value("compute_stereo_depth_map:type"):
        from kwiver.vital.algo import ComputeStereoDepthMap
        impl_name = cfg.get_value("compute_stereo_depth_map:type")
        stereo_algo = ComputeStereoDepthMap.create(impl_name)
        stereo_algo.set_configuration(cfg.subblock("compute_stereo_depth_map:" + impl_name))

    # Extract service configuration
    service_config = {
        "scale": float(cfg.get_value("service:scale")) if cfg.has_value("service:scale") else 1.0,
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
# measurement_from_annotations_template.pipe). No GPU required.

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
