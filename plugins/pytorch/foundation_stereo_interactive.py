#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved
# BSD 3-Clause License. See either the root top-level LICENSE file or
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

"""
Foundation Stereo Interactive Service

A persistent process that keeps the Foundation Stereo model loaded and handles
disparity computation requests via stdin/stdout JSON protocol. Designed for
interactive stereo annotation where lines drawn on the left image are automatically
transferred to the right image using disparity mapping.

Unlike SAM, this service proactively computes disparity maps when the user navigates
to a new frame, so the disparity is ready when they draw annotations.

Usage:
    python foundation_stereo_interactive.py [--checkpoint CHECKPOINT] [--device DEVICE]

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
    - "enable": Load the model and enable the service (requires calibration)
    - "disable": Unload the model and disable the service
    - "set_frame": Start computing disparity for stereo pair (proactive)
    - "cancel": Cancel current disparity computation
    - "get_status": Get current status (enabled, computing, ready)
    - "transfer_line": Transfer a line from left to right image using disparity
    - "shutdown": Gracefully terminate the service
"""

import argparse
import contextlib
import json
import os
import sys
import threading
import queue
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class FoundationStereoInteractiveService:
    """Persistent Foundation Stereo service with stdin/stdout JSON protocol."""

    def __init__(
        self,
        checkpoint: str,
        config_path: str = "",
        vit_size: str = "vits",
        device: str = "cuda",
        scale: float = 0.25,
        use_half_precision: bool = True,
        num_iters: int = 32,
    ):
        self.checkpoint = checkpoint
        self.config_path = config_path
        self.vit_size = vit_size
        self.device = device
        self.scale = scale
        self.use_half_precision = use_half_precision
        self.num_iters = num_iters

        # Model state
        self._model = None
        self._InputPadder = None
        self._torch_device = None
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

        # Background computation
        self._compute_thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()
        self._compute_lock = threading.Lock()
        self._compute_queue = queue.Queue()

    def initialize(self) -> None:
        """Load the Foundation Stereo model. Called when service is enabled."""
        import torch

        self._log("Initializing Foundation Stereo model...")
        self._log(f"  Checkpoint: {self.checkpoint}")
        self._log(f"  ViT Size: {self.vit_size}")
        self._log(f"  Device: {self.device}")
        self._log(f"  Scale: {self.scale}")
        self._log(f"  Half Precision: {self.use_half_precision}")

        # Add foundation-stereo to path
        foundation_stereo_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'packages', 'pytorch-libs', 'foundation-stereo'
        )
        if foundation_stereo_dir not in sys.path:
            sys.path.insert(0, foundation_stereo_dir)

        from omegaconf import OmegaConf
        from core.foundation_stereo import FoundationStereo as FoundationStereoModel
        from core.utils.utils import InputPadder

        self._InputPadder = InputPadder

        # Load model configuration
        if self.config_path and os.path.exists(self.config_path):
            model_cfg = OmegaConf.load(self.config_path)
        else:
            ckpt_dir = os.path.dirname(self.checkpoint)
            cfg_path = os.path.join(ckpt_dir, 'cfg.yaml')
            if os.path.exists(cfg_path):
                model_cfg = OmegaConf.load(cfg_path)
            else:
                model_cfg = OmegaConf.create({})

        model_cfg['vit_size'] = self.vit_size

        # Create and load model
        self._model = FoundationStereoModel(model_cfg)
        ckpt = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
        self._model.load_state_dict(ckpt['model'])

        # Setup device
        device_str = self.device
        if device_str == 'auto':
            device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._torch_device = torch.device(device_str)
        self._model.to(self._torch_device)

        # Convert to half precision if requested
        if self.use_half_precision and 'cuda' in device_str:
            self._model.half()
        self._model.eval()

        torch.set_grad_enabled(False)

        self._enabled = True
        self._log("Foundation Stereo model initialized successfully")

    def unload(self) -> None:
        """Unload the model and free resources."""
        import torch

        self._cancel_computation()
        self._model = None
        self._InputPadder = None
        self._torch_device = None
        self._current_disparity = None
        self._disparity_ready = False
        self._current_left_path = None
        self._current_right_path = None
        self._enabled = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._log("Model unloaded")

    def _log(self, message: str) -> None:
        """Log to stderr (stdout is reserved for JSON responses)."""
        print(f"[FoundationStereo] {message}", file=sys.stderr, flush=True)

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

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image and return as RGB numpy array."""
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        return np.array(img)

    def _compute_disparity_sync(
        self,
        left_path: str,
        right_path: str,
    ) -> Optional[np.ndarray]:
        """
        Compute disparity for stereo pair synchronously.
        Returns disparity map or None if cancelled.
        """
        import torch
        import cv2

        if self._cancel_event.is_set():
            return None

        # Load images
        left_npy = self._load_image(left_path)
        right_npy = self._load_image(right_path)

        if self._cancel_event.is_set():
            return None

        if left_npy.shape != right_npy.shape:
            raise RuntimeError(
                f"Left and right image dimensions must match: "
                f"{left_npy.shape} vs {right_npy.shape}"
            )

        H_orig, W_orig = left_npy.shape[:2]

        # Scale down images for speed
        if self.scale < 1.0:
            H_scaled = int(H_orig * self.scale)
            W_scaled = int(W_orig * self.scale)
            left_npy = cv2.resize(left_npy, (W_scaled, H_scaled), interpolation=cv2.INTER_AREA)
            right_npy = cv2.resize(right_npy, (W_scaled, H_scaled), interpolation=cv2.INTER_AREA)
            H, W = H_scaled, W_scaled
        else:
            H, W = H_orig, W_orig

        if self._cancel_event.is_set():
            return None

        # Convert to PyTorch tensors
        use_half = self.use_half_precision and 'cuda' in str(self._torch_device)
        left_tensor = torch.as_tensor(left_npy).to(self._torch_device)
        left_tensor = left_tensor.half() if use_half else left_tensor.float()
        left_tensor = left_tensor[None].permute(0, 3, 1, 2)
        right_tensor = torch.as_tensor(right_npy).to(self._torch_device)
        right_tensor = right_tensor.half() if use_half else right_tensor.float()
        right_tensor = right_tensor[None].permute(0, 3, 1, 2)

        if self._cancel_event.is_set():
            return None

        # Pad images
        padder = self._InputPadder(left_tensor.shape, divis_by=32, force_square=False)
        left_padded, right_padded = padder.pad(left_tensor, right_tensor)

        if self._cancel_event.is_set():
            return None

        # Run inference
        with torch.amp.autocast('cuda', enabled=True):
            disp = self._model.forward(
                left_padded, right_padded,
                iters=self.num_iters,
                test_mode=True,
                low_memory=True
            )

        if self._cancel_event.is_set():
            return None

        # Unpad and convert to numpy
        disp = padder.unpad(disp.float())
        disp_npy = disp.data.cpu().numpy().reshape(H, W)

        # Scale disparity back to original resolution if needed
        if self.scale < 1.0:
            disp_npy = cv2.resize(disp_npy, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
            disp_npy = disp_npy / self.scale

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
                        # Only update if this is still the current frame
                        if (self._current_left_path == left_path and
                                self._current_right_path == right_path):
                            self._current_disparity = disparity
                            self._disparity_ready = True
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
        """Enable the service by loading the model."""
        if self._enabled:
            return {
                "success": True,
                "message": "Already enabled",
            }

        # Load calibration if provided
        calibration = request.get("calibration")
        if calibration:
            self._load_calibration(calibration)

        try:
            self.initialize()
            self._start_background_worker()
            return {
                "success": True,
                "message": "Foundation Stereo enabled",
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to enable: {e}",
            }

    def handle_disable(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Disable the service and unload the model."""
        if not self._enabled:
            return {
                "success": True,
                "message": "Already disabled",
            }

        self.unload()
        return {
            "success": True,
            "message": "Foundation Stereo disabled",
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
        Cancels any previous computation.
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

            # Cancel current computation and start new one
            self._cancel_computation()
            self._current_left_path = left_path
            self._current_right_path = right_path
            self._current_disparity = None
            self._disparity_ready = False

        # Queue the new computation
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

    def handle_transfer_line(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transfer a line from left image to right image using disparity.

        Given a line defined by two points on the left image, compute the
        corresponding points on the right image using the disparity map.
        For horizontal stereo, x_right = x_left - disparity.
        """
        if not self._enabled:
            raise ValueError("Service not enabled. Call enable first.")

        with self._compute_lock:
            if not self._disparity_ready or self._current_disparity is None:
                raise ValueError("Disparity not ready. Wait for disparity computation.")

            line = request.get("line")
            if not line or len(line) != 2:
                raise ValueError("line must be a list of two [x, y] points")

            # Extract points
            p1 = line[0]
            p2 = line[1]

            H, W = self._current_disparity.shape

            # Clamp coordinates to image bounds
            def clamp_point(p):
                x = max(0, min(W - 1, int(round(p[0]))))
                y = max(0, min(H - 1, int(round(p[1]))))
                return x, y

            x1, y1 = clamp_point(p1)
            x2, y2 = clamp_point(p2)

            # Get disparity values at the line endpoints
            disp1 = float(self._current_disparity[y1, x1])
            disp2 = float(self._current_disparity[y2, x2])

            # For horizontal stereo: x_right = x_left - disparity
            # y stays the same (rectified stereo)
            x1_right = p1[0] - disp1
            x2_right = p2[0] - disp2

            # Return transferred line
            transferred_line = [
                [float(x1_right), float(p1[1])],
                [float(x2_right), float(p2[1])],
            ]

            # Also compute depth if calibration is available
            depth_info = None
            if self._focal_length > 0 and self._baseline > 0:
                # depth = focal_length * baseline / disparity
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

    def handle_transfer_points(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transfer multiple points from left image to right image using disparity.
        """
        if not self._enabled:
            raise ValueError("Service not enabled. Call enable first.")

        with self._compute_lock:
            if not self._disparity_ready or self._current_disparity is None:
                raise ValueError("Disparity not ready. Wait for disparity computation.")

            points = request.get("points")
            if not points:
                raise ValueError("points is required")

            H, W = self._current_disparity.shape
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
                response["id"] = request_id
                self._send_response(response)

            except json.JSONDecodeError as e:
                self._send_error(request_id, f"Invalid JSON: {e}")
            except Exception as e:
                self._log(f"Error processing request: {e}")
                self._send_error(request_id, str(e))

        self._log("Service shutting down")


def parse_pipe_config(config_path: str) -> Dict[str, Any]:
    """
    Parse a VIAME .pipe config file and extract foundation_stereo settings.
    Returns a dict with the parsed values.
    """
    config = {}
    if not os.path.exists(config_path):
        return config

    try:
        with open(config_path, 'r') as f:
            in_foundation_stereo_block = False
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Check for config block start
                if line.startswith('config foundation_stereo'):
                    in_foundation_stereo_block = True
                    continue

                # Parse settings within the foundation_stereo block
                if in_foundation_stereo_block and line.startswith(':'):
                    # Parse :key value format
                    parts = line[1:].split(None, 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        config[key] = value
    except Exception as e:
        print(f"[FoundationStereo] Warning: Failed to parse config file: {e}", file=sys.stderr)

    return config


def main():
    parser = argparse.ArgumentParser(description="Foundation Stereo Interactive Service")
    parser.add_argument(
        "--viame-path",
        default=None,
        help="Path to VIAME installation directory (used to find model checkpoint)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to Foundation Stereo checkpoint (defaults to VIAME_PATH/configs/pipelines/models/foundation_stereo_s.pth)",
    )
    parser.add_argument(
        "--config",
        default="",
        help="Path to model config yaml file",
    )
    parser.add_argument(
        "--vit-size",
        default=None,
        choices=["vitl", "vitb", "vits"],
        help="Vision Transformer backbone size",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Scale factor for input images (<=1.0). Lower = faster but less accurate.",
    )
    parser.add_argument(
        "--half-precision",
        action="store_true",
        default=False,
        help="Use half precision (FP16) for model and tensors",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=None,
        help="Number of GRU refinement iterations",
    )
    args = parser.parse_args()

    # Determine VIAME path
    viame_path = args.viame_path
    if viame_path is None:
        viame_path = os.environ.get("VIAME_INSTALL")

    if viame_path is None:
        print("[FoundationStereo] Error: No VIAME path specified and VIAME_INSTALL not set", file=sys.stderr)
        print("[FoundationStereo] Use --viame-path to specify VIAME location", file=sys.stderr)
        sys.exit(1)

    pipelines_dir = Path(viame_path) / "configs" / "pipelines"

    # Try to load settings from common_foundation_stereo.pipe config file
    pipe_config_path = pipelines_dir / "common_foundation_stereo.pipe"
    pipe_config = parse_pipe_config(str(pipe_config_path))

    if pipe_config:
        print(f"[FoundationStereo] Loaded config from: {pipe_config_path}", file=sys.stderr)

    # Get values from config file, with command-line args taking precedence
    vit_size = args.vit_size or pipe_config.get('vit_size', 'vits')
    scale = args.scale if args.scale is not None else float(pipe_config.get('scale', 0.25))
    num_iters = args.num_iters if args.num_iters is not None else int(pipe_config.get('num_iters', 32))

    # Half precision: command-line flag or config file
    use_half_precision = args.half_precision
    if not use_half_precision and pipe_config.get('use_half_precision', '').lower() == 'true':
        use_half_precision = True

    # Determine checkpoint path
    checkpoint = args.checkpoint
    if checkpoint is None:
        # Try to get from config file (relativepath format)
        checkpoint_rel = pipe_config.get('relativepath:checkpoint_path', 'models/foundation_stereo_s.pth')
        checkpoint = str(pipelines_dir / checkpoint_rel)

    # Determine model config path
    model_config = args.config
    if not model_config:
        model_config_rel = pipe_config.get('relativepath:model_config_path', '')
        if model_config_rel:
            model_config = str(pipelines_dir / model_config_rel)

    # Verify checkpoint exists
    if not Path(checkpoint).exists():
        print(f"[FoundationStereo] Error: Checkpoint not found: {checkpoint}", file=sys.stderr)
        print(f"[FoundationStereo] Expected location: {checkpoint}", file=sys.stderr)
        print("[FoundationStereo] Make sure Foundation Stereo model is installed in VIAME", file=sys.stderr)
        sys.exit(1)

    print(f"[FoundationStereo] Using checkpoint: {checkpoint}", file=sys.stderr)
    print(f"[FoundationStereo] Settings: vit_size={vit_size}, scale={scale}, half_precision={use_half_precision}, num_iters={num_iters}", file=sys.stderr)

    service = FoundationStereoInteractiveService(
        checkpoint=checkpoint,
        config_path=model_config,
        vit_size=vit_size,
        device=args.device,
        scale=scale,
        use_half_precision=use_half_precision,
        num_iters=num_iters,
    )

    try:
        service.run()
    except KeyboardInterrupt:
        print("[FoundationStereo] Interrupted", file=sys.stderr)
    except Exception as e:
        print(f"[FoundationStereo] Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
