#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved
# BSD 3-Clause License. See either the root top-level LICENSE file or
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

"""
SAM2 Interactive Inference Service

A persistent process that keeps SAM2 loaded and handles inference requests
via stdin/stdout JSON protocol. Designed to be spawned by both Desktop (Electron)
and Web (Girder) platforms for fast interactive segmentation.

Usage:
    python sam2_interactive.py [--cfg CONFIG] [--checkpoint CHECKPOINT] [--device DEVICE]

Protocol:
    Input (JSON per line on stdin):
    {
        "id": "unique-request-id",
        "command": "predict",
        "image_path": "/path/to/frame.png",
        "points": [[x1, y1], [x2, y2], ...],
        "point_labels": [1, 1, 0, ...],  // 1=foreground, 0=background
        "multimask_output": false,
        "mask_input": null  // optional low-res mask for refinement
    }

    Output (JSON per line on stdout):
    {
        "id": "unique-request-id",
        "success": true,
        "polygon": [[x1, y1], [x2, y2], ...],
        "bounds": [x_min, y_min, x_max, y_max],
        "score": 0.95,
        "low_res_mask": [...]  // for subsequent refinement
    }

    Commands:
    - "predict": Run point-based segmentation on an image
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from viame.core.segmentation_utils import (
    load_image,
    mask_to_polygon,
    adaptive_simplify_polygon,
)


class SAM2InteractiveService:
    """Persistent SAM2 inference service with stdin/stdout JSON protocol."""

    def __init__(
        self,
        cfg: str = "configs/sam2.1/sam2.1_hiera_b+.yaml",
        checkpoint: str = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        device: str = "cuda",
        hole_policy: str = "remove",
        multipolygon_policy: str = "largest",
        max_polygon_points: int = 25,
    ):
        self.cfg = cfg
        self.checkpoint = checkpoint
        self.device = device
        self.hole_policy = hole_policy
        self.multipolygon_policy = multipolygon_policy
        self.max_polygon_points = max_polygon_points

        self.predictor = None
        self.model = None
        self._current_image_path: Optional[str] = None

    def initialize(self) -> None:
        """Load the SAM2 model. Called once on startup."""
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self._log("Initializing SAM2 model...")
        self._log(f"  Config: {self.cfg}")
        self._log(f"  Checkpoint: {self.checkpoint}")
        self._log(f"  Device: {self.device}")

        self.model = build_sam2(
            config_file=self.cfg,
            ckpt_path=self.checkpoint,
            device=self.device,
            mode='eval',
            apply_postprocessing=True,
        )
        self.predictor = SAM2ImagePredictor(self.model)
        self._log("SAM2 model initialized successfully")

    def _log(self, message: str) -> None:
        """Log to stderr (stdout is reserved for JSON responses)."""
        print(f"[SAM2] {message}", file=sys.stderr, flush=True)

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

    def handle_predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a predict command.

        Args:
            request: Dict containing:
                - image_path: Path to the image file
                - points: List of [x, y] point coordinates
                - point_labels: List of labels (1=foreground, 0=background)
                - multimask_output: Whether to return multiple masks (default: False)
                - mask_input: Optional low-res mask for refinement
        """
        import torch

        image_path = request.get("image_path")
        points = request.get("points", [])
        point_labels = request.get("point_labels", [])
        multimask_output = request.get("multimask_output", False)
        mask_input = request.get("mask_input")

        if not image_path:
            raise ValueError("image_path is required")

        if not points:
            raise ValueError("At least one point is required")

        if len(points) != len(point_labels):
            raise ValueError("points and point_labels must have same length")

        # Load image if different from cached
        if self._current_image_path != image_path:
            self._log(f"Loading image: {image_path}")
            imdata = load_image(image_path)
            self.predictor.set_image(imdata)
            self._current_image_path = image_path

        # Prepare prompts
        point_coords = np.array(points, dtype=np.float32)
        point_labels_arr = np.array(point_labels, dtype=np.int32)

        # Prepare mask input if provided
        mask_input_arr = None
        if mask_input is not None:
            mask_input_arr = np.array(mask_input, dtype=np.float32)
            if len(mask_input_arr.shape) == 2:
                mask_input_arr = mask_input_arr[None, :, :]  # Add channel dim

        # Run inference
        if self.predictor.device.type == 'cuda':
            autocast_context = torch.autocast(self.predictor.device.type, dtype=torch.bfloat16)
        else:
            autocast_context = contextlib.nullcontext()

        with torch.inference_mode(), autocast_context:
            masks, scores, low_res_masks = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels_arr,
                mask_input=mask_input_arr,
                multimask_output=multimask_output,
            )

        # Select best mask (highest score)
        if multimask_output:
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            score = float(scores[best_idx])
            low_res_mask = low_res_masks[best_idx]
        else:
            mask = masks[0]
            score = float(scores[0])
            low_res_mask = low_res_masks[0]

        # Convert mask to polygon
        polygon, bounds = mask_to_polygon(mask, self.hole_policy, self.multipolygon_policy)

        # Adaptively simplify polygon based on shape complexity
        # Simple shapes will use fewer points, complex shapes will use more (up to max)
        original_points = len(polygon) if polygon else 0
        if polygon and original_points > 4:
            polygon = adaptive_simplify_polygon(polygon, self.max_polygon_points, min_points=4)
            if len(polygon) != original_points:
                self._log(f"Adaptively simplified polygon: {original_points} -> {len(polygon)} points")

        return {
            "success": True,
            "polygon": polygon,
            "bounds": bounds,
            "score": score,
            "low_res_mask": low_res_mask.tolist(),
            "mask_shape": list(mask.shape),
        }

    def handle_set_image(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-load an image for multiple predictions."""
        image_path = request.get("image_path")
        if not image_path:
            raise ValueError("image_path is required")

        self._log(f"Pre-loading image: {image_path}")
        imdata = load_image(image_path)
        self.predictor.set_image(imdata)
        self._current_image_path = image_path

        return {
            "success": True,
            "message": f"Image loaded: {image_path}",
        }

    def handle_clear_image(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Clear the cached image."""
        self.predictor.reset_predictor()
        self._current_image_path = None
        return {
            "success": True,
            "message": "Image cache cleared",
        }

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Route request to appropriate handler."""
        command = request.get("command")

        handlers = {
            "predict": self.handle_predict,
            "set_image": self.handle_set_image,
            "clear_image": self.handle_clear_image,
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


def main():
    parser = argparse.ArgumentParser(description="SAM2 Interactive Inference Service")
    parser.add_argument(
        "--viame-path",
        default=None,
        help="Path to VIAME installation directory (used to find model checkpoint)",
    )
    parser.add_argument(
        "--cfg",
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
        help="Path to SAM2 config file (relative to sam2 package)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to SAM2 checkpoint (defaults to VIAME_PATH/configs/pipelines/models/sam2_hbp.pt)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--hole-policy",
        default="remove",
        choices=["allow", "remove"],
        help="How to handle holes in masks",
    )
    parser.add_argument(
        "--multipolygon-policy",
        default="largest",
        choices=["allow", "convex_hull", "largest"],
        help="How to handle multiple polygons",
    )
    parser.add_argument(
        "--max-polygon-points",
        type=int,
        default=25,
        help="Maximum number of points in output polygons (uses Douglas-Peucker simplification)",
    )
    args = parser.parse_args()

    # Determine checkpoint path
    checkpoint = args.checkpoint
    if checkpoint is None:
        # Try to find checkpoint in VIAME installation
        if args.viame_path:
            checkpoint = str(Path(args.viame_path) / "configs" / "pipelines" / "models" / "sam2_hbp.pt")
        else:
            # Try VIAME_INSTALL environment variable
            viame_install = os.environ.get("VIAME_INSTALL")
            if viame_install:
                checkpoint = str(Path(viame_install) / "configs" / "pipelines" / "models" / "sam2_hbp.pt")
            else:
                print("[SAM2] Error: No checkpoint specified and VIAME_INSTALL not set", file=sys.stderr)
                print("[SAM2] Use --checkpoint or --viame-path to specify model location", file=sys.stderr)
                sys.exit(1)

    # Verify checkpoint exists
    if not Path(checkpoint).exists():
        print(f"[SAM2] Error: Checkpoint not found: {checkpoint}", file=sys.stderr)
        sys.exit(1)

    service = SAM2InteractiveService(
        cfg=args.cfg,
        checkpoint=checkpoint,
        device=args.device,
        hole_policy=args.hole_policy,
        multipolygon_policy=args.multipolygon_policy,
        max_polygon_points=args.max_polygon_points,
    )

    try:
        service.initialize()
        service.run()
    except KeyboardInterrupt:
        print("[SAM2] Interrupted", file=sys.stderr)
    except Exception as e:
        print(f"[SAM2] Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
