#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved
# BSD 3-Clause License. See either the root top-level LICENSE file or
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

"""
SAM3 Interactive Inference Service

A persistent process that keeps SAM3 loaded and handles inference requests
via stdin/stdout JSON protocol. Designed to be spawned by both Desktop (Electron)
and Web (Girder) platforms for fast interactive segmentation.

SAM3 (Segment Anything Model 3) is the next generation of Meta's SAM model,
offering improved segmentation quality and additional capabilities.

Usage:
    python sam3_interactive.py [--checkpoint CHECKPOINT] [--device DEVICE]

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


def simplify_polygon_to_max_points(
    polygon: List[List[float]],
    max_points: int = 25,
    min_tolerance: float = 0.1,
    max_tolerance: float = 100.0,
) -> List[List[float]]:
    """
    Simplify a polygon to have at most max_points vertices using Douglas-Peucker algorithm.

    Uses binary search to find the optimal tolerance value that results in
    a polygon with at most max_points vertices while preserving as much detail as possible.

    Args:
        polygon: List of [x, y] coordinate pairs
        max_points: Maximum number of points allowed in output polygon
        min_tolerance: Minimum tolerance for simplification
        max_tolerance: Maximum tolerance for simplification

    Returns:
        Simplified polygon as list of [x, y] coordinate pairs
    """
    from shapely.geometry import Polygon as ShapelyPolygon

    if len(polygon) <= max_points:
        return polygon

    # Create shapely polygon
    try:
        shape = ShapelyPolygon(polygon)
        if not shape.is_valid:
            shape = shape.buffer(0)  # Fix invalid geometries
    except Exception:
        return polygon

    # Binary search to find optimal tolerance
    low = min_tolerance
    high = max_tolerance
    best_result = polygon

    for _ in range(20):  # Max iterations for binary search
        mid = (low + high) / 2
        simplified = shape.simplify(mid, preserve_topology=True)

        if simplified.is_empty:
            high = mid
            continue

        coords = list(simplified.exterior.coords)
        num_points = len(coords)

        if num_points <= max_points:
            best_result = [[float(x), float(y)] for x, y in coords]
            high = mid  # Try to find a smaller tolerance (more detail)
        else:
            low = mid  # Need more simplification

        # Close enough
        if abs(num_points - max_points) <= 2 and num_points <= max_points:
            break

    return best_result


class SAM3InteractiveService:
    """Persistent SAM3 inference service with stdin/stdout JSON protocol."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: str = "cuda",
        hole_policy: str = "remove",
        multipolygon_policy: str = "largest",
        max_polygon_points: int = 25,
        load_from_hf: bool = True,
    ):
        self.checkpoint = checkpoint
        self.device = device
        self.hole_policy = hole_policy
        self.multipolygon_policy = multipolygon_policy
        self.max_polygon_points = max_polygon_points
        self.load_from_hf = load_from_hf

        self.predictor = None
        self.model = None
        self._current_image_path: Optional[str] = None

    def initialize(self) -> None:
        """Load the SAM3 model. Called once on startup."""
        import torch
        from sam3.model_builder import build_sam3_image_model

        self._log("Initializing SAM3 model...")
        self._log(f"  Checkpoint: {self.checkpoint or 'HuggingFace default'}")
        self._log(f"  Device: {self.device}")

        # Build SAM3 model with instance interactivity enabled (for point-based segmentation)
        self.model = build_sam3_image_model(
            checkpoint_path=self.checkpoint,
            device=self.device,
            eval_mode=True,
            load_from_HF=self.load_from_hf and self.checkpoint is None,
            enable_segmentation=True,
            enable_inst_interactivity=True,
            compile=False,  # Disable compilation for compatibility
        )

        # The instance interactive predictor is the SAM1-task predictor
        self.predictor = self.model.inst_interactive_predictor
        if self.predictor is None:
            raise RuntimeError("SAM3 model does not have instance interactive predictor enabled")

        self._log("SAM3 model initialized successfully")

    def _log(self, message: str) -> None:
        """Log to stderr (stdout is reserved for JSON responses)."""
        print(f"[SAM3] {message}", file=sys.stderr, flush=True)

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

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image from path and return as numpy array."""
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        return np.array(img)

    def _mask_to_polygon(
        self, mask: np.ndarray
    ) -> Tuple[List[List[float]], List[float]]:
        """
        Convert binary mask to polygon coordinates.

        Returns:
            polygon: List of [x, y] coordinate pairs
            bounds: [x_min, y_min, x_max, y_max]
        """
        import kwimage
        from shapely.geometry import MultiPolygon, Polygon, GeometryCollection

        # Convert to kwimage mask and then to polygon
        kw_mask = kwimage.Mask.coerce(mask.astype(np.uint8))

        # Convert mask to multi-polygon
        kw_mpoly = kw_mask.to_multi_polygon(
            pixels_are='points',
            origin_convention='center',
        )

        try:
            shape = kw_mpoly.to_shapely()
        except ValueError:
            # Workaround for issues with not enough coordinates
            new_parts = []
            for kw_poly in kw_mpoly.data:
                try:
                    new_part = kw_poly.to_shapely()
                    new_parts.append(new_part)
                except ValueError:
                    pass
            if not new_parts:
                return [], [0, 0, 0, 0]
            shape = MultiPolygon(new_parts)

        # Handle empty or invalid shapes
        if shape.is_empty:
            return [], [0, 0, 0, 0]

        # Helper function to extract a single polygon from any geometry type
        def extract_polygon(geom):
            """Extract a single Polygon from various geometry types."""
            if geom.is_empty:
                return None
            if geom.type == 'Polygon':
                return geom
            elif geom.type == 'MultiPolygon':
                if geom.geoms:
                    # Return the largest polygon
                    valid_polys = [g for g in geom.geoms if g.type == 'Polygon' and not g.is_empty]
                    if valid_polys:
                        return max(valid_polys, key=lambda p: p.area)
                return None
            elif geom.type == 'GeometryCollection':
                # Extract polygons from geometry collection
                polys = [g for g in geom.geoms if g.type == 'Polygon' and not g.is_empty]
                if polys:
                    return max(polys, key=lambda p: p.area)
                return None
            else:
                # For other types (Point, LineString, etc.), return None
                return None

        # Apply multipolygon policy
        if shape.type == 'MultiPolygon' and len(shape.geoms) > 1:
            if self.multipolygon_policy == 'convex_hull':
                shape = shape.convex_hull
            elif self.multipolygon_policy == 'largest':
                valid_polys = [g for g in shape.geoms if g.type == 'Polygon' and not g.is_empty]
                if valid_polys:
                    shape = max(valid_polys, key=lambda p: p.area)
                else:
                    return [], [0, 0, 0, 0]
            # 'allow' keeps as-is

        # Extract a single polygon from whatever shape we have
        poly = extract_polygon(shape)
        if poly is None:
            return [], [0, 0, 0, 0]

        # Apply hole policy - remove interior rings
        if self.hole_policy == 'remove':
            poly = Polygon(poly.exterior)

        # Get the exterior coordinates
        if poly.is_empty or poly.exterior is None:
            return [], [0, 0, 0, 0]

        coords = list(poly.exterior.coords)
        polygon = [[float(x), float(y)] for x, y in coords]

        # Calculate bounds
        bounds = list(poly.bounds)  # (minx, miny, maxx, maxy)

        return polygon, bounds

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
            imdata = self._load_image(image_path)
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
        polygon, bounds = self._mask_to_polygon(mask)

        # Simplify polygon to maximum number of points
        if polygon and len(polygon) > self.max_polygon_points:
            polygon = simplify_polygon_to_max_points(polygon, self.max_polygon_points)
            self._log(f"Simplified polygon to {len(polygon)} points")

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
        imdata = self._load_image(image_path)
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
    parser = argparse.ArgumentParser(description="SAM3 Interactive Inference Service")
    parser.add_argument(
        "--viame-path",
        default=None,
        help="Path to VIAME installation directory (used to find model checkpoint)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to SAM3 checkpoint (defaults to downloading from HuggingFace)",
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
    parser.add_argument(
        "--no-hf-download",
        action="store_true",
        help="Disable automatic download from HuggingFace (requires --checkpoint)",
    )
    args = parser.parse_args()

    # Determine checkpoint path
    checkpoint = args.checkpoint
    load_from_hf = not args.no_hf_download

    if checkpoint is None and args.viame_path:
        # Try to find checkpoint in VIAME installation
        viame_checkpoint = Path(args.viame_path) / "configs" / "pipelines" / "models" / "sam3.pt"
        if viame_checkpoint.exists():
            checkpoint = str(viame_checkpoint)
            load_from_hf = False

    if checkpoint is None and not load_from_hf:
        print("[SAM3] Error: No checkpoint specified and HuggingFace download disabled", file=sys.stderr)
        print("[SAM3] Use --checkpoint to specify model location or remove --no-hf-download", file=sys.stderr)
        sys.exit(1)

    # Verify checkpoint exists if specified
    if checkpoint is not None and not Path(checkpoint).exists():
        print(f"[SAM3] Error: Checkpoint not found: {checkpoint}", file=sys.stderr)
        sys.exit(1)

    service = SAM3InteractiveService(
        checkpoint=checkpoint,
        device=args.device,
        hole_policy=args.hole_policy,
        multipolygon_policy=args.multipolygon_policy,
        max_polygon_points=args.max_polygon_points,
        load_from_hf=load_from_hf,
    )

    try:
        service.initialize()
        service.run()
    except KeyboardInterrupt:
        print("[SAM3] Interrupted", file=sys.stderr)
    except Exception as e:
        print(f"[SAM3] Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
