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
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


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
        hole_policy: str = "remove",
        multipolygon_policy: str = "largest",
        max_polygon_points: int = 25,
        adaptive_simplify: bool = False,
    ):
        """
        Initialize the service with configured algorithms.

        Args:
            segment_via_points_algo: Configured SegmentViaPoints algorithm instance
            perform_text_query_algo: Optional configured PerformTextQuery algorithm
            hole_policy: How to handle holes in masks ('allow' or 'remove')
            multipolygon_policy: How to handle multiple polygons ('allow', 'convex_hull', 'largest')
            max_polygon_points: Maximum number of points in output polygons
            adaptive_simplify: Use adaptive polygon simplification
        """
        self._segment_algo = segment_via_points_algo
        self._text_query_algo = perform_text_query_algo
        self._hole_policy = hole_policy
        self._multipolygon_policy = multipolygon_policy
        self._max_polygon_points = max_polygon_points
        self._adaptive_simplify = adaptive_simplify
        self._current_image_path: Optional[str] = None
        self._current_image_container = None

    def _log(self, message: str) -> None:
        """Log to stderr (stdout is reserved for JSON responses)."""
        print(f"[SegmentationService] {message}", file=sys.stderr, flush=True)

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

    def _load_image(self, image_path: str):
        """Load an image and return a vital ImageContainer."""
        from kwiver.vital.types.types import ImageContainer, Image

        from viame.core.segmentation_utils import load_image

        imdata = load_image(image_path)
        return ImageContainer(Image(imdata))

    def _detections_to_response(self, detected_objects) -> List[Dict[str, Any]]:
        """Convert DetectedObjectSet to response dictionaries."""
        from viame.core.segmentation_utils import (
            mask_to_polygon,
            simplify_polygon_to_max_points,
            adaptive_simplify_polygon,
        )

        results = []

        for det_obj in detected_objects:
            bbox = det_obj.bounding_box()
            bounds = [bbox.min_x(), bbox.min_y(), bbox.max_x(), bbox.max_y()]
            score = det_obj.confidence()

            # Get polygon from mask if available
            polygon = None
            rle_mask = None
            mask_shape = None

            if det_obj.mask is not None:
                mask = det_obj.mask.image().asarray()
                if mask is not None and mask.size > 0:
                    # Convert to binary mask
                    if mask.ndim == 3:
                        mask = mask[:, :, 0]
                    mask = (mask > 0).astype(np.uint8)

                    polygon, _ = mask_to_polygon(
                        mask, self._hole_policy, self._multipolygon_policy
                    )

                    # Simplify polygon if needed
                    if polygon and len(polygon) > self._max_polygon_points:
                        original_points = len(polygon)
                        if self._adaptive_simplify:
                            polygon = adaptive_simplify_polygon(
                                polygon, self._max_polygon_points, min_points=4
                            )
                        else:
                            polygon = simplify_polygon_to_max_points(
                                polygon, self._max_polygon_points
                            )
                        if len(polygon) != original_points:
                            self._log(f"Simplified polygon: {original_points} -> {len(polygon)} points")

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

            if rle_mask is not None:
                result["rle_mask"] = rle_mask
                result["mask_shape"] = mask_shape

            # Get class label if available
            if det_obj.type() is not None:
                result["label"] = det_obj.type().get_most_likely_class()

            results.append(result)

        return results

    def handle_predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a predict command using SegmentViaPoints algorithm."""
        from kwiver.vital.types import Point2d

        image_path = request.get("image_path")
        points = request.get("points", [])
        point_labels = request.get("point_labels", [])

        if not image_path:
            raise ValueError("image_path is required")
        if not points:
            raise ValueError("At least one point is required")
        if len(points) != len(point_labels):
            raise ValueError("points and point_labels must have same length")

        # Load image if different from cached
        if self._current_image_path != image_path:
            self._log(f"Loading image: {image_path}")
            self._current_image_container = self._load_image(image_path)
            self._current_image_path = image_path

        # Convert points to vital Point2d objects
        vital_points = [Point2d(float(p[0]), float(p[1])) for p in points]
        vital_labels = [int(label) for label in point_labels]

        # Run segmentation
        detected_objects = self._segment_algo.segment(
            self._current_image_container,
            vital_points,
            vital_labels
        )

        # Convert results
        results = self._detections_to_response(detected_objects)

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

    def handle_text_query(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a text_query command using PerformTextQuery algorithm."""
        if self._text_query_algo is None:
            raise ValueError("Text query not configured")

        from kwiver.vital.types import Timestamp

        image_path = request.get("image_path")
        text = request.get("text", "")

        if not image_path:
            raise ValueError("image_path is required")
        if not text:
            raise ValueError("text query is required")

        # Load image
        image_container = self._load_image(image_path)

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
                    bbox = det_obj.bounding_box()
                    bounds = [bbox.min_x(), bbox.min_y(), bbox.max_x(), bbox.max_y()]
                    score = det_obj.confidence()

                    detection = {
                        "bounds": bounds,
                        "score": score,
                        "track_id": track.id(),
                    }

                    if det_obj.type() is not None:
                        detection["label"] = det_obj.type().get_most_likely_class()

                    # Get polygon from mask if available
                    if det_obj.mask is not None:
                        mask = det_obj.mask.image().asarray()
                        if mask is not None and mask.size > 0:
                            from viame.core.segmentation_utils import mask_to_polygon
                            if mask.ndim == 3:
                                mask = mask[:, :, 0]
                            mask = (mask > 0).astype(np.uint8)
                            polygon, _ = mask_to_polygon(
                                mask, self._hole_policy, self._multipolygon_policy
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
        if not image_path:
            raise ValueError("image_path is required")

        self._log(f"Pre-loading image: {image_path}")
        self._current_image_container = self._load_image(image_path)
        self._current_image_path = image_path

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

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Route request to appropriate handler."""
        command = request.get("command")

        handlers = {
            "predict": self.handle_predict,
            "set_image": self.handle_set_image,
            "clear_image": self.handle_clear_image,
        }

        # Add text_query handler if algorithm is configured
        if self._text_query_algo is not None:
            handlers["text_query"] = self.handle_text_query

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
                import traceback
                traceback.print_exc(file=sys.stderr)
                self._send_error(request_id, str(e))

        self._log("Service shutting down")


def load_algorithms_from_config(config_path: str, plugin_paths: List[str] = None):
    """
    Load and configure algorithms from a KWIVER config file.

    Args:
        config_path: Path to the config file
        plugin_paths: Optional list of additional plugin paths to load

    Returns:
        Tuple of (segment_via_points_algo, perform_text_query_algo, service_config)
    """
    from kwiver.vital.algo import SegmentViaPoints, PerformTextQuery
    from kwiver.vital.config import config as vital_config
    from kwiver.vital.modules import modules as vital_modules

    # Load plugin modules
    vital_modules.load_known_modules()

    if plugin_paths:
        for path in plugin_paths:
            if os.path.isdir(path):
                vital_modules.load_module(path)

    # Read config file
    cfg = vital_config.empty_config()

    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                cfg.set_value(key.strip(), value.strip())

    # Create segment_via_points algorithm
    segment_algo = None
    if cfg.has_value("segment_via_points:type"):
        impl_name = cfg.get_value("segment_via_points:type")
        segment_algo = SegmentViaPoints.create(impl_name)
        segment_algo.set_configuration(cfg.subblock("segment_via_points:" + impl_name))

    # Create perform_text_query algorithm (optional)
    text_query_algo = None
    if cfg.has_value("perform_text_query:type"):
        impl_name = cfg.get_value("perform_text_query:type")
        text_query_algo = PerformTextQuery.create(impl_name)
        text_query_algo.set_configuration(cfg.subblock("perform_text_query:" + impl_name))

    # Extract service configuration
    service_config = {
        "hole_policy": cfg.get_value("service:hole_policy") if cfg.has_value("service:hole_policy") else "remove",
        "multipolygon_policy": cfg.get_value("service:multipolygon_policy") if cfg.has_value("service:multipolygon_policy") else "largest",
        "max_polygon_points": int(cfg.get_value("service:max_polygon_points")) if cfg.has_value("service:max_polygon_points") else 25,
        "adaptive_simplify": cfg.get_value("service:adaptive_simplify").lower() in ('true', '1', 'yes') if cfg.has_value("service:adaptive_simplify") else False,
    }

    return segment_algo, text_query_algo, service_config


def find_viame_config(model_type: str = "sam3") -> Optional[str]:
    """
    Find the default segmentation config file in VIAME install.

    Args:
        model_type: Type of model ('sam2' or 'sam3')

    Returns:
        Path to config file if found, None otherwise
    """
    viame_install = os.environ.get("VIAME_INSTALL")
    if not viame_install:
        return None

    pipelines_dir = Path(viame_install) / "configs" / "pipelines"

    # Map model type to config file
    config_files = {
        "sam2": "common_sam2_segmenter.conf",
        "sam3": "common_sam3_segmenter.conf",
    }

    config_name = config_files.get(model_type)
    if config_name:
        config_path = pipelines_dir / config_name
        if config_path.exists():
            return str(config_path)

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

# Point-based segmentation algorithm
segment_via_points:type = sam2
segment_via_points:sam2:checkpoint =
segment_via_points:sam2:cfg = configs/sam2.1/sam2.1_hiera_b+.yaml
segment_via_points:sam2:device = cuda

# Service settings
service:hole_policy = remove
service:multipolygon_policy = largest
service:max_polygon_points = 25
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
service:hole_policy = remove
service:multipolygon_policy = largest
service:max_polygon_points = 25
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
    args = parser.parse_args()

    # Handle config generation
    if args.generate_config:
        create_default_config(args.generate_config, args.model)
        return

    # Require config file
    if not args.config:
        parser.error("--config is required (or use --generate-config to create one)")

    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    try:
        # Load algorithms from config
        segment_algo, text_query_algo, service_config = load_algorithms_from_config(
            args.config, args.plugin_path
        )

        if segment_algo is None:
            print("Error: No segment_via_points algorithm configured", file=sys.stderr)
            sys.exit(1)

        # Create and run service
        service = InteractiveSegmentationService(
            segment_via_points_algo=segment_algo,
            perform_text_query_algo=text_query_algo,
            **service_config
        )
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
