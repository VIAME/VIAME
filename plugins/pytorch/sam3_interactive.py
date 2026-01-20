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
offering improved segmentation quality and additional capabilities including
text-based queries for open-vocabulary detection and segmentation.

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

    Text Query Input:
    {
        "id": "unique-request-id",
        "command": "text_query",
        "image_path": "/path/to/frame.png",
        "text": "fish swimming near coral",
        "box_threshold": 0.3,  // confidence threshold for detections
        "max_detections": 10,  // maximum number of detections to return
        // Optional refinement inputs:
        "boxes": [[x1, y1, x2, y2], ...],  // boxes to refine
        "points": [[x, y], ...],  // keypoints
        "point_labels": [1, 0, ...],  // point labels
        "masks": [...]  // masks to refine
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

    Text Query Output:
    {
        "id": "unique-request-id",
        "success": true,
        "detections": [
            {
                "box": [x1, y1, x2, y2],
                "polygon": [[x1, y1], ...],
                "score": 0.95,
                "label": "fish"
            },
            ...
        ]
    }

    Commands:
    - "predict": Run point-based segmentation on an image
    - "text_query": Run text-based detection/segmentation on an image
    - "refine": Refine existing detections with additional prompts
    - "set_image": Pre-load an image for multiple predictions
    - "clear_image": Clear the cached image
    - "shutdown": Gracefully terminate the service
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from viame.core.segmentation_utils import (
    load_image,
    mask_to_polygon,
    simplify_polygon_to_max_points,
)
from viame.pytorch.sam3_utilities import get_autocast_context
from viame.pytorch.utilities import resolve_device


class SAM3InteractiveService:
    """Persistent SAM3 inference service with stdin/stdout JSON protocol."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        model_config: Optional[str] = None,
        device: str = "cuda",
        hole_policy: str = "remove",
        multipolygon_policy: str = "largest",
        max_polygon_points: int = 25,
        load_from_hf: bool = True,
    ):
        self.checkpoint = checkpoint
        self.model_config = model_config
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
        import os
        import torch

        self._log("Initializing SAM3 model...")
        self._log(f"  Checkpoint: {self.checkpoint or 'HuggingFace default'}")
        self._log(f"  Model config: {self.model_config or 'auto'}")
        self._log(f"  Device: {self.device}")

        # Check if using local model files
        is_local = (
            (self.checkpoint and os.path.exists(self.checkpoint)) or
            (self.model_config and os.path.exists(self.model_config))
        )

        if is_local:
            self._initialize_local_model()
        else:
            self._initialize_hf_model()

        self._log("SAM3 model initialized successfully")

    def _initialize_local_model(self) -> None:
        """Initialize SAM3 from local model files."""
        import os
        import json
        import torch

        checkpoint = self.checkpoint
        config_path = self.model_config

        # Determine model directory and paths
        if checkpoint and os.path.isdir(checkpoint):
            model_dir = checkpoint
            checkpoint = os.path.join(model_dir, 'sam3_weights.pt')
        elif checkpoint:
            model_dir = os.path.dirname(checkpoint)
        else:
            model_dir = os.path.dirname(config_path) if config_path else None

        if not config_path and model_dir:
            config_path = os.path.join(model_dir, 'sam3_config.json')

        self._log(f"  Loading from local: {model_dir}")

        # Try transformers AutoModel first
        try:
            from transformers import AutoModel, AutoProcessor, AutoConfig

            processor_config = os.path.join(model_dir, 'sam3_processor_config.json')
            if os.path.exists(processor_config):
                self._processor = AutoProcessor.from_pretrained(
                    model_dir, local_files_only=True
                )

            model_config = AutoConfig.from_pretrained(
                model_dir, local_files_only=True
            )

            self.model = AutoModel.from_pretrained(
                model_dir,
                config=model_config,
                local_files_only=True
            ).to(self.device)
            self.model.eval()

            # Get the predictor interface
            if hasattr(self.model, 'get_image_predictor'):
                self.predictor = self.model.get_image_predictor()
            elif hasattr(self.model, 'inst_interactive_predictor'):
                self.predictor = self.model.inst_interactive_predictor
            elif hasattr(self.model, 'image_predictor'):
                self.predictor = self.model.image_predictor
            else:
                # Use the model directly with a wrapper
                self.predictor = SAM3ImagePredictorWrapper(self.model, self._processor, self.device)

            self._log("  Loaded via transformers AutoModel")
            return
        except Exception as e:
            self._log(f"  AutoModel failed: {e}")

        # Try native sam3 module
        try:
            from sam3.model_builder import build_sam3_image_model

            self.model = build_sam3_image_model(
                checkpoint_path=checkpoint,
                config_path=config_path,
                device=self.device,
                eval_mode=True,
                enable_segmentation=True,
                enable_inst_interactivity=True,
                compile=False,
            )

            self.predictor = self.model.inst_interactive_predictor
            if self.predictor is None:
                raise RuntimeError("SAM3 model does not have instance interactive predictor")

            self._log("  Loaded via native sam3 module")
            return
        except Exception as e:
            self._log(f"  Native sam3 failed: {e}")

        raise RuntimeError("Failed to load SAM3 model from local files")

    def _initialize_hf_model(self) -> None:
        """Initialize SAM3 from HuggingFace."""
        import torch

        try:
            from sam3.model_builder import build_sam3_image_model

            self.model = build_sam3_image_model(
                checkpoint_path=self.checkpoint,
                device=self.device,
                eval_mode=True,
                load_from_HF=self.load_from_hf and self.checkpoint is None,
                enable_segmentation=True,
                enable_inst_interactivity=True,
                compile=False,
            )

            self.predictor = self.model.inst_interactive_predictor
            if self.predictor is None:
                raise RuntimeError("SAM3 model does not have instance interactive predictor")
        except ImportError:
            # Fallback to transformers
            from transformers import Sam2Model, Sam2Processor

            model_id = self.checkpoint or "facebook/sam2.1-hiera-large"
            self._processor = Sam2Processor.from_pretrained(model_id)
            self.model = Sam2Model.from_pretrained(model_id).to(self.device)
            self.model.eval()
            self.predictor = SAM3ImagePredictorWrapper(self.model, self._processor, self.device)


class SAM3ImagePredictorWrapper:
    """
    Wrapper to provide a SAM2-like predictor interface for HuggingFace SAM3 models.
    """

    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        self._image_embeddings = None
        self._original_size = None

    def set_image(self, image):
        """Set the image for prediction."""
        import torch
        from PIL import Image

        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        self._original_size = pil_image.size[::-1]  # (H, W)

        # Process image
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            self._image_embeddings = self.model.get_image_embeddings(inputs.pixel_values)

        self._inputs = inputs

    def reset_predictor(self):
        """Reset the predictor state."""
        self._image_embeddings = None
        self._original_size = None

    def predict(
        self,
        point_coords=None,
        point_labels=None,
        box=None,
        mask_input=None,
        multimask_output=False,
    ):
        """Run prediction with the given prompts."""
        import torch

        if self._image_embeddings is None:
            raise RuntimeError("Must call set_image before predict")

        # Prepare inputs
        input_points = None
        input_labels = None
        input_boxes = None

        if point_coords is not None:
            input_points = torch.tensor(point_coords, dtype=torch.float32).unsqueeze(0).to(self.device)
            if len(input_points.shape) == 2:
                input_points = input_points.unsqueeze(0)

        if point_labels is not None:
            input_labels = torch.tensor(point_labels, dtype=torch.int64).unsqueeze(0).to(self.device)
            if len(input_labels.shape) == 1:
                input_labels = input_labels.unsqueeze(0)

        if box is not None:
            boxes = np.atleast_2d(box)
            input_boxes = torch.tensor(boxes, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                image_embeddings=self._image_embeddings,
                input_points=input_points,
                input_labels=input_labels,
                input_boxes=input_boxes,
                multimask_output=multimask_output,
            )

        # Process outputs
        masks = outputs.pred_masks.squeeze(0).cpu().numpy()
        scores = outputs.iou_scores.squeeze(0).cpu().numpy()

        # Ensure masks are binary
        masks = (masks > 0).astype(np.uint8)

        # Low-res masks for refinement
        low_res_masks = masks  # Simplified - actual implementation would get low-res

        return masks, scores, low_res_masks

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
        autocast_context = get_autocast_context(self.predictor.device)

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

    def handle_text_query(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a text-based query command for open-vocabulary detection/segmentation.

        Args:
            request: Dict containing:
                - image_path: Path to the image file
                - text: Text query describing what to find (e.g., "fish", "person swimming")
                - box_threshold: Confidence threshold for detections (default: 0.3)
                - max_detections: Maximum number of detections to return (default: 10)
                - boxes: Optional list of [x1, y1, x2, y2] boxes to refine
                - points: Optional list of [x, y] keypoints for refinement
                - point_labels: Optional list of labels for points (1=foreground, 0=background)
                - masks: Optional masks for refinement

        Returns:
            Dict with detections, each containing box, polygon, score, and label
        """
        import torch

        image_path = request.get("image_path")
        text = request.get("text", "")
        box_threshold = request.get("box_threshold", 0.3)
        max_detections = request.get("max_detections", 10)

        # Optional refinement inputs
        boxes = request.get("boxes")
        points = request.get("points")
        point_labels = request.get("point_labels")
        masks = request.get("masks")

        if not image_path:
            raise ValueError("image_path is required")

        if not text:
            raise ValueError("text query is required")

        # Load image if different from cached
        if self._current_image_path != image_path:
            self._log(f"Loading image: {image_path}")
            imdata = load_image(image_path)
            self._current_image_path = image_path
        else:
            imdata = load_image(image_path)

        self._log(f"Text query: '{text}' (threshold={box_threshold}, max={max_detections})")

        # Check if model has text query capability
        if not hasattr(self.model, 'predict_with_text') and not hasattr(self.model, 'text_predictor'):
            # Fallback: use automatic mask generation and return top results
            self._log("Model does not have native text query support, using auto-mask generation")
            return self._handle_text_query_fallback(imdata, text, box_threshold, max_detections)

        # Use native text query if available
        autocast_context = get_autocast_context(self.predictor.device)

        detections = []
        with torch.inference_mode(), autocast_context:
            # Try different API patterns that SAM3 might support
            if hasattr(self.model, 'predict_with_text'):
                # Direct text prediction API
                results = self.model.predict_with_text(
                    image=imdata,
                    text=text,
                    box_threshold=box_threshold,
                )
                detections = self._process_text_results(results, text, max_detections)

            elif hasattr(self.model, 'text_predictor'):
                # Separate text predictor
                text_pred = self.model.text_predictor
                text_pred.set_image(imdata)
                results = text_pred.predict(text=text, threshold=box_threshold)
                detections = self._process_text_results(results, text, max_detections)

        return {
            "success": True,
            "detections": detections,
            "query": text,
        }

    def _handle_text_query_fallback(
        self,
        imdata: np.ndarray,
        text: str,
        box_threshold: float,
        max_detections: int,
    ) -> Dict[str, Any]:
        """
        Fallback for models without native text query support.
        Uses automatic mask generation and returns results labeled with the query text.
        """
        import torch

        # Set image for prediction
        self.predictor.set_image(imdata)

        autocast_context = get_autocast_context(self.predictor.device)

        detections = []

        # Try to use automatic mask generator if available
        if hasattr(self.model, 'mask_generator') or hasattr(self.model, 'generate_masks'):
            with torch.inference_mode(), autocast_context:
                if hasattr(self.model, 'mask_generator'):
                    masks_data = self.model.mask_generator.generate(imdata)
                else:
                    masks_data = self.model.generate_masks(imdata)

                # Process each mask
                for i, mask_info in enumerate(masks_data[:max_detections]):
                    if isinstance(mask_info, dict):
                        mask = mask_info.get('segmentation', mask_info.get('mask'))
                        score = mask_info.get('predicted_iou', mask_info.get('score', 0.5))
                        bbox = mask_info.get('bbox')  # [x, y, w, h] format typically
                    else:
                        mask = mask_info
                        score = 0.5
                        bbox = None

                    if mask is None:
                        continue

                    # Convert mask to polygon
                    polygon, bounds = mask_to_polygon(mask, self.hole_policy, self.multipolygon_policy)

                    if not polygon:
                        continue

                    # Simplify polygon
                    if len(polygon) > self.max_polygon_points:
                        polygon = simplify_polygon_to_max_points(polygon, self.max_polygon_points)

                    # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2] if needed
                    if bbox is not None:
                        if len(bbox) == 4:
                            x, y, w, h = bbox
                            box = [float(x), float(y), float(x + w), float(y + h)]
                        else:
                            box = [float(b) for b in bbox]
                    else:
                        box = bounds

                    detections.append({
                        "box": box,
                        "polygon": polygon,
                        "score": float(score),
                        "label": text,  # Label with the query text
                    })

        return {
            "success": True,
            "detections": detections,
            "query": text,
            "fallback": True,  # Indicate this used fallback method
        }

    def _process_text_results(
        self,
        results: Any,
        text: str,
        max_detections: int,
    ) -> List[Dict[str, Any]]:
        """Process results from text-based prediction into standardized format."""
        detections = []

        # Handle different result formats
        if isinstance(results, dict):
            boxes = results.get('boxes', results.get('bboxes', []))
            masks = results.get('masks', results.get('segmentations', []))
            scores = results.get('scores', results.get('confidences', []))
            labels = results.get('labels', results.get('classes', [text] * len(boxes)))
        elif isinstance(results, (list, tuple)) and len(results) >= 2:
            boxes, masks = results[0], results[1]
            scores = results[2] if len(results) > 2 else [0.5] * len(boxes)
            labels = results[3] if len(results) > 3 else [text] * len(boxes)
        else:
            return detections

        for i in range(min(len(boxes), max_detections)):
            box = boxes[i] if i < len(boxes) else None
            mask = masks[i] if i < len(masks) else None
            score = scores[i] if i < len(scores) else 0.5
            label = labels[i] if i < len(labels) else text

            # Convert box to list format
            if box is not None:
                if hasattr(box, 'tolist'):
                    box = box.tolist()
                box = [float(b) for b in box]

            # Convert mask to polygon
            polygon = []
            bounds = box or [0, 0, 0, 0]
            if mask is not None:
                if hasattr(mask, 'numpy'):
                    mask = mask.numpy()
                polygon, bounds = mask_to_polygon(mask, self.hole_policy, self.multipolygon_policy)
                if polygon and len(polygon) > self.max_polygon_points:
                    polygon = simplify_polygon_to_max_points(polygon, self.max_polygon_points)

            # Convert score
            if hasattr(score, 'item'):
                score = score.item()

            # Convert label
            if hasattr(label, 'item'):
                label = str(label.item())
            elif not isinstance(label, str):
                label = str(label)

            detections.append({
                "box": box or bounds,
                "polygon": polygon,
                "score": float(score),
                "label": label,
            })

        return detections

    def handle_refine(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine existing detections with additional prompts (boxes, points, masks, labels).

        Args:
            request: Dict containing:
                - image_path: Path to the image file
                - detections: List of detections to refine, each with box/polygon/label
                - points: Optional additional keypoints for refinement
                - point_labels: Labels for additional points
                - refine_masks: Whether to refine masks (default: True)
        """
        import torch

        image_path = request.get("image_path")
        input_detections = request.get("detections", [])
        points = request.get("points", [])
        point_labels = request.get("point_labels", [])
        refine_masks = request.get("refine_masks", True)

        if not image_path:
            raise ValueError("image_path is required")

        if not input_detections:
            raise ValueError("At least one detection is required for refinement")

        # Load image if different from cached
        if self._current_image_path != image_path:
            self._log(f"Loading image: {image_path}")
            imdata = load_image(image_path)
            self.predictor.set_image(imdata)
            self._current_image_path = image_path

        autocast_context = get_autocast_context(self.predictor.device)

        refined_detections = []

        with torch.inference_mode(), autocast_context:
            for det in input_detections:
                box = det.get("box")
                label = det.get("label", "object")
                existing_score = det.get("score", 0.5)

                # Prepare prompts for this detection
                prompt_kwargs = {}

                if box:
                    prompt_kwargs["box"] = np.array(box, dtype=np.float32)

                # Add any additional points
                if points and point_labels:
                    prompt_kwargs["point_coords"] = np.array(points, dtype=np.float32)
                    prompt_kwargs["point_labels"] = np.array(point_labels, dtype=np.int32)

                # Run prediction with box prompt
                if prompt_kwargs:
                    try:
                        masks, scores, low_res_masks = self.predictor.predict(
                            **prompt_kwargs,
                            multimask_output=False,
                        )

                        mask = masks[0]
                        score = float(scores[0])

                        # Convert mask to polygon
                        polygon, bounds = mask_to_polygon(mask, self.hole_policy, self.multipolygon_policy)

                        if polygon and len(polygon) > self.max_polygon_points:
                            polygon = simplify_polygon_to_max_points(polygon, self.max_polygon_points)

                        refined_detections.append({
                            "box": bounds if not box else box,
                            "polygon": polygon,
                            "score": score,
                            "label": label,
                            "low_res_mask": low_res_masks[0].tolist() if refine_masks else None,
                        })
                    except Exception as e:
                        self._log(f"Error refining detection: {e}")
                        # Keep original detection if refinement fails
                        refined_detections.append(det)
                else:
                    # No prompts, keep original
                    refined_detections.append(det)

        return {
            "success": True,
            "detections": refined_detections,
        }

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Route request to appropriate handler."""
        command = request.get("command")

        handlers = {
            "predict": self.handle_predict,
            "text_query": self.handle_text_query,
            "refine": self.handle_refine,
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
        help="Path to SAM3 checkpoint/weights (defaults to downloading from HuggingFace)",
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="Path to SAM3 config JSON file (for local model loading)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run inference on (auto selects best available)",
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

    # Determine checkpoint and config paths
    checkpoint = args.checkpoint
    model_config = args.model_config
    load_from_hf = not args.no_hf_download

    if checkpoint is None and args.viame_path:
        # Try to find SAM3 model files in VIAME installation
        viame_models = Path(args.viame_path) / "configs" / "pipelines" / "models"

        # Check for SAM3 weights
        sam3_weights = viame_models / "sam3_weights.pt"
        sam3_config = viame_models / "sam3_config.json"

        if sam3_weights.exists():
            checkpoint = str(sam3_weights)
            load_from_hf = False
            print(f"[SAM3] Found SAM3 weights: {checkpoint}", file=sys.stderr)

            if sam3_config.exists() and model_config is None:
                model_config = str(sam3_config)
                print(f"[SAM3] Found SAM3 config: {model_config}", file=sys.stderr)
        else:
            # Fallback to older naming
            sam3_pt = viame_models / "sam3.pt"
            if sam3_pt.exists():
                checkpoint = str(sam3_pt)
                load_from_hf = False

    if checkpoint is None and not load_from_hf:
        print("[SAM3] Error: No checkpoint specified and HuggingFace download disabled", file=sys.stderr)
        print("[SAM3] Use --checkpoint to specify model location or remove --no-hf-download", file=sys.stderr)
        sys.exit(1)

    # Verify checkpoint exists if specified
    if checkpoint is not None and not Path(checkpoint).exists():
        print(f"[SAM3] Error: Checkpoint not found: {checkpoint}", file=sys.stderr)
        sys.exit(1)

    # Resolve device (handles "auto" and returns actual device string)
    device = str(resolve_device(args.device))

    service = SAM3InteractiveService(
        checkpoint=checkpoint,
        model_config=model_config,
        device=device,
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
