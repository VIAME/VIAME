# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Shared utilities for SAM3 (SAM 2.1) based algorithms.

This module provides common functionality used by both sam3_tracker,
sam3_refiner, and sam3_interactive, including:
- Model initialization (SAM2, Grounding DINO)
- Text-based object detection
- SAM segmentation with box prompts
- Mask to polygon/points conversion
- IoU computation
- Configuration management
- Inference context helpers
"""

import contextlib
import scriptconfig as scfg
import numpy as np


class SAM3BaseConfig(scfg.DataConfig):
    """
    Base configuration for SAM3-based algorithms.

    Contains shared configuration parameters for SAM2 and Grounding DINO models.
    """
    # Model configuration
    sam_model_id = scfg.Value(
        "facebook/sam2.1-hiera-large",
        help='SAM 2.1 model ID from HuggingFace or local path'
    )
    grounding_model_id = scfg.Value(
        "IDEA-Research/grounding-dino-tiny",
        help='Grounding DINO model ID for text-based detection'
    )

    # Device configuration
    device = scfg.Value('cuda', help='Device to run models on (cuda, cpu, auto)')

    # Text query configuration
    text_query = scfg.Value(
        'object',
        help='Text query describing objects. Can be comma-separated for multiple classes.'
    )

    # Detection thresholds
    detection_threshold = scfg.Value(
        0.3,
        help='Confidence threshold for text-based detections'
    )
    text_threshold = scfg.Value(
        0.25,
        help='Text matching threshold for grounding detection'
    )

    # Output configuration
    output_type = scfg.Value(
        'polygon',
        help='Type of output: "polygon" for mask contours, "points" for centroid/keypoints, "both"'
    )
    polygon_simplification = scfg.Value(
        0.01,
        help='Douglas-Peucker simplification epsilon (relative to perimeter). 0 to disable.'
    )
    num_points = scfg.Value(
        5,
        help='Number of points to output when output_type includes points'
    )

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.text_query, str):
            self.text_query_list = [q.strip() for q in self.text_query.split(',')]
        else:
            self.text_query_list = [self.text_query]


class SAM3ModelManager:
    """
    Manages SAM2 and Grounding DINO model initialization and inference.

    This class provides a shared interface for model operations used by
    both sam3_tracker and sam3_refiner.
    """

    def __init__(self):
        self._sam_predictor = None
        self._sam_model = None
        self._sam_processor = None
        self._grounding_processor = None
        self._grounding_model = None
        self._video_predictor = None
        self._device = None

    @property
    def device(self):
        return self._device

    def init_models(self, config, use_video_predictor=False):
        """
        Initialize SAM and Grounding DINO models.

        Args:
            config: Configuration object with model paths and device settings
                    (should have sam_model_id, grounding_model_id, device attributes)
            use_video_predictor: If True, initialize SAM2 video predictor
                                 instead of image predictor
        """
        from viame.pytorch.utilities import resolve_device

        self._device = resolve_device(config.device)

        # Initialize Grounding DINO
        self._init_grounding_dino(config.grounding_model_id)

        # Initialize SAM2
        if use_video_predictor:
            self._init_sam2_video(config.sam_model_id)
        else:
            self._init_sam2_image(config.sam_model_id)

    def _init_grounding_dino(self, model_id):
        """Initialize Grounding DINO for text-based detection."""
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

            self._grounding_processor = AutoProcessor.from_pretrained(model_id)
            self._grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id
            ).to(self._device)
            self._grounding_model.eval()
        except Exception as e:
            print(f"[SAM3] Warning: Could not load Grounding DINO: {e}")
            self._grounding_model = None

    def _init_sam2_image(self, model_id):
        """Initialize SAM2 for single image prediction."""
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            sam_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            model = build_sam2(
                config_file=sam_cfg,
                ckpt_path=model_id,
                device=str(self._device),
                mode='eval',
            )
            self._sam_predictor = SAM2ImagePredictor(model)
        except ImportError:
            self._init_sam2_huggingface(model_id)

    def _init_sam2_video(self, model_id):
        """Initialize SAM2 for video prediction."""
        try:
            from sam2.build_sam import build_sam2_video_predictor

            sam_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            self._video_predictor = build_sam2_video_predictor(
                sam_cfg, model_id, device=self._device
            )
        except ImportError:
            self._init_sam2_huggingface(model_id)

    def _init_sam2_huggingface(self, model_id):
        """Fallback: Initialize SAM2 via HuggingFace transformers."""
        try:
            from transformers import Sam2Model, Sam2Processor

            self._sam_processor = Sam2Processor.from_pretrained(model_id)
            self._sam_model = Sam2Model.from_pretrained(model_id).to(self._device)
            self._sam_model.eval()
        except Exception as e:
            print(f"[SAM3] Warning: Could not load SAM2: {e}")

    def detect_with_text(self, image_np, text_query_list,
                         detection_threshold, text_threshold):
        """
        Detect objects in image using text query via Grounding DINO.

        Args:
            image_np: RGB image as numpy array
            text_query_list: List of text labels to detect
            detection_threshold: Confidence threshold
            text_threshold: Text matching threshold

        Returns:
            List of (box, score, class_name) tuples where box is [x1, y1, x2, y2]
        """
        if self._grounding_model is None:
            return []

        import torch
        from PIL import Image

        pil_img = Image.fromarray(image_np)
        text_labels = [text_query_list]

        inputs = self._grounding_processor(
            images=pil_img, text=text_labels, return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            outputs = self._grounding_model(**inputs)

        results = self._grounding_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=detection_threshold,
            text_threshold=text_threshold,
            target_sizes=[pil_img.size[::-1]]
        )

        detections = []
        if len(results) > 0:
            result = results[0]
            for box, score, label in zip(
                result["boxes"], result["scores"], result["text_labels"]
            ):
                box_np = box.cpu().numpy()
                score_val = float(score.cpu().numpy())
                detections.append((box_np, score_val, label))

        return detections

    def segment_with_sam(self, image_np, boxes):
        """
        Segment objects in image using SAM with box prompts.

        Args:
            image_np: RGB image as numpy array
            boxes: List of [x1, y1, x2, y2] bounding boxes

        Returns:
            List of binary masks (numpy arrays)
        """
        if len(boxes) == 0:
            return []

        import torch

        # Use SAM2 image predictor if available
        if self._sam_predictor is not None:
            self._sam_predictor.set_image(image_np)

            prompts = {
                'box': np.array(boxes),
                'multimask_output': False
            }

            with torch.inference_mode():
                masks, scores, _ = self._sam_predictor.predict(**prompts)

            # Handle shape - ensure we have [N, 1, H, W] or similar
            if len(masks.shape) == 3:
                masks = masks[None, :, :, :]

            return [masks[i, 0] for i in range(len(boxes))]

        # Fallback to HuggingFace SAM2
        if self._sam_model is not None:
            from PIL import Image

            pil_img = Image.fromarray(image_np)
            masks = []

            for box in boxes:
                inputs = self._sam_processor(
                    images=pil_img,
                    input_boxes=[[[box[0], box[1], box[2], box[3]]]],
                    return_tensors="pt"
                ).to(self._device)

                with torch.no_grad():
                    outputs = self._sam_model(**inputs)

                mask = outputs.pred_masks[0, 0, 0].cpu().numpy() > 0
                masks.append(mask)

            return masks

        # No SAM model - return full masks
        return [np.ones((image_np.shape[0], image_np.shape[1]), dtype=bool)] * len(boxes)


def mask_to_polygon(mask, simplification=0.01):
    """
    Convert a binary mask to a KWIVER Polygon.

    Args:
        mask: Binary mask as numpy array
        simplification: Douglas-Peucker simplification epsilon (relative to perimeter)

    Returns:
        kwiver.vital.types.Polygon or None if conversion fails
    """
    import cv2
    from kwiver.vital.types import Polygon

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    contour = max(contours, key=cv2.contourArea)

    if simplification > 0:
        perimeter = cv2.arcLength(contour, True)
        epsilon = simplification * perimeter
        contour = cv2.approxPolyDP(contour, epsilon, True)

    if len(contour) < 3:
        return None

    points = contour.squeeze()
    if len(points.shape) == 1:
        return None

    polygon = Polygon()
    for point in points:
        polygon.push_back((float(point[0]), float(point[1])))

    return polygon


def mask_to_points(mask, num_points):
    """
    Extract representative points from a mask.

    Args:
        mask: Binary mask as numpy array
        num_points: Number of points to extract

    Returns:
        List of (x, y) tuples
    """
    import cv2

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return []

    contour = max(contours, key=cv2.contourArea)

    M = cv2.moments(contour)
    if M["m00"] == 0:
        return []

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    points = [(cx, cy)]

    if num_points > 1 and len(contour) > 0:
        step = max(1, len(contour) // (num_points - 1))
        for i in range(0, len(contour), step):
            if len(points) >= num_points:
                break
            pt = contour[i].squeeze()
            points.append((int(pt[0]), int(pt[1])))

    return points[:num_points]


def box_from_mask(mask):
    """
    Get bounding box from mask.

    Args:
        mask: Binary mask as numpy array

    Returns:
        kwiver.vital.types.BoundingBoxD or None if mask is empty
    """
    from kwiver.vital.types import BoundingBoxD

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    return BoundingBoxD(float(x1), float(y1), float(x2), float(y2))


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes [x1, y1, x2, y2].

    Args:
        box1: First box as [x1, y1, x2, y2]
        box2: Second box as [x1, y1, x2, y2]

    Returns:
        float: IoU value in [0, 1]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def image_to_rgb_numpy(image_container):
    """
    Convert a KWIVER image container to RGB numpy array.

    Args:
        image_container: kwiver.vital.types.ImageContainer

    Returns:
        numpy array in RGB format, uint8
    """
    img_np = image_container.image().asarray().astype('uint8')

    # Handle grayscale
    if len(img_np.shape) == 2:
        img_np = np.stack((img_np,) * 3, axis=-1)
    elif img_np.shape[2] == 1:
        img_np = np.stack((img_np[:, :, 0],) * 3, axis=-1)
    elif img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]

    # Convert BGR to RGB
    if img_np.shape[2] == 3:
        img_np = img_np[:, :, ::-1].copy()

    return img_np


def get_autocast_context(device):
    """
    Get the appropriate autocast context for the given device.

    For CUDA devices, returns a torch.autocast context with bfloat16 dtype.
    For other devices (CPU, etc.), returns a null context.

    Args:
        device: torch device or device string (e.g., 'cuda', 'cpu', torch.device('cuda:0'))

    Returns:
        Context manager for use with `with` statement

    Example:
        >>> with get_autocast_context(model.device):
        ...     output = model(input)
    """
    import torch

    # Handle both string and torch.device inputs
    if hasattr(device, 'type'):
        device_type = device.type
    else:
        device_type = str(device).split(':')[0]

    if device_type == 'cuda':
        return torch.autocast(device_type, dtype=torch.bfloat16)
    else:
        return contextlib.nullcontext()
