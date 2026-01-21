# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Shared utilities for SAM3 (SAM 2.1) based algorithms.

This module provides common functionality used by sam3_tracker,
sam3_refiner, sam3_segmenter, and sam3_text_query, including:
- Shared model cache for memory efficiency
- Model initialization (SAM2, Grounding DINO)
- Text-based object detection
- SAM segmentation with box prompts
- Mask to polygon/points conversion
- IoU computation
- Configuration management
- Inference context helpers
"""

import contextlib
import os
import sys
import threading
from typing import Optional, Tuple, Any

import scriptconfig as scfg
import numpy as np


# =============================================================================
# Shared Model Cache for SAM3 Algorithms
# =============================================================================

class SharedSAM3ModelCache:
    """
    Thread-safe cache for SAM3 models to avoid loading duplicates.

    When both SAM3Segmenter and SAM3TextQuery are configured with the same
    checkpoint and device, they will share the same model instance.

    Usage:
        model, predictor = SharedSAM3ModelCache.get_or_create(
            checkpoint="/path/to/model.pt",
            model_config=None,
            device="cuda",
            logger=print
        )

        # Use the model/predictor with the lock:
        with SharedSAM3ModelCache.get_lock(checkpoint, model_config, device):
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(...)
    """

    _cache = {}  # Key: (checkpoint, model_config, device) -> (model, predictor)
    _locks = {}  # Key: (checkpoint, model_config, device) -> threading.RLock
    _global_lock = threading.Lock()

    @classmethod
    def _make_key(cls, checkpoint: Optional[str], model_config: Optional[str],
                  device: str) -> Tuple[str, str, str]:
        """Create a cache key from configuration parameters."""
        return (checkpoint or "", model_config or "", str(device))

    @classmethod
    def get_lock(cls, checkpoint: Optional[str] = None,
                 model_config: Optional[str] = None,
                 device: str = "cuda") -> threading.RLock:
        """
        Get the lock for a specific model configuration.

        Use this lock when performing inference to ensure thread safety.

        Args:
            checkpoint: Path to model checkpoint
            model_config: Path to model config JSON
            device: Device string

        Returns:
            threading.RLock for the model configuration
        """
        key = cls._make_key(checkpoint, model_config, device)
        with cls._global_lock:
            if key not in cls._locks:
                cls._locks[key] = threading.RLock()
            return cls._locks[key]

    @classmethod
    def get_or_create(
        cls,
        checkpoint: Optional[str] = None,
        model_config: Optional[str] = None,
        device: str = "cuda",
        logger=None,
    ) -> Tuple[Any, Any]:
        """
        Get or create a shared SAM3 model instance.

        If a model with the same configuration already exists in the cache,
        return it. Otherwise, create a new one and cache it.

        Args:
            checkpoint: Path to model checkpoint (or HuggingFace model ID)
            model_config: Path to model config JSON (optional)
            device: Device to run on ('cuda', 'cpu', 'auto')
            logger: Optional logging function (e.g., print)

        Returns:
            Tuple of (model, predictor)
        """
        key = cls._make_key(checkpoint, model_config, device)

        with cls._global_lock:
            if key in cls._cache:
                if logger:
                    logger(f"Using cached SAM3 model for {key}")
                return cls._cache[key]

        # Load model outside global lock (loading can take time)
        if logger:
            logger(f"Loading new SAM3 model for {key}")

        model, predictor = cls._load_model(checkpoint, model_config, device, logger)

        with cls._global_lock:
            # Double-check in case another thread loaded it while we were loading
            if key not in cls._cache:
                cls._cache[key] = (model, predictor)
                if key not in cls._locks:
                    cls._locks[key] = threading.RLock()
            return cls._cache[key]

    @classmethod
    def _load_model(
        cls,
        checkpoint: Optional[str],
        model_config: Optional[str],
        device: str,
        logger=None,
    ) -> Tuple[Any, Any]:
        """
        Load the SAM3 model.

        Returns:
            Tuple of (model, predictor)
        """
        def log(msg):
            if logger:
                logger(msg)

        # Check if using local model files
        is_local = (
            (checkpoint and os.path.exists(checkpoint)) or
            (model_config and os.path.exists(model_config))
        )

        if is_local:
            return cls._load_local_model(checkpoint, model_config, device, log)
        else:
            return cls._load_hf_model(checkpoint, device, log)

    @classmethod
    def _load_local_model(cls, checkpoint, model_config, device, log):
        """Load model from local files."""
        # Determine model directory and paths
        if checkpoint and os.path.isdir(checkpoint):
            model_dir = checkpoint
            checkpoint = os.path.join(model_dir, 'model_weights.pt')
        elif checkpoint:
            model_dir = os.path.dirname(checkpoint)
        else:
            model_dir = os.path.dirname(model_config) if model_config else None

        log(f"  Loading from local: {model_dir or checkpoint}")

        # Try native sam2 module first
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

            model = build_sam2(
                config_file=cfg,
                ckpt_path=checkpoint,
                device=device,
                mode='eval',
                apply_postprocessing=True,
            )
            predictor = SAM2ImagePredictor(model)
            log("  Loaded via native sam2 module")
            return model, predictor
        except Exception as e:
            log(f"  Native module failed: {e}")

        raise RuntimeError("Failed to load SAM3 model from local files")

    @classmethod
    def _load_hf_model(cls, checkpoint, device, log):
        """Load model from HuggingFace."""
        try:
            from sam3.model_builder import build_sam3_image_model

            model = build_sam3_image_model(
                checkpoint_path=checkpoint,
                device=device,
                eval_mode=True,
                load_from_HF=checkpoint is None,
                enable_segmentation=True,
                enable_inst_interactivity=True,
                compile=False,
            )

            predictor = model.inst_interactive_predictor
            if predictor is None:
                raise RuntimeError("Model does not have instance interactive predictor")
            log("  Loaded via sam3 module")
            return model, predictor
        except ImportError:
            pass

        # Fallback to transformers
        from transformers import Sam2Model, Sam2Processor

        model_id = checkpoint or "facebook/sam2.1-hiera-large"
        processor = Sam2Processor.from_pretrained(model_id)
        model = Sam2Model.from_pretrained(model_id).to(device)
        model.eval()

        # Create wrapper predictor
        predictor = _SharedSAM3PredictorWrapper(model, processor, device)
        log("  Loaded via transformers")
        return model, predictor

    @classmethod
    def clear(cls):
        """Clear all cached models. Useful for testing or memory cleanup."""
        with cls._global_lock:
            cls._cache.clear()
            cls._locks.clear()


class _SharedSAM3PredictorWrapper:
    """
    Wrapper to provide a SAM2-like predictor interface for HuggingFace SAM models.

    This is the shared version used by SharedSAM3ModelCache.
    """

    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        self._image_embeddings = None
        self._original_size = None
        self._inputs = None

    def set_image(self, image):
        """Set the image for prediction."""
        import torch
        from PIL import Image

        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        self._original_size = pil_image.size[::-1]  # (H, W)

        self._inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            self._image_embeddings = self.model.get_image_embeddings(self._inputs.pixel_values)

    def predict(self, point_coords=None, point_labels=None, box=None,
                mask_input=None, multimask_output=True):
        """Run prediction with the given prompts."""
        import torch

        if self._image_embeddings is None:
            raise RuntimeError("Must call set_image before predict")

        # Prepare inputs
        input_points = None
        input_labels = None

        if point_coords is not None:
            input_points = torch.tensor(point_coords, dtype=torch.float32, device=self.device)
            if input_points.ndim == 2:
                input_points = input_points.unsqueeze(0)
        if point_labels is not None:
            input_labels = torch.tensor(point_labels, dtype=torch.int64, device=self.device)
            if input_labels.ndim == 1:
                input_labels = input_labels.unsqueeze(0)

        input_boxes = None
        if box is not None:
            input_boxes = torch.tensor(box, dtype=torch.float32, device=self.device)
            if input_boxes.ndim == 1:
                input_boxes = input_boxes.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(
                image_embeddings=self._image_embeddings,
                input_points=input_points,
                input_labels=input_labels,
                input_boxes=input_boxes,
                multimask_output=multimask_output,
            )

        masks = outputs.pred_masks.squeeze(0).cpu().numpy()
        scores = outputs.iou_scores.squeeze(0).cpu().numpy()

        # Resize masks to original size if needed
        if masks.shape[-2:] != self._original_size:
            import cv2
            resized_masks = []
            for m in masks:
                resized = cv2.resize(
                    m.astype(np.float32),
                    (self._original_size[1], self._original_size[0]),
                    interpolation=cv2.INTER_LINEAR
                )
                resized_masks.append(resized > 0.5)
            masks = np.array(resized_masks)

        low_res_masks = masks

        return masks, scores, low_res_masks


class SAM3BaseConfig(scfg.DataConfig):
    """
    Base configuration for SAM3-based algorithms.

    Contains shared configuration parameters for SAM2 and Grounding DINO models.
    """
    # Model configuration
    sam_model_id = scfg.Value(
        "facebook/sam2.1-hiera-large",
        help='SAM model ID from HuggingFace, local path to weights (.pt), or directory'
    )
    model_config = scfg.Value(
        None,
        help='Path to SAM3 config JSON file (for local model loading)'
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

        # Initialize Grounding DINO (if model ID provided)
        grounding_model_id = getattr(config, 'grounding_model_id', None)
        if grounding_model_id:
            self._init_grounding_dino(grounding_model_id)

        # Check if using local SAM3 model files
        model_config = getattr(config, 'model_config', None)
        sam_model_id = config.sam_model_id

        # Determine if this is a local model (path to .pt file or directory with config)
        is_local = self._is_local_model(sam_model_id, model_config)

        if is_local:
            self._init_sam3_local(sam_model_id, model_config, use_video_predictor)
        elif use_video_predictor:
            self._init_sam2_video(sam_model_id)
        else:
            self._init_sam2_image(sam_model_id)

    def _is_local_model(self, model_id, model_config):
        """Check if the model should be loaded from local files."""
        import os
        if model_config and os.path.exists(str(model_config)):
            return True
        if model_id and os.path.exists(str(model_id)):
            return True
        return False

    def _init_sam3_local(self, weights_path, config_path, use_video_predictor=False):
        """
        Initialize SAM3 from local model files.

        Args:
            weights_path: Path to sam3_weights.pt or model directory
            config_path: Path to sam3_config.json
            use_video_predictor: If True, initialize for video prediction
        """
        import os
        import json
        import torch

        print(f"[SAM3] Loading SAM3 from local files...")
        print(f"[SAM3]   Weights: {weights_path}")
        print(f"[SAM3]   Config: {config_path}")

        # Determine model directory
        if os.path.isdir(weights_path):
            model_dir = weights_path
            weights_path = os.path.join(model_dir, 'sam3_weights.pt')
        else:
            model_dir = os.path.dirname(weights_path)

        # If config_path not provided, look for it in model directory
        if not config_path:
            config_path = os.path.join(model_dir, 'sam3_config.json')

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"SAM3 weights not found: {weights_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"SAM3 config not found: {config_path}")

        # Try to load using transformers library first
        try:
            self._init_sam3_transformers(model_dir, weights_path, config_path, use_video_predictor)
            return
        except Exception as e:
            print(f"[SAM3] Could not load via transformers: {e}")

        # Fallback: try to load using sam3 module if available
        try:
            self._init_sam3_native(weights_path, config_path, use_video_predictor)
            return
        except Exception as e:
            print(f"[SAM3] Could not load via native sam3: {e}")

        raise RuntimeError("Failed to load SAM3 model from local files")

    def _init_sam3_transformers(self, model_dir, weights_path, config_path, use_video_predictor):
        """Load SAM3 using HuggingFace transformers library."""
        import os
        import json
        import torch

        # Load config to determine model type
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        model_type = config_data.get('model_type', 'sam3_video')
        print(f"[SAM3] Model type from config: {model_type}")

        # Check if model_type is sam3_* which requires custom transformers
        if model_type.startswith('sam3'):
            # Try loading via a registered Sam3 model first
            try:
                from transformers import AutoModel, AutoConfig
                model_config = AutoConfig.from_pretrained(
                    model_dir,
                    local_files_only=True
                )
                self._sam_model = AutoModel.from_pretrained(
                    model_dir,
                    config=model_config,
                    local_files_only=True
                ).to(self._device)
                self._sam_model.eval()
                print(f"[SAM3] Successfully loaded SAM3 via transformers AutoModel")
                self._setup_predictor_interface(use_video_predictor)
                return
            except ValueError as e:
                if "Unrecognized model" in str(e):
                    print(f"[SAM3] model_type '{model_type}' not registered in transformers")
                    print(f"[SAM3] This model requires custom transformers with Sam3 support")
                else:
                    raise

        # Fallback: Try loading as Sam2 model (for sam2_* model types or as fallback)
        try:
            from transformers import Sam2Model, Sam2Processor

            # Check if there's a processor config
            processor_config_path = os.path.join(model_dir, 'sam3_processor_config.json')
            if os.path.exists(processor_config_path):
                try:
                    self._sam_processor = Sam2Processor.from_pretrained(
                        model_dir,
                        local_files_only=True
                    )
                except Exception as e:
                    print(f"[SAM3] Could not load processor: {e}")

            self._sam_model = Sam2Model.from_pretrained(
                model_dir,
                local_files_only=True
            ).to(self._device)
            self._sam_model.eval()
            print(f"[SAM3] Successfully loaded model via Sam2Model")
            self._setup_predictor_interface(use_video_predictor)
            return
        except Exception as e:
            print(f"[SAM3] Sam2Model loading failed: {e}")

        # Try standard SAM2 Video model for video predictor
        if use_video_predictor:
            try:
                from transformers import Sam2VideoModel, Sam2VideoProcessor
                self._sam_processor = Sam2VideoProcessor.from_pretrained(
                    model_dir,
                    local_files_only=True
                )
                self._sam_model = Sam2VideoModel.from_pretrained(
                    model_dir,
                    local_files_only=True
                ).to(self._device)
                self._sam_model.eval()
                print(f"[SAM3] Successfully loaded model via Sam2VideoModel")
                self._setup_predictor_interface(use_video_predictor)
                return
            except Exception as e:
                print(f"[SAM3] Sam2VideoModel loading failed: {e}")

        raise RuntimeError(
            f"Could not load SAM3 model via transformers. "
            f"Model type '{model_type}' may require custom transformers with Sam3 support."
        )

    def _setup_predictor_interface(self, use_video_predictor):
        """Set up the predictor interface from the loaded model."""
        if self._sam_model is None:
            return

        # Set up predictor interface based on model capabilities
        if hasattr(self._sam_model, 'get_image_predictor'):
            self._sam_predictor = self._sam_model.get_image_predictor()
        elif hasattr(self._sam_model, 'image_predictor'):
            self._sam_predictor = self._sam_model.image_predictor

        if use_video_predictor:
            if hasattr(self._sam_model, 'get_video_predictor'):
                self._video_predictor = self._sam_model.get_video_predictor()
            elif hasattr(self._sam_model, 'video_predictor'):
                self._video_predictor = self._sam_model.video_predictor

    def _init_sam3_native(self, weights_path, config_path, use_video_predictor):
        """Load SAM3 using native sam3 module if available."""
        try:
            from sam3.model_builder import build_sam3_video_model, build_sam3_image_model

            if use_video_predictor:
                self._video_predictor = build_sam3_video_model(
                    checkpoint_path=weights_path,
                    config_path=config_path,
                    device=str(self._device),
                    eval_mode=True,
                )
            else:
                model = build_sam3_image_model(
                    checkpoint_path=weights_path,
                    config_path=config_path,
                    device=str(self._device),
                    eval_mode=True,
                )
                if hasattr(model, 'inst_interactive_predictor'):
                    self._sam_predictor = model.inst_interactive_predictor
                else:
                    self._sam_predictor = model

            print(f"[SAM3] Successfully loaded SAM3 via native sam3 module")
        except ImportError:
            raise ImportError("sam3 module not available for native loading")

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


def parse_bool(value):
    """
    Parse a value as a boolean.

    Handles string values from KWIVER config files like 'True', 'true', '1',
    as well as actual boolean and integer values.

    Args:
        value: Value to parse (str, bool, int)

    Returns:
        bool: Parsed boolean value

    Example:
        >>> parse_bool('True')
        True
        >>> parse_bool('false')
        False
        >>> parse_bool(1)
        True
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    return bool(value)
