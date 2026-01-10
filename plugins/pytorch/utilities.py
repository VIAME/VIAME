# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Utility functions and base classes for VIAME PyTorch plugins.

This module provides:
- Device resolution utilities
- Base classes for detectors and trainers
- Detection format conversion utilities (kwimage <-> kwiver)
- Configuration management helpers
- Image processing utilities
- File utilities
"""

from __future__ import print_function

import os
import shutil

import numpy as np

# Lazy imports to avoid circular dependencies
# kwiver imports are done inside functions


# =============================================================================
# Image Processing Utilities
# =============================================================================

def pad_img_to_fit_bbox(img, x1, y1, x2, y2):
    """
    Pad an image with zeros so that a bounding box fits within it.

    Args:
        img: Input image (numpy array with shape [H, W, C])
        x1: Left coordinate of bounding box
        y1: Top coordinate of bounding box
        x2: Right coordinate of bounding box
        y2: Bottom coordinate of bounding box

    Returns:
        tuple: (padded_img, new_x1, new_x2, new_y1, new_y2)
    """
    import cv2

    img = cv2.copyMakeBorder(
        img,
        -min(0, y1),
        max(y2 - img.shape[0], 0),
        -min(0, x1),
        max(x2 - img.shape[1], 0),
        cv2.BORDER_CONSTANT,
    )

    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)

    return img, x1, x2, y1, y2


def safe_crop(img, x1, y1, x2, y2):
    """
    Safely crop an image, padding if the bounding box extends outside the image.

    Args:
        img: Input image (numpy array with shape [H, W, C])
        x1: Left coordinate of crop region
        y1: Top coordinate of crop region
        x2: Right coordinate of crop region
        y2: Bottom coordinate of crop region

    Returns:
        numpy.ndarray: Cropped image region
    """
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, y1, x2, y2)
    return img[y1:y2, x1:x2, :]


# =============================================================================
# File Utilities
# =============================================================================

def recurse_copy(src, dst, max_depth=10, ignore=".json"):
    """
    Recursively copy files from source to destination.

    Args:
        src: Source path (file or directory)
        dst: Destination directory
        max_depth: Maximum recursion depth (default 10)
        ignore: File extension to ignore (default ".json")

    Returns:
        str: The source path
    """
    if max_depth < 0:
        return src
    if os.path.isdir(src):
        for entry in os.listdir(src):
            recurse_copy(os.path.join(src, entry), dst, max_depth - 1, ignore)
    elif not src.endswith(ignore):
        shutil.copy2(src, dst)


# =============================================================================
# Device Utilities
# =============================================================================

def resolve_device(device_spec):
    """
    Resolve a device specification to an actual device.

    Handles 'auto' to automatically select CUDA if available, otherwise CPU.

    Args:
        device_spec: Device specification string. Can be:
            - 'auto': Automatically select CUDA if available, else CPU
            - 'cpu': Use CPU
            - 'cuda': Use default CUDA device
            - 'cuda:N': Use specific CUDA device N
            - int: Use CUDA device with that index
            - torch.device: Return as-is

    Returns:
        torch.device or str: Resolved device specification
    """
    import torch

    if device_spec == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        return torch.device('cpu')

    if isinstance(device_spec, int):
        if torch.cuda.is_available():
            return torch.device(f'cuda:{device_spec}')
        return torch.device('cpu')

    if isinstance(device_spec, str):
        if device_spec in ('cpu', 'cuda'):
            return torch.device(device_spec)
        if device_spec.startswith('cuda:'):
            return torch.device(device_spec)
        # Try to parse as integer
        try:
            idx = int(device_spec)
            if torch.cuda.is_available():
                return torch.device(f'cuda:{idx}')
            return torch.device('cpu')
        except ValueError:
            pass

    # Return as-is if already a torch.device or unrecognized
    return device_spec


def resolve_device_str(device_spec):
    """
    Resolve a device specification to a string.

    Similar to resolve_device but returns a string instead of torch.device.
    Useful for libraries that expect string device specs.

    Args:
        device_spec: Device specification (see resolve_device)

    Returns:
        str: Device string like 'cpu', 'cuda', or 'cuda:0'
    """
    import torch

    if device_spec == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

    if isinstance(device_spec, int):
        if torch.cuda.is_available():
            return f'cuda:{device_spec}'
        return 'cpu'

    if isinstance(device_spec, torch.device):
        return str(device_spec)

    if isinstance(device_spec, str):
        if device_spec in ('cpu', 'cuda'):
            return device_spec
        if device_spec.startswith('cuda:'):
            return device_spec
        try:
            idx = int(device_spec)
            if torch.cuda.is_available():
                return f'cuda:{idx}'
            return 'cpu'
        except ValueError:
            pass

    return str(device_spec)


def get_cuda_device_count():
    """Get the number of available CUDA devices."""
    import torch
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def is_cuda_available():
    """Check if CUDA is available."""
    import torch
    return torch.cuda.is_available()


# =============================================================================
# Configuration Utilities
# =============================================================================

def vital_config_update(cfg, cfg_in):
    """
    Update a vital Config object from a dictionary or another Config.

    This is a utility to work around the fact that vital's merge_config
    doesn't support dictionary input.

    Args:
        cfg (kwiver.vital.config.config.Config): Config object to update
        cfg_in (dict | kwiver.vital.config.config.Config): New values

    Returns:
        kwiver.vital.config.config.Config: The updated config object

    Raises:
        KeyError: If cfg_in contains a key not present in cfg
    """
    if isinstance(cfg_in, dict):
        for key, value in cfg_in.items():
            if cfg.has_value(key):
                cfg.set_value(key, str(value))
            else:
                raise KeyError(f"cfg has no key={key}")
    else:
        cfg.merge_config(cfg_in)
    return cfg


def parse_bool(value):
    """
    Parse a value as boolean.

    Handles strings like 'true', 'false', '1', '0', 'yes', 'no'.

    Args:
        value: Value to parse (str, bool, int)

    Returns:
        bool: Parsed boolean value
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    return bool(value)


# =============================================================================
# Detection Conversion Utilities
# =============================================================================

def kwimage_to_kwiver_detections(detections):
    """
    Convert kwimage.Detections to kwiver DetectedObjectSet.

    Handles bounding boxes, scores, class indices, and optional segmentation masks.

    Args:
        detections (kwimage.Detections): Detections from kwimage

    Returns:
        kwiver.vital.types.DetectedObjectSet: Converted detection set
    """
    try:
        from kwiver.vital.types import BoundingBoxD
    except ImportError:
        from kwiver.vital.types import BoundingBox as BoundingBoxD

    from kwiver.vital.types import DetectedObjectSet
    from kwiver.vital.types import DetectedObject
    from kwiver.vital.types import DetectedObjectType
    from kwiver.vital.types.types import ImageContainer, Image

    segmentations = None
    if 'segmentations' in detections.data:
        segmentations = detections.data['segmentations']

    try:
        boxes = detections.boxes.to_ltrb()
    except Exception:
        boxes = detections.boxes.to_tlbr()

    scores = detections.scores
    class_idxs = detections.class_idxs

    if not segmentations:
        segmentations = (None,) * len(boxes)

    detected_objects = DetectedObjectSet()

    for tlbr, score, cidx, seg in zip(boxes.data, scores, class_idxs, segmentations):
        class_name = detections.classes[cidx]

        bbox_int = np.round(tlbr).astype(np.int32)
        bounding_box = BoundingBoxD(
            bbox_int[0], bbox_int[1], bbox_int[2], bbox_int[3])

        detected_object_type = DetectedObjectType(class_name, score)
        detected_object = DetectedObject(
            bounding_box, score, detected_object_type)

        if seg:
            mask = seg.to_relative_mask().numpy().data
            detected_object.mask = ImageContainer(Image(mask))

        detected_objects.add(detected_object)

    return detected_objects


def kwiver_to_kwimage_detections(detected_objects):
    """
    Convert kwiver DetectedObjectSet to kwimage.Detections.

    Args:
        detected_objects (kwiver.vital.types.DetectedObjectSet): KWIVER detections

    Returns:
        kwimage.Detections: Converted detections
    """
    import kwimage
    import ubelt as ub

    boxes = []
    scores = []
    class_idxs = []
    classes = []

    if len(detected_objects) > 0:
        obj = ub.peek(detected_objects)
        try:
            classes = obj.type.all_class_names()
        except AttributeError:
            classes = obj.type().all_class_names()

    for obj in detected_objects:
        box = obj.bounding_box()
        tlbr = [box.min_x(), box.min_y(), box.max_x(), box.max_y()]
        score = obj.confidence()
        cname = obj.type().get_most_likely_class()
        cidx = classes.index(cname)
        boxes.append(tlbr)
        scores.append(score)
        class_idxs.append(cidx)

    dets = kwimage.Detections(
        boxes=kwimage.Boxes(np.array(boxes), 'ltrb'),
        scores=np.array(scores),
        class_idxs=np.array(class_idxs),
        classes=classes,
    )
    return dets


def supervision_to_kwiver_detections(detections, class_names):
    """
    Convert supervision.Detections to kwiver DetectedObjectSet.

    Args:
        detections: supervision.Detections object with xyxy, confidence, class_id
        class_names: List of class names indexed by class_id

    Returns:
        kwiver.vital.types.DetectedObjectSet: Converted detection set
    """
    try:
        from kwiver.vital.types import BoundingBoxD
    except ImportError:
        from kwiver.vital.types import BoundingBox as BoundingBoxD

    from kwiver.vital.types import DetectedObjectSet
    from kwiver.vital.types import DetectedObject
    from kwiver.vital.types import DetectedObjectType

    output = DetectedObjectSet()

    for i in range(len(detections.xyxy)):
        box = detections.xyxy[i]
        score = detections.confidence[i]
        class_id = detections.class_id[i]

        if class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = str(class_id)

        bbox = BoundingBoxD(
            float(box[0]), float(box[1]),
            float(box[2]), float(box[3])
        )

        detected_object_type = DetectedObjectType(class_name, float(score))
        detected_object = DetectedObject(bbox, float(score), detected_object_type)

        output.add(detected_object)

    return output


def vital_to_kwimage_box(vital_bbox):
    """
    Convert a vital BoundingBox to a kwimage Box.

    Args:
        vital_bbox (kwiver.vital.types.BoundingBox): Vital bounding box

    Returns:
        kwimage.Box: Converted bounding box
    """
    import kwimage
    xyxy = [vital_bbox.min_x(), vital_bbox.min_y(),
            vital_bbox.max_x(), vital_bbox.max_y()]
    return kwimage.Box.coerce(xyxy, format='ltrb')


# =============================================================================
# Base Classes for Detectors
# =============================================================================

class BaseImageObjectDetector:
    """
    Mixin class providing common configuration management for ImageObjectDetector.

    Subclasses should:
    1. Call super().__init__() and set self._config to a scriptconfig.DataConfig instance
    2. Override _post_config_set() for post-configuration setup (e.g., model loading)
    3. Override detect() with the actual detection logic

    Example:
        class MyDetector(BaseImageObjectDetector, ImageObjectDetector):
            def __init__(self):
                ImageObjectDetector.__init__(self)
                BaseImageObjectDetector.__init__(self, MyConfig())

            def _post_config_set(self):
                self._build_model()

            def detect(self, image_data):
                # detection logic
                pass
    """

    def __init__(self, config):
        """
        Initialize with a configuration object.

        Args:
            config: A scriptconfig.DataConfig instance
        """
        self._config = config

    def get_configuration(self):
        """
        Get the algorithm configuration.

        Returns:
            kwiver.vital.config.config.Config: Configuration object
        """
        from kwiver.vital.algo import ImageObjectDetector
        cfg = super(ImageObjectDetector, self).get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        """
        Set the algorithm configuration.

        Args:
            cfg_in: Configuration dictionary or Config object

        Returns:
            bool: True on success
        """
        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)

        for key in self._config.keys():
            self._config[key] = str(cfg.get_value(key))

        self._config.__post_init__()

        # Set underscore-prefixed attributes for convenient access
        for key, value in self._config.items():
            setattr(self, "_" + key, value)

        self._post_config_set()
        return True

    def _post_config_set(self):
        """
        Called after configuration is set.

        Override this method to perform post-configuration setup,
        such as loading models or initializing resources.
        """
        pass

    def check_configuration(self, cfg):
        """
        Check if the configuration is valid.

        Override for custom validation.

        Args:
            cfg: Configuration object

        Returns:
            bool: True if configuration is valid
        """
        return True


class BaseTrainDetector:
    """
    Mixin class providing common configuration management for TrainDetector.

    Subclasses should:
    1. Call super().__init__() and set self._config to a scriptconfig.DataConfig instance
    2. Override _post_config_set() for post-configuration setup
    3. Override add_data_from_disk() and update_model() with training logic

    Example:
        class MyTrainer(BaseTrainDetector, TrainDetector):
            def __init__(self):
                TrainDetector.__init__(self)
                BaseTrainDetector.__init__(self, MyTrainerConfig())

            def _post_config_set(self):
                self._setup_directories()

            def add_data_from_disk(self, ...):
                # data handling logic
                pass

            def update_model(self):
                # training logic
                pass
    """

    def __init__(self, config):
        """
        Initialize with a configuration object.

        Args:
            config: A scriptconfig.DataConfig instance
        """
        self._config = config

    def get_configuration(self):
        """
        Get the algorithm configuration.

        Returns:
            kwiver.vital.config.config.Config: Configuration object
        """
        from kwiver.vital.algo import TrainDetector
        cfg = super(TrainDetector, self).get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        """
        Set the algorithm configuration.

        Args:
            cfg_in: Configuration dictionary or Config object

        Returns:
            bool: True on success
        """
        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)

        for key in self._config.keys():
            self._config[key] = str(cfg.get_value(key))

        self._config.__post_init__()

        # Set underscore-prefixed attributes for convenient access
        for key, value in self._config.items():
            setattr(self, "_" + key, value)

        self._post_config_set()
        return True

    def _post_config_set(self):
        """
        Called after configuration is set.

        Override this method to perform post-configuration setup,
        such as creating directories or initializing writers.
        """
        pass

    def check_configuration(self, cfg):
        """
        Check if the configuration is valid.

        Override for custom validation.

        Args:
            cfg: Configuration object

        Returns:
            bool: True if configuration is valid
        """
        return True


# =============================================================================
# Training Signal Handler
# =============================================================================

class TrainingInterruptHandler:
    """
    Context manager for handling training interruption signals gracefully.

    Captures SIGINT and SIGTERM to allow graceful shutdown of training.

    Example:
        with TrainingInterruptHandler("MyTrainer") as handler:
            for epoch in range(epochs):
                train_one_epoch()
                if handler.interrupted:
                    break
            save_model()
    """

    def __init__(self, trainer_name="Trainer"):
        """
        Initialize the interrupt handler.

        Args:
            trainer_name: Name to display in interrupt messages
        """
        import signal
        import threading

        self.trainer_name = trainer_name
        self.interrupted = False
        self._original_sigint = None
        self._original_sigterm = None
        self._signal = signal
        self._threading = threading

    def __enter__(self):
        def signal_handler(signum, frame):
            print(f"\n[{self.trainer_name}] Training interrupted, saving model...")
            self.interrupted = True

        if self._threading.current_thread().__class__.__name__ == '_MainThread':
            self._original_sigint = self._signal.signal(
                self._signal.SIGINT, signal_handler)
            self._original_sigterm = self._signal.signal(
                self._signal.SIGTERM, signal_handler)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._threading.current_thread().__class__.__name__ == '_MainThread':
            if self._original_sigint is not None:
                self._signal.signal(self._signal.SIGINT, self._original_sigint)
            if self._original_sigterm is not None:
                self._signal.signal(self._signal.SIGTERM, self._original_sigterm)
        return False


# =============================================================================
# Algorithm Registration Helper
# =============================================================================

def register_vital_algorithm(algorithm_class, implementation_name, description):
    """
    Register a KWIVER vital algorithm.

    This is a helper to reduce boilerplate in __vital_algorithm_register__ functions.

    Args:
        algorithm_class: The algorithm class to register
        implementation_name: Unique name for this implementation
        description: Human-readable description of the algorithm

    Example:
        def __vital_algorithm_register__():
            register_vital_algorithm(
                MyDetector, "my_detector", "My custom detector implementation")
    """
    from kwiver.vital.algo import algorithm_factory

    if algorithm_factory.has_algorithm_impl_name(
            algorithm_class.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(
        implementation_name, description, algorithm_class)

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
