# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from __future__ import print_function

from kwiver.vital.algo import ImageObjectDetector

try:
    from kwiver.vital.types import BoundingBoxD
except ImportError:
    from kwiver.vital.types import BoundingBox as BoundingBoxD

from kwiver.vital.types import DetectedObjectSet
from kwiver.vital.types import DetectedObject
from kwiver.vital.types import DetectedObjectType

import scriptconfig as scfg
import numpy as np
import ubelt as ub


class RFDETRDetectorConfig(scfg.DataConfig):
    """
    The configuration for :class:`RFDETRDetector`.
    """
    weight = scfg.Value(None, help='Path to a trained RF-DETR checkpoint (.pt file)')
    model_size = scfg.Value('base', help='Model size: nano, small, medium, base, or large')
    device = scfg.Value('auto', help='Device to run on: auto, cpu, cuda, or cuda:N')
    threshold = scfg.Value(0.5, help='Detection confidence threshold')
    optimize_inference = scfg.Value(True, help='Whether to optimize model for inference')

    def __post_init__(self):
        super().__post_init__()


class RFDETRDetector(ImageObjectDetector):
    """
    Implementation of ImageObjectDetector using RF-DETR

    RF-DETR is a real-time object detection model based on DETR architecture
    with DINOv2 backbone.
    """

    def __init__(self):
        ImageObjectDetector.__init__(self)
        self._kwiver_config = RFDETRDetectorConfig()
        self._model = None
        self._classes = None

    def get_configuration(self):
        cfg = super(ImageObjectDetector, self).get_configuration()
        for key, value in self._kwiver_config.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        _vital_config_update(cfg, cfg_in)

        for key in self._kwiver_config.keys():
            self._kwiver_config[key] = str(cfg.get_value(key))

        self._build_model()
        return True

    def _build_model(self):
        import torch

        weight_fpath = self._kwiver_config['weight']
        model_size = self._kwiver_config['model_size'].lower()
        device = self._kwiver_config['device']
        optimize = self._kwiver_config['optimize_inference']

        if isinstance(optimize, str):
            optimize = optimize.lower() in ('true', '1', 'yes')

        # Handle device selection
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

        # Import the appropriate RF-DETR model class based on size
        if model_size == 'nano':
            from rfdetr import RFDETRNano as RFDETRModel
        elif model_size == 'small':
            from rfdetr import RFDETRSmall as RFDETRModel
        elif model_size == 'medium':
            from rfdetr import RFDETRMedium as RFDETRModel
        elif model_size == 'base':
            from rfdetr import RFDETRBase as RFDETRModel
        elif model_size == 'large':
            from rfdetr import RFDETRLarge as RFDETRModel
        else:
            raise ValueError(f"Unknown model size: {model_size}. "
                           f"Expected: nano, small, medium, base, or large")

        # Load model
        print(f"[RFDETRDetector] Loading {model_size} model on {device}")

        if weight_fpath and ub.Path(weight_fpath).exists():
            # Load trained weights
            checkpoint = torch.load(weight_fpath, map_location=device)

            # Determine number of classes from checkpoint
            if 'args' in checkpoint and 'num_classes' in checkpoint['args']:
                num_classes = checkpoint['args']['num_classes']
            else:
                num_classes = 90  # default COCO classes

            # Get class names if available
            if 'args' in checkpoint and 'class_names' in checkpoint['args']:
                self._classes = checkpoint['args']['class_names']

            self._model = RFDETRModel(
                pretrain_weights=None,
                num_classes=num_classes,
                device=device
            )

            # Load the state dict
            if 'model' in checkpoint:
                self._model.model.model.load_state_dict(checkpoint['model'])
            else:
                self._model.model.model.load_state_dict(checkpoint)

            if self._classes:
                self._model.model.class_names = self._classes
        else:
            # Use pretrained weights
            self._model = RFDETRModel(device=device)

        # Set up class names
        if self._classes is None:
            self._classes = list(self._model.class_names.values())

        # Optimize for inference if requested
        if optimize:
            print("[RFDETRDetector] Optimizing model for inference")
            self._model.optimize_for_inference(compile=False)

        print(f"[RFDETRDetector] Model loaded with {len(self._classes)} classes")

    def check_configuration(self, cfg):
        return True

    def detect(self, image_data):
        import torch
        from PIL import Image

        threshold = float(self._kwiver_config['threshold'])

        # Convert kwiver image to numpy array
        full_rgb = image_data.asarray()

        # Convert to PIL Image (RF-DETR expects PIL or numpy)
        pil_img = Image.fromarray(full_rgb)

        # Run inference
        with torch.no_grad():
            detections = self._model.predict(pil_img, threshold=threshold)

        # Convert supervision Detections to kwiver format
        output = DetectedObjectSet()

        for i in range(len(detections.xyxy)):
            box = detections.xyxy[i]
            score = detections.confidence[i]
            class_id = detections.class_id[i]

            # Get class name
            if class_id < len(self._classes):
                class_name = self._classes[class_id]
            else:
                class_name = str(class_id)

            # Create bounding box
            bbox = BoundingBoxD(
                float(box[0]), float(box[1]),
                float(box[2]), float(box[3])
            )

            # Create detected object
            detected_object_type = DetectedObjectType(class_name, float(score))
            detected_object = DetectedObject(bbox, float(score), detected_object_type)

            output.add(detected_object)

        return output


def _vital_config_update(cfg, cfg_in):
    """
    Treat a vital Config object like a python dictionary
    """
    if isinstance(cfg_in, dict):
        for key, value in cfg_in.items():
            if cfg.has_value(key):
                cfg.set_value(key, str(value))
            else:
                raise KeyError('cfg has no key={}'.format(key))
    else:
        cfg.merge_config(cfg_in)
    return cfg


def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    implementation_name = "rf_detr"

    if algorithm_factory.has_algorithm_impl_name(
            RFDETRDetector.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(
        implementation_name, "PyTorch RF-DETR detection routine",
        RFDETRDetector)

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
