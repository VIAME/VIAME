# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from kwiver.vital.algo import ImageObjectDetector

import scriptconfig as scfg
import ubelt as ub

from viame.pytorch.utilities import (
    resolve_device_str,
    vital_config_update,
    supervision_to_kwiver_detections,
    register_vital_algorithm,
    parse_bool,
)


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
        device = resolve_device_str(self._kwiver_config['device'])
        optimize = parse_bool(self._kwiver_config['optimize_inference'])

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
        output = supervision_to_kwiver_detections(detections, self._classes)
        return output


def __vital_algorithm_register__():
    register_vital_algorithm(
        RFDETRDetector, "rf_detr", "PyTorch RF-DETR detection routine"
    )
