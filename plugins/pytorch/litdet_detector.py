# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from kwiver.vital.algo import ImageObjectDetector

import scriptconfig as scfg
import ubelt as ub

from viame.pytorch.utilities import (
    resolve_device_str,
    vital_config_update,
    register_vital_algorithm,
    parse_bool,
)


class LitDetDetectorConfig(scfg.DataConfig):
    """
    The configuration for :class:`LitDetDetector`.
    """
    checkpoint = scfg.Value(None, help='Path to a trained LitDet checkpoint (.ckpt file)')
    model_type = scfg.Value('faster_rcnn', help='Model type: faster_rcnn, ssd, ssdlite, retinanet, fcos')
    config_path = scfg.Value(None, help='Path to the Hydra config directory (optional)')
    config_name = scfg.Value('train.yaml', help='Name of the config file to use')
    device = scfg.Value('auto', help='Device to run on: auto, cpu, cuda, or cuda:N')
    threshold = scfg.Value(0.5, help='Detection confidence threshold')
    batch_size = scfg.Value(1, help='Batch size for inference')
    num_classes = scfg.Value(None, help='Number of classes (auto-detected from checkpoint if not set)')

    def __post_init__(self):
        super().__post_init__()


class LitDetDetector(ImageObjectDetector):
    """
    Implementation of ImageObjectDetector using LitDet (Lightning Hydra Detection).

    LitDet is a configurable object detection framework built on PyTorch Lightning
    and Hydra. It supports various detection architectures like Faster R-CNN.
    """

    def __init__(self):
        ImageObjectDetector.__init__(self)
        self._kwiver_config = LitDetDetectorConfig()
        self._model = None
        self._device = None
        self._classes = None
        self._transforms = None

    def get_configuration(self):
        cfg = super(ImageObjectDetector, self).get_configuration()
        for key, value in self._kwiver_config.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)

        for key in self._kwiver_config.keys():
            self._kwiver_config[key] = str(cfg.get_value(key))

        self._build_model()
        return True

    def _build_model(self):
        import torch
        import hydra
        from omegaconf import OmegaConf

        checkpoint_path = self._kwiver_config['checkpoint']
        device_str = resolve_device_str(self._kwiver_config['device'])
        self._device = torch.device(device_str)

        if not checkpoint_path or not ub.Path(checkpoint_path).exists():
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

        print(f"[LitDetDetector] Loading checkpoint from {checkpoint_path}")

        # Load checkpoint to get hyperparameters
        checkpoint = torch.load(checkpoint_path, map_location=self._device)

        # Get the config from checkpoint if available
        if 'hyper_parameters' in checkpoint:
            hparams = checkpoint['hyper_parameters']
        else:
            hparams = {}

        # Try to load the model using Lightning's load_from_checkpoint
        from lightning_hydra_detection.tasks.detect_module import DetectLitModule

        # Build the model configuration
        config_path = self._kwiver_config['config_path']
        config_name = self._kwiver_config['config_name']

        if config_path and ub.Path(config_path).exists():
            # Use user-specified config
            with hydra.initialize_config_dir(config_dir=str(ub.Path(config_path).resolve())):
                cfg = hydra.compose(config_name=config_name)
        else:
            # Use default litdet config
            with hydra.initialize(
                version_base="1.3",
                config_path="pkg://lightning_hydra_detection.configs"
            ):
                cfg = hydra.compose(config_name="train.yaml")

        # Instantiate the task (model)
        task = hydra.utils.instantiate(cfg.task)

        # Load the state dict from checkpoint
        if 'state_dict' in checkpoint:
            task.load_state_dict(checkpoint['state_dict'])
        else:
            task.load_state_dict(checkpoint)

        task = task.to(self._device)
        task.eval()

        self._model = task

        # Set up transforms for inference
        import torchvision.transforms.v2 as transforms
        from torch import float32

        self._transforms = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(dtype=float32, scale=True),
        ])

        # Try to get class names from checkpoint or config
        if 'classes' in hparams:
            self._classes = hparams['classes']
        else:
            # Default to COCO classes or generic numbered classes
            self._classes = None

        print(f"[LitDetDetector] Model loaded on {self._device}")

    def check_configuration(self, cfg):
        if not cfg.has_value("checkpoint") or len(cfg.get_value("checkpoint")) == 0:
            print("A checkpoint path must be specified!")
            return False
        return True

    def detect(self, image_data):
        import torch
        import numpy as np

        try:
            from kwiver.vital.types import BoundingBoxD
        except ImportError:
            from kwiver.vital.types import BoundingBox as BoundingBoxD

        from kwiver.vital.types import DetectedObjectSet
        from kwiver.vital.types import DetectedObject
        from kwiver.vital.types import DetectedObjectType

        threshold = float(self._kwiver_config['threshold'])

        # Convert kwiver image to numpy array
        full_rgb = image_data.asarray()

        # Apply transforms
        img_tensor = self._transforms(full_rgb)
        img_tensor = img_tensor.to(self._device)

        # Run inference
        with torch.no_grad():
            # Model expects a list of images
            predictions = self._model([img_tensor])

        # predictions is a list of dicts with 'boxes', 'labels', 'scores'
        output = DetectedObjectSet()

        if len(predictions) > 0:
            pred = predictions[0]
            boxes = pred['boxes'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()

            for i in range(len(boxes)):
                score = float(scores[i])
                if score < threshold:
                    continue

                box = boxes[i]
                label = int(labels[i])

                # Get class name
                if self._classes is not None and label < len(self._classes):
                    class_name = self._classes[label]
                else:
                    class_name = str(label)

                bbox = BoundingBoxD(
                    float(box[0]), float(box[1]),
                    float(box[2]), float(box[3])
                )

                detected_object_type = DetectedObjectType(class_name, score)
                detected_object = DetectedObject(bbox, score, detected_object_type)

                output.add(detected_object)

        return output


def __vital_algorithm_register__():
    register_vital_algorithm(
        LitDetDetector, "litdet", "PyTorch LitDet detection routine"
    )
