# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from kwiver.vital.algo import ImageObjectDetector

import scriptconfig as scfg
import os

from viame.pytorch.utilities import (
    report_cuda_errors,
    resolve_device_str,
    vital_config_update,
    register_vital_algorithm,
)


class LitDetDetectorConfig(scfg.DataConfig):
    """Configuration for LitDetDetector."""
    checkpoint = scfg.Value('', help='Path to a trained LitDet checkpoint (.ckpt file)')
    config_file = scfg.Value('', help='Path to the user Hydra YAML configuration file')
    threshold = scfg.Value(0.5, help='Detection confidence threshold')
    device = scfg.Value('auto', help='Device to run on: auto, cpu, cuda, or cuda:N')

    def __post_init__(self):
        super().__post_init__()


class LitDetDetector(ImageObjectDetector):
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

    @report_cuda_errors("LitDetDetector initialization")
    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)

        for key in self._kwiver_config.keys():
            self._kwiver_config[key] = str(cfg.get_value(key))

        self._build_model()
        return True

    def _build_model(self):
        import torch

        checkpoint_path = self._kwiver_config['checkpoint']
        device_str = resolve_device_str(self._kwiver_config['device'])
        self._device = torch.device(device_str)

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise ValueError(f"[LitDetDetector] Checkpoint path does not exist: {checkpoint_path}")

        print(f"[LitDetDetector] Loading checkpoint from {checkpoint_path}")

        from litdet.tasks.detect_module import DetectLitModule

        try:
            task = DetectLitModule.load_from_checkpoint(
                checkpoint_path,
                map_location=self._device,
                weights_only=False
            )
        except Exception as e:
            print(f"[LitDetDetector] ERROR loading checkpoint natively: {e}")
            raise e

        task = task.to(self._device)
        task.eval()
        self._model = task

        import torchvision.transforms.v2 as transforms
        from torch import float32

        self._transforms = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(dtype=float32, scale=True),
        ])

        checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)

        if 'hyper_parameters' in checkpoint and 'classes' in checkpoint['hyper_parameters']:
            self._classes = checkpoint['hyper_parameters']['classes']
        elif 'hyper_parameters' in checkpoint and 'datamodule_kwargs' in checkpoint['hyper_parameters']:
            self._classes = None
        else:
            self._classes = None

        print(f"[LitDetDetector] Model loaded successfully on {self._device}")

    def check_configuration(self, cfg):
        if not cfg.has_value("checkpoint") or len(cfg.get_value("checkpoint")) == 0:
            print("[LitDetDetector] A checkpoint path must be specified!")
            return False
        return True

    @report_cuda_errors("LitDetDetector detection")
    def detect(self, image_data):
        import torch

        try:
            from kwiver.vital.types import BoundingBoxD
        except ImportError:
            from kwiver.vital.types import BoundingBox as BoundingBoxD

        from kwiver.vital.types import DetectedObjectSet, DetectedObject, DetectedObjectType

        threshold = float(self._kwiver_config['threshold'])

        full_rgb = image_data.asarray()
        img_tensor = self._transforms(full_rgb).to(self._device)

        with torch.no_grad():
            predictions = self._model([img_tensor])

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
                class_name = self._classes[label] if (self._classes and label < len(self._classes)) else str(label)

                bbox = BoundingBoxD(float(box[0]), float(box[1]), float(box[2]), float(box[3]))

                detected_object_type = DetectedObjectType(class_name, score)
                detected_object = DetectedObject(bbox, score, detected_object_type)
                output.add(detected_object)

        return output


def __vital_algorithm_register__():
    register_vital_algorithm(
        LitDetDetector, "litdet", "PyTorch LitDet detection routine"
    )
