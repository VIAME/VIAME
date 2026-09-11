# This file is part of VIAME, and is distributed under an OSI-approved
# BSD 3-Clause License. See the root LICENSE file for details.
"""Shared crop geometry and portable SLEAP-NN keypoint model inference.

This module deliberately does not import KWIVER so training subprocesses and
CPU integration tests use exactly the same crop and model code as the refiner.
"""
from pathlib import Path
import numpy as np


MODEL_FORMAT = 'viame.sleap_keypoints.v1'


def parse_keypoint_names(value):
    names = value.split(',') if isinstance(value, str) else list(value)
    names = [str(name).strip() for name in names]
    if not names or not all(names) or len({n.lower() for n in names}) != len(names):
        raise ValueError('keypoint_names must contain unique, nonempty names')
    return names


def read_config(defaults, cfg):
    """Read KWIVER's string values using the declared default types."""
    result = {}
    for key, default in defaults.items():
        value = cfg.get_value(key)
        if isinstance(default, bool):
            text = str(value).lower().strip()
            if text not in ('true', 'false', '1', '0', 'yes', 'no'):
                raise ValueError('Invalid boolean for %s: %s' % (key, value))
            result[key] = text in ('true', '1', 'yes')
        else:
            result[key] = type(default)(value)
    return result


def as_rgb(image):
    """Normalize supported image layouts to uint8 RGB without changing geometry."""
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
        raise ValueError('SLEAP expects a grayscale, RGB, or RGBA image')
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    image = image[..., :3]
    if image.dtype != np.uint8:
        raise ValueError('SLEAP expects byte images; configure a byte image conversion upstream')
    return np.ascontiguousarray(image)


def crop_detection(image, box, size, padding):
    """Return a letterboxed crop and its image-to-crop affine, or None for bad boxes.

    ``size`` is (height, width); padding multiplies the box extent. Padding at
    image edges stays black rather than shifting the crop and its object center.
    """
    import cv2
    box = np.asarray(box, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        return None
    x1, y1, x2, y2 = box
    ih, iw = image.shape[:2]
    if x2 <= x1 or y2 <= y1 or x2 <= 0 or y2 <= 0 or x1 >= iw or y1 >= ih:
        return None
    h, w = size
    if h <= 0 or w <= 0 or not np.isfinite(padding) or padding < 1:
        raise ValueError('Crop dimensions must be positive and padding must be >= 1')
    scale = min(w / ((x2 - x1) * padding), h / ((y2 - y1) * padding))
    affine = np.array([[scale, 0, w / 2 - scale * (x1 + x2) / 2],
                       [0, scale, h / 2 - scale * (y1 + y2) / 2]], dtype=np.float64)
    crop = cv2.warpAffine(image, affine, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return crop, affine


def transform_points(points, affine, inverse=False):
    points = np.asarray(points, dtype=np.float64)
    if inverse:
        import cv2
        affine = cv2.invertAffineTransform(affine)
    return points @ affine[:, :2].T + affine[:, 2]


def detection_points(detection, names):
    """Match named KWIVER points case-insensitively; missing slots are NaN."""
    available = {name.lower(): pt for name, pt in detection.keypoints.items()}
    points = np.full((len(names), 2), np.nan)
    for i, name in enumerate(names):
        if name.lower() in available:
            points[i] = np.asarray(available[name.lower()].value, dtype=float)
    return points


def load_artifact(path):
    import torch
    artifact = torch.load(Path(path), map_location='cpu', weights_only=True)
    if not isinstance(artifact, dict) or artifact.get('format') != MODEL_FORMAT:
        raise ValueError('Expected a VIAME SLEAP keypoint model produced by sleap training')
    names = parse_keypoint_names(artifact['keypoint_names'])
    size = artifact['crop_size']
    if len(size) != 2 or any(not isinstance(v, int) or v <= 0 for v in size):
        raise ValueError('Invalid crop dimensions in SLEAP model')
    if not np.isfinite(artifact['crop_padding']) or artifact['crop_padding'] < 1:
        raise ValueError('Invalid crop padding in SLEAP model')
    config = artifact['model_config']
    if config['head_configs']['single_instance']['confmaps']['part_names'] != names:
        raise ValueError('SLEAP model keypoint metadata does not match its head')
    return artifact


class SleapPredictor:
    """Run the native SLEAP model and decoder on batches of VIAME box crops."""
    def __init__(self, path, device='auto', threshold=0.2):
        import torch
        from omegaconf import OmegaConf
        from sleap_nn.architectures.model import Model
        from sleap_nn.inference.layers.single_instance import SingleInstanceLayer
        from sleap_nn.inference.layers.backends.torch_backend import TorchBackend
        from sleap_nn.inference.layers.configs import PostprocessConfig

        self.artifact = load_artifact(path)
        self.names = self.artifact['keypoint_names']
        self.size = tuple(self.artifact['crop_size'])
        self.padding = self.artifact['crop_padding']
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        cfg = OmegaConf.create(self.artifact['model_config'])
        backbone = cfg.backbone_config.unet
        model = Model(backbone_type='unet', backbone_config=backbone,
                      head_configs=cfg.head_configs.single_instance,
                      model_type='single_instance')
        model.load_state_dict(self.artifact['state_dict'], strict=True)
        # SLEAP's inference layers provide B x 1 x C x H x W to their
        # Lightning modules. The exported raw architecture takes B x C x H x W.
        class CropModel(torch.nn.Module):
            def __init__(self, network):
                super().__init__()
                self.network = network

            def forward(self, images):
                if images.ndim == 5:
                    images = images.squeeze(1)
                return self.network(images)

        self.layer = SingleInstanceLayer(
            backend=TorchBackend(model=CropModel(model), device=device),
            output_stride=cfg.head_configs.single_instance.confmaps.output_stride,
            max_stride=backbone.max_stride,
            postprocess_config=PostprocessConfig(peak_threshold=threshold))

    def predict(self, crops):
        import torch
        batch = torch.from_numpy(np.stack(crops).transpose(0, 3, 1, 2)).float() / 255.0
        with torch.inference_mode():
            output = self.layer.predict(batch.to(self.device))
        return (output.pred_keypoints[:, 0].cpu().numpy(),
                output.pred_peak_values[:, 0].cpu().numpy())
