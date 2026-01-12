# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
KWIVER algorithm for stereo depth/disparity estimation using NVIDIA Foundation-Stereo.

This algorithm accepts left and right stereo image pairs and outputs either
disparity maps or depth maps depending on configuration.
"""

import os
import sys
import json
import numpy as np

import scriptconfig as scfg

from kwiver.vital.algo import ComputeStereoDepthMap
from kwiver.vital.types import Image, ImageContainer

from viame.core.utils import str2bool

from viame.pytorch.utilities import vital_config_update


class FoundationStereoConfig(scfg.DataConfig):
    """
    Configuration for :class:`FoundationStereo`.
    """
    checkpoint_path = scfg.Value(
        '', help='Path to the pretrained model checkpoint (.pth file)')
    vit_size = scfg.Value(
        'vitl', help='Vision Transformer backbone size: vitl (large), vitb (base), or vits (small)')
    device = scfg.Value(
        'auto', help="Device to run inference on: 'auto' (use GPU if available), 'cpu', or specific GPU (e.g., 'cuda:0')")
    num_iters = scfg.Value(
        32, help='Number of GRU refinement iterations during inference')
    use_hierarchical = scfg.Value(
        False, help='Use hierarchical inference for high-resolution images (>1K pixels)')
    hierarchical_ratio = scfg.Value(
        0.5, help='Scale ratio for first pass in hierarchical inference')
    mixed_precision = scfg.Value(
        True, help='Use mixed precision (FP16) inference for faster computation')
    low_memory = scfg.Value(
        False, help='Enable low memory mode for limited GPU RAM')
    output_mode = scfg.Value(
        'disparity', help="Output mode: 'disparity' (scaled by 256, uint16) or 'depth' (millimeters, uint16)")
    calibration_file = scfg.Value(
        '', help='Path to KWIVER stereo calibration file (JSON format) - required for depth output')
    remove_invisible = scfg.Value(
        True, help='Set invalid disparity values (negative x in right image) to infinity')


class FoundationStereo(ComputeStereoDepthMap):
    """
    Algorithm for stereo disparity/depth estimation using NVIDIA Foundation-Stereo.

    This algorithm takes left and right stereo images as input and produces
    either disparity maps or depth maps based on configuration.

    When output_mode is 'disparity', returns disparity scaled by 256 as uint16.
    When output_mode is 'depth', returns depth in millimeters as uint16
    (requires calibration_file to be set).
    """

    def __init__(self):
        ComputeStereoDepthMap.__init__(self)

        # Configuration
        self._config = FoundationStereoConfig()

        # Camera parameters (loaded from calibration file)
        self._focal_length = 0.0
        self._baseline = 0.0
        self._principal_x = 0.0
        self._principal_y = 0.0

        # Model (loaded in set_configuration)
        self._model = None
        self._InputPadder = None
        self._torch_device = None

    def get_configuration(self):
        cfg = super(ComputeStereoDepthMap, self).get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        import torch

        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)

        for key in self._config.keys():
            self._config[key] = str(cfg.get_value(key))

        # Convert types appropriately
        self._config['num_iters'] = int(self._config['num_iters'])
        self._config['use_hierarchical'] = str2bool(self._config['use_hierarchical'])
        self._config['hierarchical_ratio'] = float(self._config['hierarchical_ratio'])
        self._config['mixed_precision'] = str2bool(self._config['mixed_precision'])
        self._config['low_memory'] = str2bool(self._config['low_memory'])
        self._config['remove_invisible'] = str2bool(self._config['remove_invisible'])

        # Load calibration if needed for depth output
        calibration_file = self._config['calibration_file']
        if calibration_file and os.path.exists(calibration_file):
            self._load_calibration(calibration_file)

        # Validate configuration
        checkpoint_path = self._config['checkpoint_path']
        if not checkpoint_path:
            raise RuntimeError("checkpoint_path must be specified")
        if not os.path.exists(checkpoint_path):
            raise RuntimeError(f"Checkpoint file not found: {checkpoint_path}")

        if self._config['output_mode'] == 'depth':
            if self._focal_length <= 0 or self._baseline <= 0:
                raise RuntimeError("calibration_file with valid focal length and baseline required for depth output")

        # Add foundation-stereo to path
        foundation_stereo_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'packages', 'pytorch-libs', 'foundation-stereo'
        )
        if foundation_stereo_dir not in sys.path:
            sys.path.insert(0, foundation_stereo_dir)

        # Import foundation-stereo modules
        from omegaconf import OmegaConf
        from core.foundation_stereo import FoundationStereo as FoundationStereoModel
        from core.utils.utils import InputPadder

        # Store InputPadder class for later use
        self._InputPadder = InputPadder

        # Load model configuration
        ckpt_dir = os.path.dirname(checkpoint_path)
        cfg_path = os.path.join(ckpt_dir, 'cfg.yaml')
        if os.path.exists(cfg_path):
            model_cfg = OmegaConf.load(cfg_path)
        else:
            model_cfg = OmegaConf.create({})

        # Set vit_size
        model_cfg['vit_size'] = self._config['vit_size']

        # Create model
        self._model = FoundationStereoModel(model_cfg)

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        self._model.load_state_dict(ckpt['model'])

        # Resolve device (handle 'auto' mode)
        device = self._config['device']
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._torch_device = torch.device(device)
        self._model.to(self._torch_device)
        self._model.eval()

        # Disable gradients
        torch.set_grad_enabled(False)

        return True

    def check_configuration(self, cfg):
        if not cfg.has_value("checkpoint_path"):
            print("Error: checkpoint_path is required")
            return False
        checkpoint_path = str(cfg.get_value("checkpoint_path"))
        if not checkpoint_path:
            print("Error: checkpoint_path must not be empty")
            return False

        output_mode = str(cfg.get_value("output_mode"))
        if output_mode not in ['disparity', 'depth']:
            print(f"Error: output_mode must be 'disparity' or 'depth', got '{output_mode}'")
            return False

        return True

    def _load_calibration(self, cal_fpath):
        """Load stereo calibration from a JSON file.

        Supports KWIVER stereo rig JSON format with keys:
            - fx_left, fy_left, cx_left, cy_left: left camera intrinsics
            - T: translation vector between cameras (baseline)

        Args:
            cal_fpath: Path to calibration JSON file
        """
        with open(cal_fpath, 'r') as f:
            data = json.load(f)

        # Extract focal length from left camera (use fx_left as primary)
        self._focal_length = float(data.get('fx_left', 0.0))

        # Extract principal point from left camera
        self._principal_x = float(data.get('cx_left', 0.0))
        self._principal_y = float(data.get('cy_left', 0.0))

        # Compute baseline from translation vector
        T = data.get('T', [0.0, 0.0, 0.0])
        if isinstance(T, list) and len(T) >= 3:
            # Baseline is typically the absolute X component for horizontal stereo
            self._baseline = abs(T[0])
            # If X is near zero, use full magnitude
            if self._baseline < 1e-6:
                self._baseline = np.sqrt(T[0]**2 + T[1]**2 + T[2]**2)
        else:
            self._baseline = 0.0

        print(f"Loaded calibration: focal_length={self._focal_length}, "
              f"baseline={self._baseline}, principal=({self._principal_x}, {self._principal_y})")

    def _format_image(self, image_container):
        """Convert KWIVER ImageContainer to numpy array.

        Args:
            image_container: KWIVER ImageContainer

        Returns:
            numpy array in (H, W, C) format, RGB, uint8
        """
        img_npy = image_container.image().asarray().astype('uint8')

        # Handle grayscale images
        if len(img_npy.shape) == 2:
            img_npy = np.stack((img_npy,) * 3, axis=-1)
        elif img_npy.shape[2] == 1:
            img_npy = np.concatenate([img_npy] * 3, axis=-1)

        return img_npy

    def compute(self, left_image, right_image):
        """Compute stereo depth/disparity map from left and right images.

        Args:
            left_image: Left stereo image (ImageContainer)
            right_image: Right stereo image (ImageContainer)

        Returns:
            ImageContainer containing either:
                - Disparity map (scaled by 256, uint16) if output_mode='disparity'
                - Depth map (millimeters, uint16) if output_mode='depth'
        """
        import torch

        # Convert to numpy arrays
        left_npy = self._format_image(left_image)
        right_npy = self._format_image(right_image)

        # Validate dimensions match
        if left_npy.shape != right_npy.shape:
            raise RuntimeError(
                f"Left and right image dimensions must match: "
                f"{left_npy.shape} vs {right_npy.shape}"
            )

        H, W = left_npy.shape[:2]

        # Convert to PyTorch tensors (B, C, H, W)
        left_tensor = torch.as_tensor(left_npy).to(self._torch_device).float()
        left_tensor = left_tensor[None].permute(0, 3, 1, 2)
        right_tensor = torch.as_tensor(right_npy).to(self._torch_device).float()
        right_tensor = right_tensor[None].permute(0, 3, 1, 2)

        # Pad images to be divisible by 32
        padder = self._InputPadder(left_tensor.shape, divis_by=32, force_square=False)
        left_padded, right_padded = padder.pad(left_tensor, right_tensor)

        # Run inference
        with torch.cuda.amp.autocast(self._config['mixed_precision']):
            if self._config['use_hierarchical']:
                disp = self._model.run_hierachical(
                    left_padded, right_padded,
                    iters=self._config['num_iters'],
                    test_mode=True,
                    small_ratio=self._config['hierarchical_ratio']
                )
            else:
                disp = self._model.forward(
                    left_padded, right_padded,
                    iters=self._config['num_iters'],
                    test_mode=True,
                    low_memory=self._config['low_memory']
                )

        # Unpad and convert to numpy
        disp = padder.unpad(disp.float())
        disp_npy = disp.data.cpu().numpy().reshape(H, W)

        # Handle invisible regions if requested
        if self._config['remove_invisible']:
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            us_right = xx - disp_npy
            invalid = us_right < 0
            disp_npy[invalid] = np.inf

        # Output based on mode
        if self._config['output_mode'] == 'depth':
            # Compute depth: depth = focal_length * baseline / disparity
            safe_disp = np.where(disp_npy > 0, disp_npy, 1e-6)
            depth_npy = (self._focal_length * self._baseline) / safe_disp
            depth_npy = np.where(disp_npy > 0, depth_npy, 0)

            # Scale depth to millimeters as uint16
            depth_mm = (depth_npy * 1000).clip(0, 65535).astype(np.uint16)
            output = ImageContainer(Image(depth_mm))
        else:
            # Output disparity scaled by 256 as uint16
            disp_output = (disp_npy * 256).clip(0, 65535).astype(np.uint16)
            output = ImageContainer(Image(disp_output))

        return output


def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    implementation_name = "foundation_stereo"

    if algorithm_factory.has_algorithm_impl_name(
            FoundationStereo.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(
        implementation_name,
        "Stereo depth/disparity estimation using NVIDIA Foundation-Stereo model",
        FoundationStereo
    )

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
