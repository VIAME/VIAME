# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
KWIVER algorithm for stereo depth/disparity estimation using NVIDIA
Fast-Foundation-Stereo (real-time variant of Foundation-Stereo).

Mirrors the foundation_stereo wrapper, with two differences driven by how
fast-foundation-stereo ships its weights:

1. Each weight directory contains a fully-serialized model (.pth) plus a
   cfg.yaml describing the architecture, so the model is loaded directly
   via torch.load instead of constructing a model from a config and then
   loading a state_dict.
2. The forward path takes `optimize_build_volume` and exposes max_disp
   as a runtime knob via model.args.
"""

import os
import sys
import numpy as np

import scriptconfig as scfg

from viame.algo import ComputeStereoDepthMap
from viame.types import Image, ImageContainer

from viame.utilities.utils import (
    str2bool, image_container_to_uint8_hwc, read_stereo_calibration,
)

from viame.object_detectors.base import vital_config_update, report_cuda_errors
from viame import image_kernels


class FastFoundationStereoConfig(scfg.DataConfig):
    """
    Configuration for :class:`FastFoundationStereo`.
    """

    checkpoint_path = scfg.Value(
        "",
        help="Path to the serialized model checkpoint (.pth file). The "
        "parent directory must contain cfg.yaml from the release.",
    )
    package_dir = scfg.Value(
        "",
        help="Directory containing the fast-foundation-stereo source "
        "(must contain core/ and Utils.py). When empty, the wrapper "
        "searches via VIAME_SOURCE_DIR env var, the install tree "
        "sibling layout, and src-tree relative paths.",
    )
    device = scfg.Value(
        "auto",
        help="Device to run inference on: 'auto' (use GPU if available), 'cpu', or specific GPU (e.g., 'cuda:0')",
    )
    num_iters = scfg.Value(
        -1,
        help="Number of GRU refinement iterations during inference. -1 uses cfg.yaml default (typically 8).",
    )
    max_disp = scfg.Value(
        -1, help="Maximum disparity. -1 uses cfg.yaml default (typically 192-416)."
    )
    use_hierarchical = scfg.Value(
        False, help="Use hierarchical inference for high-resolution images (>1K pixels)"
    )
    hierarchical_ratio = scfg.Value(
        0.5, help="Scale ratio for first pass in hierarchical inference"
    )
    mixed_precision = scfg.Value(
        True, help="Use mixed precision (FP16) inference for faster computation"
    )
    optimize_build_volume = scfg.Value(
        "pytorch1",
        help="Cost-volume build optimization: 'pytorch1' (default, fast), 'pytorch2', or '' to disable.",
    )
    output_mode = scfg.Value(
        "disparity",
        help="Output mode: 'disparity' (scaled by 256, uint16) or 'depth' (millimeters, uint16)",
    )
    calibration_file = scfg.Value(
        "",
        help="Path to KWIVER stereo calibration file (JSON format) - required for depth output",
    )
    remove_invisible = scfg.Value(
        True,
        help="Set invalid disparity values (negative x in right image) to infinity",
    )
    scale = scfg.Value(
        1.0,
        help="Scale factor for input images (<=1.0). Reduces memory usage by downscaling before inference.",
    )
    use_half_precision = scfg.Value(
        False,
        help="Use half precision (FP16) for model weights. Reduces memory usage on CUDA devices.",
    )


class FastFoundationStereo(ComputeStereoDepthMap):
    """
    Algorithm for stereo disparity/depth estimation using NVIDIA
    Fast-Foundation-Stereo. Real-time-oriented sibling of foundation_stereo.

    Output semantics match the foundation_stereo wrapper:
      - output_mode='disparity' returns disparity * 256 as uint16
      - output_mode='depth' returns depth in millimeters as uint16
        (requires calibration_file)
    """

    def __init__(self):
        ComputeStereoDepthMap.__init__(self)

        self._config = FastFoundationStereoConfig()

        # Camera parameters (loaded from calibration file)
        self._focal_length = 0.0
        self._baseline = 0.0
        self._principal_x = 0.0
        self._principal_y = 0.0

        # Model and helpers (loaded in set_configuration)
        self._model = None
        self._InputPadder = None
        self._torch_device = None
        self._amp_dtype = None

    def get_configuration(self):
        cfg = super(ComputeStereoDepthMap, self).get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    @report_cuda_errors("FastFoundationStereo initialization")
    def set_configuration(self, cfg_in):
        import torch
        import yaml

        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)

        for key in self._config.keys():
            self._config[key] = str(cfg.get_value(key))

        # Type conversions
        self._config["num_iters"] = int(self._config["num_iters"])
        self._config["max_disp"] = int(self._config["max_disp"])
        self._config["use_hierarchical"] = str2bool(self._config["use_hierarchical"])
        self._config["hierarchical_ratio"] = float(self._config["hierarchical_ratio"])
        self._config["mixed_precision"] = str2bool(self._config["mixed_precision"])
        self._config["remove_invisible"] = str2bool(self._config["remove_invisible"])
        self._config["scale"] = float(self._config["scale"])
        if self._config["scale"] > 1.0:
            raise RuntimeError("scale must be <= 1.0")
        self._config["use_half_precision"] = str2bool(
            self._config["use_half_precision"]
        )

        # Calibration (optional unless output_mode=depth)
        calibration_file = self._config["calibration_file"]
        if calibration_file and os.path.exists(calibration_file):
            self._load_calibration(calibration_file)

        checkpoint_path = self._config["checkpoint_path"]
        if not checkpoint_path:
            raise RuntimeError("checkpoint_path must be specified")
        if not os.path.exists(checkpoint_path):
            raise RuntimeError(f"Checkpoint file not found: {checkpoint_path}")

        if self._config["output_mode"] == "depth":
            if self._focal_length <= 0 or self._baseline <= 0:
                raise RuntimeError(
                    "calibration_file with valid focal length and baseline required for depth output"
                )

        # Resolve the fast-foundation-stereo source directory. Same lookup
        # used to support both running from a build install (where __file__
        # lives under site-packages and bears no relation to the src tree)
        # and running directly out of the source checkout.
        ffs_dir = self._resolve_package_dir()
        if ffs_dir is None:
            raise RuntimeError(
                "Could not locate the fast-foundation-stereo source. Set "
                ":stereo_disparity:fast_foundation_stereo:package_dir in "
                "your pipe, or export VIAME_SOURCE_DIR pointing at your "
                "viame source checkout."
            )
        if ffs_dir not in sys.path:
            sys.path.insert(0, ffs_dir)

        # Imports must come after sys.path is updated.
        from core.utils.utils import InputPadder
        from Utils import AMP_DTYPE

        self._InputPadder = InputPadder
        self._amp_dtype = AMP_DTYPE

        # Resolve device
        device = self._config["device"]
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._torch_device = torch.device(device)

        # Load the serialized model. The release ships a complete pickled
        # model (no state_dict path), so torch.load returns the model object
        # directly. Loading on CPU first avoids running short on GPU memory
        # if the device is busy.
        self._model = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )

        # Apply runtime args from cfg.yaml + per-run overrides. cfg.yaml
        # lives next to the .pth and carries architecture defaults plus
        # sensible runtime values for valid_iters / max_disp.
        ckpt_dir = os.path.dirname(checkpoint_path)
        cfg_path = os.path.join(ckpt_dir, "cfg.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as ff:
                yaml_cfg = yaml.safe_load(ff) or {}
            for k, v in yaml_cfg.items():
                if hasattr(self._model.args, k):
                    setattr(self._model.args, k, v)

        if self._config["num_iters"] >= 0:
            self._model.args.valid_iters = self._config["num_iters"]
        if self._config["max_disp"] >= 0:
            self._model.args.max_disp = self._config["max_disp"]

        self._model = self._model.to(self._torch_device)
        if self._config["use_half_precision"] and "cuda" in str(self._torch_device):
            self._model.half()
        self._model.eval()

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
        if output_mode not in ["disparity", "depth"]:
            print(
                f"Error: output_mode must be 'disparity' or 'depth', got '{output_mode}'"
            )
            return False

        return True

    def _resolve_package_dir(self):
        """Find a directory that contains core/ and Utils.py from the
        fast-foundation-stereo release. Returns the path or None."""
        candidates = []

        explicit = self._config.get("package_dir", "")
        if explicit:
            candidates.append(explicit)

        env = os.environ.get("VIAME_SOURCE_DIR", "")
        if env:
            candidates.append(
                os.path.join(env, "packages", "pytorch-libs", "fast-foundation-stereo")
            )

        # Walk up from this file looking for a sibling 'packages/pytorch-libs/
        # fast-foundation-stereo' tree. Works whether __file__ is the src
        # checkout (.../src/library/measurement/stereo/fast_foundation_stereo.py) or
        # the installed copy (.../site-packages/viame/pytorch/...) — though
        # only the src layout actually has the package dir alongside.
        here = os.path.dirname(os.path.abspath(__file__))
        for _ in range(6):
            candidates.append(
                os.path.join(here, "packages", "pytorch-libs", "fast-foundation-stereo")
            )
            here = os.path.dirname(here)

        for d in candidates:
            if not d:
                continue
            if os.path.isfile(os.path.join(d, "Utils.py")) and os.path.isdir(
                os.path.join(d, "core")
            ):
                return d
        return None

    def _load_calibration(self, cal_fpath):
        cal = read_stereo_calibration(cal_fpath)
        self._focal_length = cal["focal_length"]
        self._principal_x = cal["principal_x"]
        self._principal_y = cal["principal_y"]
        self._baseline = cal["baseline"]

    def _format_image(self, image_container):
        return image_container_to_uint8_hwc(image_container)

    @report_cuda_errors("FastFoundationStereo computation")
    def compute(self, left_image, right_image):
        """Run fast-foundation-stereo on a left/right pair.

        Returns either a disparity map (scaled * 256, uint16) or a depth map
        (millimeters, uint16) depending on output_mode.
        """
        import torch
        import cv2
        from viame import image_kernels

        left_npy = self._format_image(left_image)
        right_npy = self._format_image(right_image)

        if left_npy.shape != right_npy.shape:
            raise RuntimeError(
                f"Left and right image dimensions must match: "
                f"{left_npy.shape} vs {right_npy.shape}"
            )

        H_orig, W_orig = left_npy.shape[:2]
        scale = self._config["scale"]

        if scale < 1.0:
            H_scaled = int(H_orig * scale)
            W_scaled = int(W_orig * scale)
            left_npy = image_kernels.resize_area(left_npy, W_scaled, H_scaled)
            right_npy = image_kernels.resize_area(right_npy, W_scaled, H_scaled)
            H, W = H_scaled, W_scaled
        else:
            H, W = H_orig, W_orig

        use_half = self._config["use_half_precision"] and "cuda" in str(
            self._torch_device
        )

        left_tensor = torch.as_tensor(left_npy).to(self._torch_device)
        left_tensor = left_tensor.half() if use_half else left_tensor.float()
        left_tensor = left_tensor[None].permute(0, 3, 1, 2)
        right_tensor = torch.as_tensor(right_npy).to(self._torch_device)
        right_tensor = right_tensor.half() if use_half else right_tensor.float()
        right_tensor = right_tensor[None].permute(0, 3, 1, 2)

        padder = self._InputPadder(left_tensor.shape, divis_by=32, force_square=False)
        left_padded, right_padded = padder.pad(left_tensor, right_tensor)

        opt_build = self._config["optimize_build_volume"]
        with torch.amp.autocast(
            "cuda",
            enabled=self._config["mixed_precision"]
            and "cuda" in str(self._torch_device),
            dtype=self._amp_dtype,
        ):
            if self._config["use_hierarchical"]:
                disp = self._model.run_hierachical(
                    left_padded,
                    right_padded,
                    iters=self._model.args.valid_iters,
                    test_mode=True,
                    small_ratio=self._config["hierarchical_ratio"],
                )
            else:
                forward_kwargs = dict(
                    iters=self._model.args.valid_iters,
                    test_mode=True,
                )
                if opt_build:
                    forward_kwargs["optimize_build_volume"] = opt_build
                disp = self._model.forward(left_padded, right_padded, **forward_kwargs)

        disp = padder.unpad(disp.float())
        disp_npy = disp.data.cpu().numpy().reshape(H, W)

        if scale < 1.0:
            # Disparity scales inversely with image scale.
            disp_npy = image_kernels.resize(disp_npy, W_orig, H_orig)
            disp_npy = disp_npy / scale

        if self._config["remove_invisible"]:
            yy, xx = np.meshgrid(np.arange(H_orig), np.arange(W_orig), indexing="ij")
            us_right = xx - disp_npy
            invalid = us_right < 0
            disp_npy[invalid] = np.inf

        if self._config["output_mode"] == "depth":
            safe_disp = np.where(disp_npy > 0, disp_npy, 1e-6)
            depth_npy = (self._focal_length * self._baseline) / safe_disp
            depth_npy = np.where(disp_npy > 0, depth_npy, 0)
            depth_mm = (depth_npy * 1000).clip(0, 65535).astype(np.uint16)
            output = ImageContainer(Image(depth_mm))
        else:
            disp_output = (disp_npy * 256).clip(0, 65535).astype(np.uint16)
            output = ImageContainer(Image(disp_output))

        return output


def __vital_algorithm_register__():
    from viame.utilities.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        FastFoundationStereo,
        "fast_foundation_stereo",
        "Stereo depth/disparity estimation using NVIDIA Fast-Foundation-Stereo "
        "(real-time variant)",
    )
