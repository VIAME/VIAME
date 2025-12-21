# ckwg +29
# Copyright 2025 by Kitware, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither name of Kitware, Inc. nor the names of any contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
KWIVER Sprokit process wrapper for NVIDIA Foundation-Stereo disparity estimation.

This process accepts left and right stereo image pairs and outputs disparity maps,
with optional depth map and point cloud generation.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import numpy as np

from kwiver.sprokit.processes.kwiver_process import KwiverProcess
from kwiver.sprokit.pipeline import process

from kwiver.vital.types import ImageContainer
from kwiver.vital.util import VitalPIL
from PIL import Image as PILImage


def vital_image_container_from_ndarray(ndarray_img):
    """Convert numpy array to KWIVER ImageContainer.

    Args:
        ndarray_img (np.ndarray): Input image as an ndarray.

    Returns:
        kwiver.vital.types.ImageContainer
    """
    pil_img = PILImage.fromarray(ndarray_img)
    vital_img = ImageContainer(VitalPIL.from_pil(pil_img))
    return vital_img


# ------------------------------------------------------------------------------
class FoundationStereoProcess(KwiverProcess):
    """
    Sprokit process for stereo disparity estimation using NVIDIA Foundation-Stereo.

    This process takes left and right stereo images as input and produces
    disparity maps. Optionally, it can also compute depth maps (when camera
    parameters are provided) and generate 3D point clouds.

    Input Ports:
        left_image: Left stereo image (required)
        right_image: Right stereo image (required)
        timestamp: Frame timestamp (optional, passed through)

    Output Ports:
        disparity_image: Disparity map as float32 image
        depth_image: Depth map as float32 image (when enabled)
        point_cloud_file: Path to PLY point cloud file (when enabled)
        timestamp: Passed-through timestamp
    """

    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

        # Model configuration
        self.add_config_trait("checkpoint_path", "checkpoint_path",
            '',
            'Path to the pretrained model checkpoint (.pth file)')
        self.declare_config_using_trait("checkpoint_path")

        self.add_config_trait("vit_size", "vit_size",
            'vitl',
            'Vision Transformer backbone size: vitl (large), vitb (base), or vits (small)')
        self.declare_config_using_trait("vit_size")

        self.add_config_trait("device", "device",
            'cuda:0',
            'Device to run inference on (e.g., cuda:0, cuda:1, cpu)')
        self.declare_config_using_trait("device")

        # Inference parameters
        self.add_config_trait("num_iters", "num_iters",
            '32',
            'Number of GRU refinement iterations during inference')
        self.declare_config_using_trait("num_iters")

        self.add_config_trait("use_hierarchical", "use_hierarchical",
            'False',
            'Use hierarchical inference for high-resolution images (>1K pixels)')
        self.declare_config_using_trait("use_hierarchical")

        self.add_config_trait("hierarchical_ratio", "hierarchical_ratio",
            '0.5',
            'Scale ratio for first pass in hierarchical inference')
        self.declare_config_using_trait("hierarchical_ratio")

        self.add_config_trait("mixed_precision", "mixed_precision",
            'True',
            'Use mixed precision (FP16) inference for faster computation')
        self.declare_config_using_trait("mixed_precision")

        self.add_config_trait("low_memory", "low_memory",
            'False',
            'Enable low memory mode for limited GPU RAM')
        self.declare_config_using_trait("low_memory")

        # Output control
        self.add_config_trait("output_depth", "output_depth",
            'False',
            'Output depth map (requires focal_length and baseline)')
        self.declare_config_using_trait("output_depth")

        self.add_config_trait("output_point_cloud", "output_point_cloud",
            'False',
            'Output 3D point cloud PLY file (requires focal_length and baseline)')
        self.declare_config_using_trait("output_point_cloud")

        # Camera parameters for depth/point cloud
        self.add_config_trait("focal_length", "focal_length",
            '0.0',
            'Focal length in pixels (for depth conversion)')
        self.declare_config_using_trait("focal_length")

        self.add_config_trait("baseline", "baseline",
            '0.0',
            'Stereo camera baseline in meters (for depth conversion)')
        self.declare_config_using_trait("baseline")

        self.add_config_trait("principal_x", "principal_x",
            '0.0',
            'Principal point X coordinate (defaults to image center if 0)')
        self.declare_config_using_trait("principal_x")

        self.add_config_trait("principal_y", "principal_y",
            '0.0',
            'Principal point Y coordinate (defaults to image center if 0)')
        self.declare_config_using_trait("principal_y")

        self.add_config_trait("z_far", "z_far",
            '10.0',
            'Maximum depth to include in point cloud (meters)')
        self.declare_config_using_trait("z_far")

        self.add_config_trait("point_cloud_dir", "point_cloud_dir",
            '',
            'Directory to save point cloud PLY files (required if output_point_cloud is enabled)')
        self.declare_config_using_trait("point_cloud_dir")

        self.add_config_trait("remove_invisible", "remove_invisible",
            'True',
            'Remove non-overlapping observations from point cloud')
        self.declare_config_using_trait("remove_invisible")

        # Add custom port traits for stereo inputs
        self.add_port_trait("left_image", "image", "Left stereo image")
        self.add_port_trait("right_image", "image", "Right stereo image")
        self.add_port_trait("disparity_image", "image", "Output disparity map")
        self.add_port_trait("depth_image", "image", "Output depth map")
        self.add_port_trait("point_cloud_file", "string", "Path to output PLY file")

        # Set up port flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        # Input ports
        self.declare_input_port_using_trait('left_image', required)
        self.declare_input_port_using_trait('right_image', required)
        self.declare_input_port_using_trait('timestamp', optional)

        # Output ports
        self.declare_output_port_using_trait('disparity_image', optional)
        self.declare_output_port_using_trait('depth_image', optional)
        self.declare_output_port_using_trait('point_cloud_file', optional)
        self.declare_output_port_using_trait('timestamp', optional)

    # --------------------------------------------------------------------------
    def _configure(self):
        import torch

        # Read configuration values
        self._checkpoint_path = str(self.config_value('checkpoint_path'))
        self._vit_size = str(self.config_value('vit_size'))
        self._device = str(self.config_value('device'))
        self._num_iters = int(self.config_value('num_iters'))
        self._use_hierarchical = str(self.config_value('use_hierarchical')).lower() == 'true'
        self._hierarchical_ratio = float(self.config_value('hierarchical_ratio'))
        self._mixed_precision = str(self.config_value('mixed_precision')).lower() == 'true'
        self._low_memory = str(self.config_value('low_memory')).lower() == 'true'
        self._output_depth = str(self.config_value('output_depth')).lower() == 'true'
        self._output_point_cloud = str(self.config_value('output_point_cloud')).lower() == 'true'
        self._focal_length = float(self.config_value('focal_length'))
        self._baseline = float(self.config_value('baseline'))
        self._principal_x = float(self.config_value('principal_x'))
        self._principal_y = float(self.config_value('principal_y'))
        self._z_far = float(self.config_value('z_far'))
        self._point_cloud_dir = str(self.config_value('point_cloud_dir'))
        self._remove_invisible = str(self.config_value('remove_invisible')).lower() == 'true'

        # Validate checkpoint path
        if not self._checkpoint_path:
            raise RuntimeError("checkpoint_path must be specified")
        if not os.path.exists(self._checkpoint_path):
            raise RuntimeError(f"Checkpoint file not found: {self._checkpoint_path}")

        # Validate depth/point cloud requirements
        if (self._output_depth or self._output_point_cloud) and \
           (self._focal_length <= 0 or self._baseline <= 0):
            print("Warning: focal_length and baseline required for depth/point cloud output")

        # Add foundation-stereo to path
        foundation_stereo_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'packages', 'pytorch-libs', 'foundation-stereo'
        )
        if foundation_stereo_dir not in sys.path:
            sys.path.insert(0, foundation_stereo_dir)

        # Import foundation-stereo modules
        from omegaconf import OmegaConf
        from core.foundation_stereo import FoundationStereo
        from core.utils.utils import InputPadder

        # Store InputPadder class for later use
        self._InputPadder = InputPadder

        # Load model configuration
        ckpt_dir = os.path.dirname(self._checkpoint_path)
        cfg_path = os.path.join(ckpt_dir, 'cfg.yaml')
        if os.path.exists(cfg_path):
            cfg = OmegaConf.load(cfg_path)
        else:
            cfg = OmegaConf.create({})

        # Set vit_size
        cfg['vit_size'] = self._vit_size

        # Create model
        self._model = FoundationStereo(cfg)

        # Load checkpoint
        ckpt = torch.load(self._checkpoint_path, map_location='cpu')
        self._model.load_state_dict(ckpt['model'])

        # Move to device and set eval mode
        self._torch_device = torch.device(self._device)
        self._model.to(self._torch_device)
        self._model.eval()

        # Disable gradients
        torch.set_grad_enabled(False)

        # Frame counter for point cloud naming
        self._frame_counter = 0

        # Create point cloud output directory if needed
        if self._output_point_cloud:
            if not self._point_cloud_dir:
                raise RuntimeError("point_cloud_dir must be specified when output_point_cloud is enabled")
            os.makedirs(self._point_cloud_dir, exist_ok=True)

        self._base_configure()

    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    def _step(self):
        import torch

        # Get input images
        left_img_c = self.grab_input_using_trait('left_image')
        right_img_c = self.grab_input_using_trait('right_image')

        # Get optional timestamp
        timestamp = None
        if self.has_input_port_edge_using_trait('timestamp'):
            timestamp = self.grab_input_using_trait('timestamp')

        # Convert to numpy arrays
        left_npy = self._format_image(left_img_c)
        right_npy = self._format_image(right_img_c)

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
        with torch.cuda.amp.autocast(self._mixed_precision):
            if self._use_hierarchical:
                disp = self._model.run_hierachical(
                    left_padded, right_padded,
                    iters=self._num_iters,
                    test_mode=True,
                    small_ratio=self._hierarchical_ratio
                )
            else:
                disp = self._model.forward(
                    left_padded, right_padded,
                    iters=self._num_iters,
                    test_mode=True,
                    low_memory=self._low_memory
                )

        # Unpad and convert to numpy
        disp = padder.unpad(disp.float())
        disp_npy = disp.data.cpu().numpy().reshape(H, W)

        # Handle invisible regions if requested
        if self._remove_invisible:
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            us_right = xx - disp_npy
            invalid = us_right < 0
            disp_npy_filtered = disp_npy.copy()
            disp_npy_filtered[invalid] = np.inf
        else:
            disp_npy_filtered = disp_npy

        # Compute depth if needed
        depth_npy = None
        if self._output_depth or self._output_point_cloud:
            if self._focal_length > 0 and self._baseline > 0:
                # Avoid division by zero
                safe_disp = np.where(disp_npy_filtered > 0, disp_npy_filtered, 1e-6)
                depth_npy = (self._focal_length * self._baseline) / safe_disp
                depth_npy = np.where(disp_npy_filtered > 0, depth_npy, 0)

        # Generate point cloud if requested
        point_cloud_path = ""
        if self._output_point_cloud and depth_npy is not None:
            try:
                import open3d as o3d

                # Build camera intrinsic matrix
                cx = self._principal_x if self._principal_x > 0 else W / 2.0
                cy = self._principal_y if self._principal_y > 0 else H / 2.0
                K = np.array([
                    [self._focal_length, 0, cx],
                    [0, self._focal_length, cy],
                    [0, 0, 1]
                ], dtype=np.float32)

                # Import point cloud utilities
                from Utils import depth2xyzmap, toOpen3dCloud

                xyz_map = depth2xyzmap(depth_npy, K)
                pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), left_npy.reshape(-1, 3))

                # Filter by depth
                keep_mask = (np.asarray(pcd.points)[:, 2] > 0) & \
                           (np.asarray(pcd.points)[:, 2] <= self._z_far)
                keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
                pcd = pcd.select_by_index(keep_ids)

                # Save point cloud
                point_cloud_path = os.path.join(
                    self._point_cloud_dir,
                    f'point_cloud_{self._frame_counter:06d}.ply'
                )
                o3d.io.write_point_cloud(point_cloud_path, pcd)

            except ImportError:
                print("Warning: Open3D not available, skipping point cloud output")
            except Exception as e:
                print(f"Warning: Failed to generate point cloud: {e}")

        self._frame_counter += 1

        # Output disparity image if port is connected
        if self.count_output_port_edges('disparity_image') > 0:
            # Convert to float32 for output (preserves precision)
            disp_output = disp_npy.astype(np.float32)
            disp_container = vital_image_container_from_ndarray(
                (disp_output * 256).clip(0, 65535).astype(np.uint16)
            )
            self.push_to_port_using_trait('disparity_image', disp_container)

        # Output depth image
        if self._output_depth and depth_npy is not None:
            # Scale depth to uint16 (millimeters)
            depth_mm = (depth_npy * 1000).clip(0, 65535).astype(np.uint16)
            depth_container = vital_image_container_from_ndarray(depth_mm)
            self.push_to_port_using_trait('depth_image', depth_container)
        else:
            self.push_to_port_using_trait('depth_image', ImageContainer())

        # Output point cloud path
        self.push_to_port_using_trait('point_cloud_file', point_cloud_path)

        # Pass through timestamp
        if timestamp is not None:
            self.push_to_port_using_trait('timestamp', timestamp)

        self._base_step()


# ==============================================================================
def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory

    module_name = 'python:viame.pytorch.FoundationStereoProcess'

    if process_factory.is_process_module_loaded(module_name):
        return

    process_factory.add_process(
        'foundation_stereo',
        'Stereo disparity estimation using NVIDIA Foundation-Stereo model',
        FoundationStereoProcess
    )

    process_factory.mark_process_module_as_loaded(module_name)
