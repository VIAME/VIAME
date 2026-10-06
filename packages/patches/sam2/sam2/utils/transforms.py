# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import platform
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize, Resize, ToTensor


class _SafeNormalize(object):
    """Per-channel normalization, one channel at a time, avoiding the in-place
    broadcasting sub_/div_ that torchvision.transforms.Normalize uses.

    Those return garbage for tensors with spatial dimensions >= 128x128 --
    around a billion in magnitude, not a rounding error -- in a torch built
    from source by this tree. It is NOT an upstream bug and NOT a Windows
    bug, which is what this said before: on the same machine and the same
    Windows, `torch 2.12.1+cu126` with `torchvision 0.27.1+cu126` from the
    index is correct at every size from 64x64 to 1024x1024, as are 0.25.0,
    0.27.0 and 0.29.1. Only the local compilation is wrong.

    The fault is in torch, not torchvision: a bare
    `tensor.sub_(mean).div_(std)` fails the same way at 256x256 while
    `tensor.to(float32) / 255.0`, which broadcasts nothing, is fine. The
    threshold is where the vectorised/parallel path starts, so this reads as
    bad codegen for those kernels. `torchvision.transforms.v2.ToDtype(
    torch.float32, scale=True)` is the same bug seen from another angle, and
    it silently fed blank images to RF-DETR training in the v0.23.5 Windows
    desktop binaries -- 25 epochs to mAP 0.002 where an index build reaches
    0.24.

    So this is a workaround for a build defect, not for upstream. It can go
    when torch stops being built from source here, or when whatever
    optimisation flag miscompiles those kernels is found;
    `VIAME_BUILD_PYTORCH_FROM_SOURCE` is the switch.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        result = torch.zeros_like(tensor)
        for i in range(tensor.shape[0]):
            result[i] = (tensor[i] - self.mean[i]) / self.std[i]
        return result


class SAM2Transforms(nn.Module):
    def __init__(
        self, resolution, mask_threshold, max_hole_area=0.0, max_sprinkle_area=0.0
    ):
        """
        Transforms for SAM2.
        """
        super().__init__()
        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.max_hole_area = max_hole_area
        self.max_sprinkle_area = max_sprinkle_area
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.to_tensor = ToTensor()
        # A source-built torch miscompiles the broadcasting sub_/div_ behind
        # Normalize for spatial dims >= 128x128; see _SafeNormalize. Keep the
        # per-channel form outside the JIT-scripted pipeline. Gated on Windows
        # because that is where this tree builds torch from source.
        if platform.system() == "Windows":
            self._resize = torch.jit.script(
                nn.Sequential(
                    Resize((self.resolution, self.resolution)),
                )
            )
            self._normalize = _SafeNormalize(self.mean, self.std)
        else:
            self._resize = None
            self._normalize = None
            self.transforms = torch.jit.script(
                nn.Sequential(
                    Resize((self.resolution, self.resolution)),
                    Normalize(self.mean, self.std),
                )
            )

    def _apply_transforms(self, x):
        if self._resize is not None:
            return self._normalize(self._resize(x))
        return self.transforms(x)

    def __call__(self, x):
        x = self.to_tensor(x)
        return self._apply_transforms(x)

    def forward_batch(self, img_list):
        img_batch = [self._apply_transforms(self.to_tensor(img)) for img in img_list]
        img_batch = torch.stack(img_batch, dim=0)
        return img_batch

    def transform_coords(
        self, coords: torch.Tensor, normalize=False, orig_hw=None
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. The coordinates can be in absolute image or normalized coordinates,
        If the coords are in absolute image coordinates, normalize should be set to True and original image size is required.

        Returns
            Un-normalized coordinates in the range of [0, 1] which is expected by the SAM2 model.
        """
        if normalize:
            assert orig_hw is not None
            h, w = orig_hw
            coords = coords.clone()
            coords[..., 0] = coords[..., 0] / w
            coords[..., 1] = coords[..., 1] / h

        coords = coords * self.resolution  # unnormalize coords
        return coords

    def transform_boxes(
        self, boxes: torch.Tensor, normalize=False, orig_hw=None
    ) -> torch.Tensor:
        """
        Expects a tensor of shape Bx4. The coordinates can be in absolute image or normalized coordinates,
        if the coords are in absolute image coordinates, normalize should be set to True and original image size is required.
        """
        boxes = self.transform_coords(boxes.reshape(-1, 2, 2), normalize, orig_hw)
        return boxes

    def postprocess_masks(self, masks: torch.Tensor, orig_hw) -> torch.Tensor:
        """
        Perform PostProcessing on output masks.
        """
        from sam2.utils.misc import get_connected_components

        masks = masks.float()
        input_masks = masks
        mask_flat = masks.flatten(0, 1).unsqueeze(1)  # flatten as 1-channel image
        try:
            if self.max_hole_area > 0:
                # Holes are those connected components in background with area <= self.fill_hole_area
                # (background regions are those with mask scores <= self.mask_threshold)
                labels, areas = get_connected_components(
                    mask_flat <= self.mask_threshold
                )
                is_hole = (labels > 0) & (areas <= self.max_hole_area)
                is_hole = is_hole.reshape_as(masks)
                # We fill holes with a small positive mask score (10.0) to change them to foreground.
                masks = torch.where(is_hole, self.mask_threshold + 10.0, masks)

            if self.max_sprinkle_area > 0:
                labels, areas = get_connected_components(
                    mask_flat > self.mask_threshold
                )
                is_hole = (labels > 0) & (areas <= self.max_sprinkle_area)
                is_hole = is_hole.reshape_as(masks)
                # We fill holes with negative mask score (-10.0) to change them to background.
                masks = torch.where(is_hole, self.mask_threshold - 10.0, masks)
        except Exception as e:
            # Skip the post-processing step if the CUDA kernel fails
            warnings.warn(
                f"{e}\n\nSkipping the post-processing step due to the error above. You can "
                "still use SAM 2 and it's OK to ignore the error above, although some post-processing "
                "functionality may be limited (which doesn't affect the results in most cases; see "
                "https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).",
                category=UserWarning,
                stacklevel=2,
            )
            masks = input_masks

        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        return masks
