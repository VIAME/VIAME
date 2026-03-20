# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Triton kernel for euclidean distance transform (EDT)

On Windows, Triton is not available, so we fall back to an OpenCV-based
CPU implementation of the distance transform.
"""

import platform
import torch

_TRITON_AVAILABLE = False
if platform.system() != "Windows":
    try:
        import triton
        import triton.language as tl
        _TRITON_AVAILABLE = True
    except ImportError:
        pass


def _edt_opencv_fallback(data: torch.Tensor) -> torch.Tensor:
    """
    Fallback EDT using OpenCV's distanceTransform (CPU).

    Equivalent to edt_triton but runs on CPU via OpenCV when Triton
    is unavailable (e.g. on Windows).
    """
    import cv2
    import numpy as np

    assert data.dim() == 3
    B, H, W = data.shape
    device = data.device
    result = torch.zeros_like(data, dtype=torch.float32)

    data_cpu = data.cpu().numpy()
    for i in range(B):
        # cv2.distanceTransform expects 8-bit single-channel, 0 = foreground
        mask = (data_cpu[i] == 0).astype(np.uint8)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 0)
        result[i] = torch.from_numpy(dist)

    return result.to(device)


if _TRITON_AVAILABLE:
    @triton.jit
    def edt_kernel(inputs_ptr, outputs_ptr, v, z, height, width, horizontal: tl.constexpr):
        batch_id = tl.program_id(axis=0)
        if horizontal:
            row_id = tl.program_id(axis=1)
            block_start = (batch_id * height * width) + row_id * width
            length = width
            stride = 1
        else:
            col_id = tl.program_id(axis=1)
            block_start = (batch_id * height * width) + col_id
            length = height
            stride = width

        k = 0
        for q in range(1, length):
            cur_input = tl.load(inputs_ptr + block_start + (q * stride))
            r = tl.load(v + block_start + (k * stride))
            z_k = tl.load(z + block_start + (k * stride))
            previous_input = tl.load(inputs_ptr + block_start + (r * stride))
            s = (cur_input - previous_input + q * q - r * r) / (q - r) / 2

            while s <= z_k and k - 1 >= 0:
                k = k - 1
                r = tl.load(v + block_start + (k * stride))
                z_k = tl.load(z + block_start + (k * stride))
                previous_input = tl.load(inputs_ptr + block_start + (r * stride))
                s = (cur_input - previous_input + q * q - r * r) / (q - r) / 2

            k = k + 1
            tl.store(v + block_start + (k * stride), q)
            tl.store(z + block_start + (k * stride), s)
            if k + 1 < length:
                tl.store(z + block_start + ((k + 1) * stride), 1e9)

        k = 0
        for q in range(length):
            while (
                k + 1 < length
                and tl.load(
                    z + block_start + ((k + 1) * stride), mask=(k + 1) < length, other=q
                )
                < q
            ):
                k += 1
            r = tl.load(v + block_start + (k * stride))
            d = q - r
            old_value = tl.load(inputs_ptr + block_start + (r * stride))
            tl.store(outputs_ptr + block_start + (q * stride), old_value + d * d)

    def edt_triton(data: torch.Tensor):
        """
        Computes the Euclidean Distance Transform (EDT) of a batch of binary images.

        Args:
            data: A tensor of shape (B, H, W) representing a batch of binary images.

        Returns:
            A tensor of the same shape as data containing the EDT.
            It should be equivalent to a batched version of cv2.distanceTransform(input, cv2.DIST_L2, 0)
        """
        assert data.dim() == 3
        assert data.is_cuda
        B, H, W = data.shape
        data = data.contiguous()

        output = torch.where(data, 1e18, 0.0)
        assert output.is_contiguous()

        parabola_loc = torch.zeros(B, H, W, dtype=torch.uint32, device=data.device)
        parabola_inter = torch.empty(B, H, W, dtype=torch.float, device=data.device)
        parabola_inter[:, :, 0] = -1e18
        parabola_inter[:, :, 1] = 1e18

        grid = (B, H)
        edt_kernel[grid](
            output.clone(),
            output,
            parabola_loc,
            parabola_inter,
            H,
            W,
            horizontal=True,
        )

        parabola_loc.zero_()
        parabola_inter[:, :, 0] = -1e18
        parabola_inter[:, :, 1] = 1e18

        grid = (B, W)
        edt_kernel[grid](
            output.clone(),
            output,
            parabola_loc,
            parabola_inter,
            H,
            W,
            horizontal=False,
        )
        return output.sqrt()

else:
    # Windows / no-Triton fallback
    def edt_triton(data: torch.Tensor):
        """EDT fallback using OpenCV when Triton is unavailable."""
        return _edt_opencv_fallback(data)
