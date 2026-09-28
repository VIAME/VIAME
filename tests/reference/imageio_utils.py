"""Lossless array/PNG round trip for the golden fixtures and recordings.

PNG holds the 8 and 16 bit results exactly and stays viewable; everything
else (float temporal variance, bool masks, multi-channel float) goes to a
compressed .npz, which is exact for any dtype and shape. Nothing here is
allowed to rescale or reorder channels.
"""

import numpy as np
from PIL import Image


def _png_mode(array):
    if array.ndim == 2:
        if array.dtype == np.uint8:
            return "L"
        if array.dtype == np.uint16:
            return "I;16"
    if array.ndim == 3 and array.dtype == np.uint8:
        if array.shape[2] == 3:
            return "RGB"
        if array.shape[2] == 4:
            return "RGBA"
    return None


def save(path, array):
    """Write `array` to `path` without its extension; return the file written."""
    array = np.ascontiguousarray(array)

    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]

    mode = _png_mode(array)

    if mode is None:
        target = str(path) + ".npz"
        np.savez_compressed(target, image=array)
        return target

    target = str(path) + ".png"
    Image.fromarray(array, mode=mode).save(target)
    return target


def load(path):
    """Read back a file written by `save`."""
    path = str(path)

    if path.endswith(".npz"):
        with np.load(path) as data:
            return data["image"]

    if path.endswith(".npy"):
        return np.load(path)

    image = Image.open(path)
    array = np.array(image)

    # Pillow reads I;16 as int32 on some builds; the file is 16 bit either way
    if image.mode == "I;16" and array.dtype != np.uint16:
        array = array.astype(np.uint16)

    return array
