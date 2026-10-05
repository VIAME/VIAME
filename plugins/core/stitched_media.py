# This file is part of VIAME, and is distributed under an OSI-approved
# BSD 3-Clause License. See either the root top-level LICENSE file or
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

"""
Stitched stereo media: each frame holds the left camera in its left half and
the right camera in its right half.

A media path ending in ``#stitched=left`` or ``#stitched=right`` names one of
those halves. The interactive services accept such paths wherever they accept
an image or video path, load the underlying file and keep only that half.
"""

from typing import Optional, Tuple

import numpy as np

STITCHED_TAG = "#stitched="
STITCHED_SIDES = ("left", "right")


def split_stitched_path(path: str) -> Tuple[str, Optional[str]]:
    """Return (file path, side); side is None when the path is not tagged."""
    index = path.rfind(STITCHED_TAG) if path else -1
    if index == -1:
        return path, None
    side = path[index + len(STITCHED_TAG):]
    if side not in STITCHED_SIDES:
        return path, None
    return path[:index], side


def crop_stitched_array(array: np.ndarray, side: Optional[str]) -> np.ndarray:
    """
    Keep one half of an image array shaped (height, width[, channels]).

    Both halves share one width so the two cameras agree on frame size; an
    odd-width frame drops its middle column.
    """
    if side is None:
        return array
    if side not in STITCHED_SIDES:
        raise ValueError(f"stitched side must be one of {STITCHED_SIDES}, got {side!r}")
    width = array.shape[1]
    half = width // 2
    start = 0 if side == "left" else width - half
    return np.ascontiguousarray(array[:, start:start + half])


def crop_stitched_container(container, side: Optional[str]):
    """Keep one half of a vital ImageContainer."""
    if side is None or container is None:
        return container
    from kwiver.vital.types import Image, ImageContainer
    return ImageContainer(Image(
        crop_stitched_array(np.asarray(container.image().asarray()), side)))
