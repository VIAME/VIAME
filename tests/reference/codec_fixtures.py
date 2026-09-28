"""The encoded image files the codec golden decodes.

Phase 7 replaces OpenCV's `imread`/`imwrite` with stb plus an in-house TIFF
reader, so what has to be pinned first is what OpenCV decodes each container
to, byte for byte. These are the containers: the formats and the encoding
options VIAME's own data and the shipped pipelines actually present.

The pixel content comes from `fixtures.py`, so a codec fixture and a filter
fixture are the same picture in different wrappers -- a decoded PNG can be
compared against `inputs/rgb8.png` directly.

Generated once and committed. `build()` is kept so that the set can be
regenerated the same way, and needs Pillow for everything but the tiled TIFF,
which needs ImageMagick's `convert`: Pillow writes only stripped TIFFs and a
tiled one is the case the in-house reader is expected to hand off rather than
decode itself.
"""

import os
import subprocess

import numpy as np

import fixtures

SEED = fixtures.SEED


def _sources():
    """The pixel arrays the containers wrap, by name."""
    rng = np.random.default_rng(SEED)

    rgb8 = fixtures.rgb8(rng)
    gray8 = fixtures.gray8(rng)
    gray16 = fixtures.gray16(rng)

    # Alpha that is neither constant nor a copy of a colour channel, so that
    # a decoder dropping or reordering it shows up.
    ys, xs = np.mgrid[0:rgb8.shape[0], 0:rgb8.shape[1]]
    alpha = np.clip(40 + 215 * (xs / (rgb8.shape[1] - 1.0))
                    * (1.0 - 0.5 * ys / (rgb8.shape[0] - 1.0)), 0, 255)
    rgba8 = np.dstack([rgb8, alpha.astype(np.uint8)])

    return {"rgb8": rgb8, "gray8": gray8, "gray16": gray16, "rgba8": rgba8}


# name -> (source array, file extension, Pillow save options)
#
# `compression` values are Pillow's spellings of the TIFF compression tags:
# none is tag 1, tiff_lzw is 5, packbits is 32773.
CONTAINERS = {
    "png_gray8":            ("gray8",  ".png", {}),
    "png_gray16":           ("gray16", ".png", {}),
    "png_rgb8":             ("rgb8",   ".png", {}),
    "png_rgba8":            ("rgba8",  ".png", {}),

    # Quality 92 rather than the default, so the fixture is not near-lossless
    # and a decoder difference is visible, but not so low that it is noise.
    #
    # Two colour cases, because they test different things. `jpg_rgb8` is
    # 4:2:0, which is what a camera or a video frame gives you and what the
    # shipped data is; on a synthetic image with a sharply inverted blue
    # channel the chroma subsampling alone moves a pixel by over a hundred
    # counts, which would hide a decoder difference underneath it.
    # `jpg_rgb8_444` keeps full chroma, so what is left between two decoders
    # is the IDCT and the upsampling filter -- the thing the tolerance is
    # actually about.
    "jpg_gray8":            ("gray8",  ".jpg", {"quality": 92}),
    "jpg_rgb8":             ("rgb8",   ".jpg", {"quality": 92}),
    "jpg_rgb8_444":         ("rgb8",   ".jpg", {"quality": 92,
                                                "subsampling": 0}),

    "bmp_gray8":            ("gray8",  ".bmp", {}),
    "bmp_rgb8":             ("rgb8",   ".bmp", {}),

    "tiff_gray8_raw":       ("gray8",  ".tif", {"compression": None}),
    "tiff_gray8_lzw":       ("gray8",  ".tif", {"compression": "tiff_lzw"}),
    "tiff_gray8_packbits":  ("gray8",  ".tif", {"compression": "packbits"}),
    "tiff_gray16_raw":      ("gray16", ".tif", {"compression": None}),
    "tiff_gray16_lzw":      ("gray16", ".tif", {"compression": "tiff_lzw"}),
    "tiff_rgb8_raw":        ("rgb8",   ".tif", {"compression": None}),
    "tiff_rgb8_lzw":        ("rgb8",   ".tif", {"compression": "tiff_lzw"}),
    "tiff_rgb8_packbits":   ("rgb8",   ".tif", {"compression": "packbits"}),
}

# Written with ImageMagick rather than Pillow, because Pillow writes strips
# only. 16 by 16 tiles over a 96 by 64 image is six by four tiles, so the
# right edge is a partial tile column and a reader that assumes whole tiles
# gets it wrong.
TILED = {
    "tiff_rgb8_tiled": ("rgb8", ".tif", "16x16"),
}

# Also ImageMagick, for the same reason: Pillow always writes predictor 1.
# Horizontal differencing is what makes LZW worth using on a 16 bit image and
# is the one TIFF feature here that a reader can get subtly wrong -- undo it
# per row and per sample or the picture is a smear.
PREDICTED = {
    "tiff_gray16_lzw_predictor": ("gray16", ".tif", 16),
    "tiff_rgb8_lzw_predictor": ("rgb8", ".tif", 8),
}


def _pillow_mode(array):
    if array.ndim == 2:
        return "I;16" if array.dtype == np.uint16 else "L"
    return "RGBA" if array.shape[2] == 4 else "RGB"


def build(target_dir):
    """Write every container into `target_dir`; return [(name, path)]."""
    from PIL import Image

    os.makedirs(target_dir, exist_ok=True)
    sources = _sources()
    written = []

    for name, (source, extension, options) in sorted(CONTAINERS.items()):
        array = sources[source]
        path = os.path.join(target_dir, name + extension)
        Image.fromarray(array, mode=_pillow_mode(array)).save(path, **options)
        written.append((name, path))

    for name, (source, extension, depth) in sorted(PREDICTED.items()):
        array = sources[source]
        path = os.path.join(target_dir, name + extension)
        staging = os.path.join(target_dir, "." + name + ".png")
        Image.fromarray(array, mode=_pillow_mode(array)).save(staging)
        subprocess.run(
            ["convert", staging, "-depth", str(depth),
             "-define", "tiff:predictor=2",
             "-compress", "lzw", path],
            check=True)
        os.remove(staging)
        written.append((name, path))

    for name, (source, extension, geometry) in sorted(TILED.items()):
        array = sources[source]
        path = os.path.join(target_dir, name + extension)
        staging = os.path.join(target_dir, "." + name + ".png")
        Image.fromarray(array, mode=_pillow_mode(array)).save(staging)
        subprocess.run(
            ["convert", staging,
             "-define", "tiff:tile-geometry=" + geometry,
             "-compress", "none", path],
            check=True)
        os.remove(staging)
        written.append((name, path))

    return written


def paths(target_dir):
    """[(name, path)] for the containers as committed, without writing any."""
    out = []
    for name, (_, extension, _) in sorted(CONTAINERS.items()):
        out.append((name, os.path.join(target_dir, name + extension)))
    for name, (_, extension, _) in sorted(PREDICTED.items()):
        out.append((name, os.path.join(target_dir, name + extension)))
    for name, (_, extension, _) in sorted(TILED.items()):
        out.append((name, os.path.join(target_dir, name + extension)))
    return out
