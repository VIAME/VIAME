"""The handful of image primitives VIAME's python used OpenCV for.

`opencv-python-headless` was a hard dependency of the wheel for this: 789
call sites across 96 files, almost all of them reading an image, converting
a colour space, or resizing. None of that needs OpenCV. Pillow and numpy are
already dependencies and cover it.

What is deliberately *not* here: homography estimation, camera calibration
and chessboard detection. Those are OpenCV algorithms rather than
primitives, and reimplementing them is a numerical project, not a port. They
stay behind the `opencv` extra -- see `viame.utilities.opencv`.

Colour handling follows OpenCV's conventions so that ported code keeps
working: the ITU-R BT.601 luma weights it uses for grayscale, and its
channel order where a caller was relying on it.
"""

import numpy as np

__all__ = [
    "read_image", "write_image", "to_gray", "to_rgb",
    "swap_channels", "resize", "INTER_NEAREST", "INTER_LINEAR",
    "INTER_CUBIC", "INTER_AREA", "INTER_LANCZOS",
]

INTER_NEAREST = "nearest"
INTER_LINEAR = "bilinear"
INTER_CUBIC = "bicubic"
INTER_AREA = "area"
INTER_LANCZOS = "lanczos"

# BT.601, the weights cv2.cvtColor uses for COLOR_*2GRAY. Kept explicit so a
# reader can see this matches rather than having to trust it.
_LUMA = (0.299, 0.587, 0.114)


def _pil():
    from PIL import Image
    return Image


def read_image(path, grayscale=False):
    """An image as a numpy array, RGB or 2-D grayscale.

    Note the channel order: OpenCV's `imread` returns BGR, this returns RGB.
    A caller that was indexing channels or passing the result straight to
    another cv2 call needs `swap_channels`; one that was only measuring or
    displaying does not.
    """
    image = _pil().open(str(path))
    image = image.convert("L" if grayscale else "RGB")
    return np.asarray(image)


def read_unchanged(path):
    """The image exactly as stored: bit depth, channels and all.

    `cv2.imread` with `IMREAD_UNCHANGED | IMREAD_ANYDEPTH`. Nothing is
    converted -- a 16-bit single channel thermal frame stays 16-bit and
    single channel, an 8-bit colour one comes back three channel **RGB**,
    and an alpha channel survives as a fourth. The caller decides what to do
    with what it got, which is the point of asking for unchanged.
    """
    image = _pil().open(str(path))
    return np.asarray(image)


def write_image(path, array):
    """Write a 2-D grayscale or 3-D RGB array."""
    array = np.asarray(array)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    mode = "L" if array.ndim == 2 else "RGB"
    _pil().fromarray(array, mode=mode).save(str(path))


def encode_image(array, suffix=".png", quality=None):
    """An encoded image as bytes, which is `cv2.imencode`.

    `suffix` names the format the way `cv2.imencode` does, by file
    extension. `quality` is JPEG quality 0..100 where it applies, matching
    `cv2.IMWRITE_JPEG_QUALITY`; OpenCV's default is 95 and Pillow's is 75,
    so it is passed explicitly rather than left to differ.
    """
    import io

    array = np.asarray(array)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    mode = "L" if array.ndim == 2 else "RGB"
    image = _pil().fromarray(array, mode=mode)

    formats = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG",
               ".bmp": "BMP", ".tif": "TIFF", ".tiff": "TIFF",
               ".webp": "WEBP"}
    suffix = suffix.lower()
    if suffix not in formats:
        raise ValueError("cannot encode {!r}".format(suffix))

    options = {}
    if formats[suffix] == "JPEG":
        options["quality"] = 95 if quality is None else int(quality)

    buffer = io.BytesIO()
    image.save(buffer, format=formats[suffix], **options)
    return buffer.getvalue()


def decode_image(data, grayscale=False):
    """An image decoded from bytes, which is `cv2.imdecode`.

    RGB, not BGR, for the same reason `read_image` is.
    """
    import io

    image = _pil().open(io.BytesIO(bytes(data)))
    image = image.convert("L" if grayscale else "RGB")
    return np.asarray(image)


def decode_unchanged(data):
    """Bytes decoded exactly as stored, which is `cv2.imdecode` with
    `IMREAD_UNCHANGED` -- except that an alpha image comes back **RGBA**
    where cv2 hands back BGRA.
    """
    import io

    return np.asarray(_pil().open(io.BytesIO(bytes(data))))


def to_gray(array):
    """RGB to grayscale, by the same luma weights OpenCV uses."""
    array = np.asarray(array)
    if array.ndim == 2:
        return array
    weights = np.asarray(_LUMA, dtype=np.float32)
    gray = array[..., :3].astype(np.float32) @ weights
    return np.rint(gray).clip(0, 255).astype(array.dtype)


def to_rgb(array):
    """Grayscale to three identical channels."""
    array = np.asarray(array)
    if array.ndim == 3:
        return array
    return np.repeat(array[..., None], 3, axis=2)


def swap_channels(array):
    """RGB to BGR, or back. The same operation either way."""
    array = np.asarray(array)
    if array.ndim != 3:
        return array
    return array[..., ::-1]


def resize(array, width, height, interpolation=INTER_LINEAR):
    """Resize to an exact size.

    `INTER_AREA` maps to Pillow's box filter, which is what it is: an
    average over the source pixels covering each destination pixel.
    """
    Image = _pil()
    filters = {
        INTER_NEAREST: Image.NEAREST,
        INTER_LINEAR: Image.BILINEAR,
        INTER_CUBIC: Image.BICUBIC,
        INTER_AREA: Image.BOX,
        INTER_LANCZOS: Image.LANCZOS,
    }
    if interpolation not in filters:
        raise ValueError(f"unknown interpolation: {interpolation!r}")

    array = np.asarray(array)
    mode = "L" if array.ndim == 2 else "RGB"
    source = array if array.dtype == np.uint8 else np.clip(array, 0, 255).astype(np.uint8)
    out = Image.fromarray(source, mode=mode).resize(
        (int(width), int(height)), filters[interpolation])
    return np.asarray(out)
