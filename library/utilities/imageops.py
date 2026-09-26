"""Image I/O and array helpers for VIAME's converted Python callers.

Pillow handles image codecs. Native VIAME kernels provide grayscale and
nearest, bilinear, bicubic and area resizing with the conventions used by
OpenCV callers. The helper's Lanczos option retains Pillow's filter.

Colour arrays use RGB channel order. Supported native pixel types are
uint8, uint16 and float32; conversion and resizing preserve that dtype.
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
    with _pil().open(str(path)) as image:
        return np.array(image.convert("L" if grayscale else "RGB"))


def read_unchanged(path):
    """The image exactly as stored: bit depth, channels and all.

    `cv2.imread` with `IMREAD_UNCHANGED | IMREAD_ANYDEPTH`. Nothing is
    converted -- a 16-bit single channel thermal frame stays 16-bit and
    single channel, an 8-bit colour one comes back three channel **RGB**,
    and an alpha channel survives as a fourth. The caller decides what to do
    with what it got, which is the point of asking for unchanged.
    """
    with _pil().open(str(path)) as image:
        return _unchanged_array(image)


def _unchanged_array(image):
    if image.mode == "P":
        image = image.convert("RGBA" if "transparency" in image.info else "RGB")
    array = np.array(image)
    # Older Pillow versions expose 16-bit PNG data as mode I / int32.
    if image.mode == "I" and image.format == "PNG":
        array = array.astype(np.uint16)
    return array


def _image_from_array(array):
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[..., 0]
    if array.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
        # Keep the existing conversion for floating-point display images.
        array = np.clip(array, 0, 255).astype(np.uint8)
    # Let Pillow infer L, RGB, RGBA or I;16 from shape and dtype. Forcing
    # RGB reinterprets RGBA bytes and destroys both colours and alpha.
    return _pil().fromarray(array)


def write_image(path, array):
    """Write an image preserving its supported bit depth and channels.

    Return True on success; encoding and filesystem errors raise.
    """
    options = {"quality": 95} if str(path).lower().endswith((".jpg", ".jpeg")) else {}
    _image_from_array(array).save(str(path), **options)
    return True


def encode_image(array, suffix=".png", quality=None):
    """An encoded image as bytes, which is `cv2.imencode`.

    `suffix` names the format the way `cv2.imencode` does, by file
    extension. `quality` is JPEG quality 0..100 where it applies, matching
    `cv2.IMWRITE_JPEG_QUALITY`; OpenCV's default is 95 and Pillow's is 75,
    so it is passed explicitly rather than left to differ.
    """
    import io

    image = _image_from_array(array)

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

    with _pil().open(io.BytesIO(bytes(data))) as image:
        return np.array(image.convert("L" if grayscale else "RGB"))


def decode_unchanged(data):
    """Bytes decoded exactly as stored, which is `cv2.imdecode` with
    `IMREAD_UNCHANGED` -- except that an alpha image comes back **RGBA**
    where cv2 hands back BGRA.
    """
    import io

    with _pil().open(io.BytesIO(bytes(data))) as image:
        return _unchanged_array(image)


def to_gray(array):
    """RGB to grayscale, by the same luma weights OpenCV uses."""
    array = np.asarray(array)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] == 1:
        return array[..., 0]
    from viame import image_kernels
    return image_kernels.to_gray(np.ascontiguousarray(array))


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
    """Resize with the native kernels, preserving uint8, uint16 or float32.

    Lanczos retains the Pillow implementation used by this helper; the
    native kernels implement nearest, bilinear, bicubic and area.
    """
    array = np.asarray(array)
    if array.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32)):
        raise TypeError("resize expects uint8, uint16 or float32")
    if interpolation in (INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA):
        from viame import image_kernels
        return image_kernels.resize(np.ascontiguousarray(array), int(width),
                                    int(height), interpolation=interpolation)
    if interpolation != INTER_LANCZOS:
        raise ValueError(f"unknown interpolation: {interpolation!r}")

    Image = _pil()
    size = (int(width), int(height))
    # Pillow cannot filter I;16 or multichannel float arrays. Filtering each
    # plane as float preserves the range and avoids dropping alpha channels.
    def plane_resize(plane):
        source = plane if plane.dtype == np.uint8 else plane.astype(np.float32)
        return np.array(Image.fromarray(source).resize(size, Image.Resampling.LANCZOS))

    if array.ndim == 2:
        out = plane_resize(array)
    elif array.ndim == 3:
        out = np.stack([plane_resize(array[..., c]) for c in range(array.shape[2])], axis=2)
    else:
        raise ValueError("resize expects an HxW or HxWxC image")
    if np.issubdtype(array.dtype, np.integer):
        out = np.rint(out).clip(0, np.iinfo(array.dtype).max)
    return out.astype(array.dtype, copy=False)
