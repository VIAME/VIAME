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
    "swap_channels", "resize", "flip", "apply_lut", "euclidean_distance",
    "convert_colour",
    "INTER_NEAREST", "INTER_LINEAR",
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
    """Reverse the first three planes, leaving a fourth alone.

    **The fourth plane is the point.** Reversing all of them turns RGBA into
    ABGR -- alpha where red should be -- which is not a channel order anything
    holds. Only the colour triple reverses; alpha, or a fourth band of any
    kind, stays where it is. The C++ `image_kernels.swap_channels` has always
    done this and the array version did not, which corrupted every four
    channel image that went through it.
    """
    array = np.asarray(array)

    if array.ndim != 3 or array.shape[2] < 3:
        return array

    out = array.copy()
    out[..., :3] = array[..., 2::-1]

    return out


#: The colour spaces `convert_colour` knows, and how each reaches RGB.
#:
#: Keyed by name because that is how a caller thinks about it, and because a
#: pair of names cannot be ambiguous the way a single code can: reversing the
#: channel order is its own inverse, so one number would have to mean both
#: directions.
#:
#: Each entry is the pair of kernels that leave and enter RGB. `None` means the
#: space *is* RGB.
_COLOUR_SPACES = {
    "rgb": (None, None),
    "gray": ("to_gray", "to_rgb"),
    "grey": ("to_gray", "to_rgb"),
    "hsv": ("to_hsv", "from_hsv"),
    "hsv_full": ("to_hsv", "from_hsv"),
    "hls": ("to_hls", "from_hls"),
    "lab": ("to_lab", "from_lab"),
    "luv": ("to_luv", "from_luv"),
    "xyz": ("to_xyz", "from_xyz"),
    "cie": ("to_xyz", "from_xyz"),
    "ycrcb": ("to_ycrcb", "from_ycrcb"),
    "ycr_cb": ("to_ycrcb", "from_ycrcb"),
    "yuv": ("to_ycrcb", "from_ycrcb"),
}

#: The two spaces that are a channel order rather than a colour space.
_SWAPPED = ("bgr",)


def convert_colour(array, source, destination):
    """One colour space to another, by name.

    `source` and `destination` are names -- `rgb`, `bgr`, `gray`, `hsv`,
    `hsv_full`, `hls`, `lab`, `luv`, `xyz`/`cie`, `ycrcb`, `yuv` -- and the
    route between any two goes through RGB, which is the one the kernels work
    in. A conversion between two non-RGB spaces therefore costs two passes,
    which is the cost of not privileging one pair over another.

    **`bgr` is a channel order, not a colour space**, and naming it here is
    what lets a caller holding BGR say so once instead of reversing the array
    at each call.

    `hsv_full` spreads the hue over 0..255 instead of 0..179, and `yuv` and
    `ycrcb` are the same two kernels with different chroma scalings; see
    `image_kernels.to_ycrcb`.
    """
    from viame import image_kernels

    source = str(source).lower()
    destination = str(destination).lower()

    for name in (source, destination):
        if name not in _COLOUR_SPACES and name not in _SWAPPED:
            raise ValueError("no colour space named {!r}".format(name))

    if source == destination:
        return np.asarray(array)

    array = np.asarray(array)

    # Into RGB.
    if source in _SWAPPED:
        rgb = swap_channels(array)
    else:
        leave = _COLOUR_SPACES[source][1]
        rgb = array if leave is None else _run(image_kernels, leave, source,
                                              array)

    # And out of it.
    if destination in _SWAPPED:
        return swap_channels(rgb)

    enter = _COLOUR_SPACES[destination][0]

    return rgb if enter is None else _run(image_kernels, enter, destination,
                                          rgb)


def _run(image_kernels, name, space, array):
    """One conversion kernel, with the flags the space name implies."""
    array = np.ascontiguousarray(array)
    kernel = getattr(image_kernels, name)

    if space in ("hsv_full",):
        return kernel(array, True)

    if space in ("yuv",):
        return kernel(array, True)

    return kernel(array)


def flip(array, horizontal=False, vertical=False):
    """A mirrored copy, about either axis or both.

    Two booleans rather than one signed integer, so that a call says which axis
    it means without a table to look it up in.
    """
    array = np.asarray(array)

    if vertical:
        array = array[::-1]
    if horizontal:
        array = array[:, ::-1]

    return np.ascontiguousarray(array)


def apply_lut(array, table):
    """Each byte looked up in a 256 entry table.

    The table may hold one entry per level, or one per level per plane, which
    is how a per-channel curve is applied in one pass.
    """
    array = np.asarray(array)

    if array.dtype != np.dtype(np.uint8):
        raise TypeError("apply_lut takes 8-bit input, got {}".format(
            array.dtype))

    table = np.asarray(table)

    if table.size % 256:
        raise ValueError("a lookup table has 256 entries, got {}".format(
            table.size))

    if table.size == 256:
        return table.reshape(256)[array]

    table = table.reshape(256, -1)
    planes = 1 if array.ndim == 2 else array.shape[2]

    if table.shape[1] != planes:
        raise ValueError("{} tables for {} planes".format(table.shape[1],
                                                         planes))

    out = np.empty(array.shape, dtype=table.dtype)

    for plane in range(planes):
        out[..., plane] = table[array[..., plane], plane]

    return out


def euclidean_distance(mask):
    """The true distance from each set pixel to the nearest zero, float32.

    Distinct from `image_kernels.distance_transform`, which is the **chamfer
    approximation** a three-by-three chamfer window gives, which is not
    Euclidean despite being the usual thing called an L2 distance transform --
    the two differ by up to 0.22 of a pixel. This one is exact.

    `scipy.ndimage` rather than a kernel of our own because the exact
    transform is a solved problem with a good implementation already in a
    declared dependency.
    """
    from scipy.ndimage import distance_transform_edt

    foreground = np.asarray(mask) != 0
    if foreground.all():
        # With no background pixel the distance is unbounded. Match the
        # finite sentinel used by the precise OpenCV transform; SciPy would
        # instead measure from an implicit pixel outside the image.
        return np.full(foreground.shape, np.sqrt(np.finfo(np.float32).max),
                       dtype=np.float32)
    return distance_transform_edt(foreground).astype(np.float32)


def resize(array, width, height, interpolation=INTER_LINEAR):
    """Resize with the native kernels, preserving uint8, uint16 or float32.

    Lanczos retains the Pillow implementation used by this helper; the
    native kernels implement nearest, bilinear, bicubic and area.
    """
    array = np.asarray(array)
    native = (np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.int16),
              np.dtype(np.float32), np.dtype(np.float64))

    if interpolation in (INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA):
        from viame import image_kernels

        if array.dtype in native:
            return image_kernels.resize(np.ascontiguousarray(array),
                                        int(width), int(height),
                                        interpolation=interpolation)

        # Everything else -- int32 labels, booleans -- nearest neighbour only,
        # which is the one filter that never mixes two samples and so never
        # needs to know how to average them. The mapping comes from the kernel
        # rather than being rewritten here: two planes of indices go through
        # the same nearest resize and come back as the indices to gather, so
        # there is one nearest-neighbour rule in the tree and not two.
        if interpolation != INTER_NEAREST:
            raise TypeError(
                "resize of {} is nearest neighbour only; uint8, uint16, "
                "int16, float32 and float64 take every "
                "filter".format(array.dtype))

        height_in, width_in = array.shape[:2]
        columns = np.broadcast_to(np.arange(width_in, dtype=np.float32),
                                  (height_in, width_in))
        rows = np.broadcast_to(
            np.arange(height_in, dtype=np.float32).reshape(-1, 1),
            (height_in, width_in))
        take_x = image_kernels.resize(np.ascontiguousarray(columns),
                                      int(width), int(height),
                                      interpolation=INTER_NEAREST)
        take_y = image_kernels.resize(np.ascontiguousarray(rows), int(width),
                                      int(height),
                                      interpolation=INTER_NEAREST)

        return array[take_y.astype(np.intp), take_x.astype(np.intp)]
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
