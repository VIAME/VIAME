# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""The slice of OpenCV's Python API that VIAME's forked packages use.

Every module VIAME itself ships is off cv2 -- `baseline:lazy_cv2` enforces
that over the whole runtime tree -- and `opencv-python-headless` is still
declared for one reason: the packages under `packages/pytorch-libs` import it.
`mmcv` reaches cv2 from six modules, `mmdetection` from five, `imgaug` from
fourteen and `mmdeploy` from one, and `import mmdet` fails without it.

This is what their patched files import instead:

    from viame.utilities import cv2_api as cv2

It is **not** a general OpenCV replacement and does not try to be. It is the
functions and flags those packages actually reach for -- measured by reading
every call site, not guessed -- over `viame.image_kernels`,
`viame.utilities.imageops`, `viame.utilities.geometry` and
`viame.video_io.frames`. A name OpenCV has and nothing here wants is absent, so
a new call site fails loudly at the attribute rather than quietly on the
result.

**Channel order is the trap, and it is handled here rather than at the call
sites.** VIAME's kernels are RGB throughout; OpenCV's API is BGR, and names its
conversions by the order the caller holds -- `COLOR_BGR2HSV` against
`COLOR_RGB2HSV`. So each conversion code carries the swap it needs and
`imread`, `imdecode`, `imencode` and `imwrite` keep cv2's BGR convention. A
patched file that was passing BGR keeps passing BGR and gets back what it got
before. That is the one boundary where the "normalise to RGB" rule gives way,
because mmcv's own contract with mmdet is BGR arrays.

**What is faithful and what is not.** The kernels behind `cvtColor`, `resize`,
the blurs, `filter2D`, `Canny`, `addWeighted`, `equalizeHist`, `CLAHE`,
`copyMakeBorder`, `dilate`, `medianBlur`, the warps and `remap` are the ones
verified bit for bit against cv2 elsewhere in this tree; where one is only
near-exact -- the float bilinear resize, finding 2.62 -- the difference is
recorded there. Three things are deliberately not faithful and say so at the
function: `putText` draws VIAME's own 5 by 7 font rather than a Hershey one,
`findContours` emits `RETR_CCOMP` borders in a different order than OpenCV 5
does (finding 2.74), and the windowing functions have nothing to draw on and
raise.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Flags
#
# The values are OpenCV's own. They have to be: mmcv stores them in
# configuration dictionaries, imgaug compares them, and a caller that has
# recorded `interpolation=1` in a config file means bilinear by that number.
# ---------------------------------------------------------------------------

INTER_NEAREST = 0
INTER_LINEAR = 1
INTER_CUBIC = 2
INTER_AREA = 3
INTER_LANCZOS4 = 4
INTER_LINEAR_EXACT = 5
INTER_MAX = 7
WARP_FILL_OUTLIERS = 8
WARP_INVERSE_MAP = 16

WARP_POLAR_LINEAR = 0
WARP_POLAR_LOG = 256

BORDER_CONSTANT = 0
BORDER_REPLICATE = 1
BORDER_REFLECT = 2
BORDER_WRAP = 3
BORDER_REFLECT_101 = 4
BORDER_REFLECT101 = 4
BORDER_DEFAULT = 4
BORDER_TRANSPARENT = 5
BORDER_ISOLATED = 16

COLOR_BGR2BGRA = 0
COLOR_BGRA2BGR = 1
COLOR_RGBA2RGB = 1
COLOR_BGRA2RGB = 3
COLOR_RGBA2BGR = 3
COLOR_BGR2RGB = 4
COLOR_RGB2BGR = 4
COLOR_BGRA2RGBA = 5
COLOR_RGBA2BGRA = 5
COLOR_BGR2GRAY = 6
COLOR_RGB2GRAY = 7
COLOR_GRAY2BGR = 8
COLOR_GRAY2RGB = 8
COLOR_GRAY2BGRA = 9
COLOR_BGR2XYZ = 32
COLOR_RGB2XYZ = 33
COLOR_XYZ2BGR = 34
COLOR_XYZ2RGB = 35
COLOR_BGR2YCR_CB = 36
COLOR_RGB2YCR_CB = 37
COLOR_YCR_CB2BGR = 38
COLOR_YCR_CB2RGB = 39
COLOR_BGR2HSV = 40
COLOR_RGB2HSV = 41
COLOR_BGR2LAB = 44
COLOR_RGB2LAB = 45
COLOR_BGR2Lab = 44
COLOR_RGB2Lab = 45
COLOR_BGR2LUV = 50
COLOR_RGB2LUV = 51
COLOR_BGR2HLS = 52
COLOR_RGB2HLS = 53
COLOR_HSV2BGR = 54
COLOR_HSV2RGB = 55
COLOR_LAB2BGR = 56
COLOR_LAB2RGB = 57
COLOR_Lab2BGR = 56
COLOR_Lab2RGB = 57
COLOR_LUV2BGR = 58
COLOR_LUV2RGB = 59
COLOR_HLS2BGR = 60
COLOR_HLS2RGB = 61
COLOR_BGR2HSV_FULL = 66
COLOR_RGB2HSV_FULL = 67
COLOR_BGR2HLS_FULL = 68
COLOR_RGB2HLS_FULL = 69
COLOR_HSV2BGR_FULL = 70
COLOR_HSV2RGB_FULL = 71
COLOR_HLS2BGR_FULL = 72
COLOR_HLS2RGB_FULL = 73
COLOR_BGR2YUV = 82
COLOR_RGB2YUV = 83
COLOR_YUV2BGR = 84
COLOR_YUV2RGB = 85

IMREAD_UNCHANGED = -1
IMREAD_GRAYSCALE = 0
IMREAD_COLOR = 1
IMREAD_ANYDEPTH = 2
IMREAD_ANYCOLOR = 4
IMREAD_IGNORE_ORIENTATION = 128

IMWRITE_JPEG_QUALITY = 1
IMWRITE_PNG_COMPRESSION = 16

RETR_EXTERNAL = 0
RETR_LIST = 1
RETR_CCOMP = 2
RETR_TREE = 3
CHAIN_APPROX_NONE = 1
CHAIN_APPROX_SIMPLE = 2

CC_STAT_LEFT = 0
CC_STAT_TOP = 1
CC_STAT_WIDTH = 2
CC_STAT_HEIGHT = 3
CC_STAT_AREA = 4

KMEANS_RANDOM_CENTERS = 0
KMEANS_USE_INITIAL_LABELS = 1
KMEANS_PP_CENTERS = 2

TERM_CRITERIA_COUNT = 1
TERM_CRITERIA_MAX_ITER = 1
TERM_CRITERIA_EPS = 2

MORPH_RECT = 0
MORPH_CROSS = 1
MORPH_ELLIPSE = 2
MORPH_ERODE = 0
MORPH_DILATE = 1
MORPH_OPEN = 2
MORPH_CLOSE = 3

NORM_INF = 1
NORM_L1 = 2
NORM_L2 = 4
NORM_MINMAX = 32

CV_8U = 0
CV_8S = 1
CV_16U = 2
CV_16S = 3
CV_32S = 4
CV_32F = 5
CV_64F = 6
CV_8UC1 = 0
CV_8UC3 = 64
CV_16SC2 = 35
CV_32FC1 = 5
CV_32FC2 = 37

FONT_HERSHEY_SIMPLEX = 0
FONT_HERSHEY_PLAIN = 1
FONT_HERSHEY_DUPLEX = 2
FONT_HERSHEY_COMPLEX = 3

LINE_4 = 4
LINE_8 = 8
LINE_AA = 16
FILLED = -1

WINDOW_NORMAL = 0
WINDOW_AUTOSIZE = 1
WND_PROP_VISIBLE = 4

CAP_PROP_POS_MSEC = 0
CAP_PROP_POS_FRAMES = 1
CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4
CAP_PROP_FPS = 5
CAP_PROP_FOURCC = 6
CAP_PROP_FRAME_COUNT = 7

#: What `cv2.__version__` would say. Deliberately not a version OpenCV ever
#: released: `mmcv.utils.env` records it in a diagnostic banner, and a banner
#: claiming an OpenCV that is not installed is the one thing worse than a
#: banner naming this module.
__version__ = "0+viame.kernels"


class error(Exception):
    """`cv2.error`.

    `imgaug` catches it in five augmenters to turn a bad argument into its own
    message, so it has to be the type raised by the functions those augmenters
    call, not a bare `ValueError`.
    """


class UMat:
    """`cv2.UMat`, as far as anything here needs it.

    Only `imgaug.augmenters.flip` mentions it, in an `isinstance` guard that
    decides whether an input has already been uploaded to a GPU buffer. There
    is no such buffer here, so nothing is ever an instance, and the guard
    takes its other branch. Constructing one is a caller asking for OpenCV's
    transparent API, which this is not.
    """

    def __init__(self, *_arguments, **_keywords):
        raise error("viame.utilities.cv2_api has no transparent API (UMat)")


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

def _kernels():
    """Imported on use, so that `import mmcv` does not load the extension."""
    from viame import image_kernels

    return image_kernels


_BORDERS = {BORDER_CONSTANT: "constant", BORDER_REPLICATE: "replicate",
            BORDER_REFLECT: "reflect", BORDER_WRAP: "wrap",
            BORDER_REFLECT_101: "reflect_101"}

_INTERPOLATIONS = {INTER_NEAREST: "nearest", INTER_LINEAR: "bilinear",
                   INTER_CUBIC: "bicubic", INTER_AREA: "area",
                   INTER_LINEAR_EXACT: "bilinear_exact"}

#: The three the kernels take. Everything else is converted, or refused.
_NATIVE = (np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32))


def _border(mode, what):
    """A border rule's name, with `BORDER_ISOLATED` stripped off.

    OpenCV's isolated flag says not to look outside a submatrix's own bounds.
    Nothing here hands a kernel a submatrix -- every array arrives whole --
    so the flag is already true and dropping it is not a change in behaviour.
    """
    mode = int(mode) & ~BORDER_ISOLATED

    if mode not in _BORDERS:
        raise error("{}: no border rule {}".format(what, mode))

    return _BORDERS[mode]


def _interpolation(flags, what):
    """The interpolation half of a `flags` word, warp bits removed."""
    mode = int(flags) & ~(WARP_INVERSE_MAP | WARP_FILL_OUTLIERS)

    if mode == INTER_LANCZOS4:
        raise error(
            "{}: Lanczos resampling is not implemented; "
            "viame.utilities.imageops.resize has a Pillow Lanczos for the "
            "resize case".format(what))

    if mode not in _INTERPOLATIONS:
        raise error("{}: no interpolation {}".format(what, mode))

    return _INTERPOLATIONS[mode]


def _native(array, what, planes=None):
    """An array a kernel will take: contiguous, and one of its three types.

    float64 is narrowed to float32, because every caller that reaches here
    with one is an augmentation holding an image, not a solver holding a
    matrix. Anything else is refused rather than guessed at.
    """
    array = np.asarray(array)

    if array.dtype == np.dtype(np.float64):
        array = array.astype(np.float32)
    elif array.dtype == np.dtype(bool):
        array = array.astype(np.uint8)
    elif array.dtype not in _NATIVE:
        raise error("{}: cannot take {} pixels".format(what, array.dtype))

    if planes is not None and _planes(array) not in planes:
        raise error("{}: wants {} planes, got {}".format(
            what, " or ".join(str(p) for p in planes), _planes(array)))

    return np.ascontiguousarray(array)


def _planes(array):
    return 1 if array.ndim == 2 else array.shape[2]


def _restore(result, like):
    """`result` in the dtype `like` arrived as, for the float64 narrowing."""
    if like.dtype == np.dtype(np.float64) and result.dtype != like.dtype:
        return result.astype(np.float64)

    return result


def _out(result, dst):
    """OpenCV's out parameter: write into `dst` when it fits, else replace it.

    Both halves matter. `mmcv.image.imnormalize_` is on the path of every
    mmdet inference and normalises **in place** --
    `cv2.subtract(img, mean, img)` -- so a shim that always returned a fresh
    array would leave the caller's image untouched. And OpenCV's `dst` is an
    `OutputArray`, which is a *hint*: `imgaug` hands `cvtColor` a
    three-channel buffer and asks for grey, and cv2 quietly reallocates and
    returns the new shape rather than failing. Writing unconditionally turned
    that into a broadcast error.
    """
    if dst is None:
        return result

    target = np.asarray(dst)
    fresh = np.asarray(result)

    if target.shape != fresh.shape or target.dtype != fresh.dtype:
        return result

    dst[...] = result
    return dst


def _saturate(values, dtype):
    """`cv::saturate_cast` over an array: round half to even, then clamp.

    Rounding matters and is not numpy's default anywhere: `saturate_cast` to
    an integer goes through `cvRound`, which is half to **even**, and
    `astype` truncates.
    """
    dtype = np.dtype(dtype)

    if dtype.kind == "f":
        return np.asarray(values, dtype=dtype)

    limits = np.iinfo(dtype)
    return np.clip(np.rint(values), limits.min, limits.max).astype(dtype)


def _is_scalar_operand(value, planes):
    """Whether OpenCV would read `value` as a per-channel scalar.

    Its rule, from `checkScalar`: a 1 by 1, 1 by 4, 4 by 1 or 1 by `cn`
    double array is a scalar, not an image. `mmcv.image.imnormalize_` relies
    on it -- it passes a `(1, 3)` mean against an `(h, w, 3)` image.
    """
    if np.isscalar(value):
        return True

    array = np.asarray(value)

    if array.ndim == 0:
        return True

    return array.size in (1, 4, planes)


def _operand(value, planes, shape):
    """A second operand broadcast the way OpenCV would broadcast it."""
    if _is_scalar_operand(value, planes):
        flat = np.asarray(value, dtype=np.float64).reshape(-1)

        if flat.size == 1:
            return flat[0]

        if len(shape) == 2:
            return flat[0]

        return flat[:planes].reshape((1,) * (len(shape) - 1) + (planes,))

    return np.asarray(value)


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------

def _swap(array):
    return np.ascontiguousarray(np.asarray(array)[..., ::-1])


def _to_gray(array):
    return _kernels().to_gray(_native(array, "cvtColor", planes=(3,)))


def _hsv(array, full, inverse, swap_in, swap_out):
    kernels = _kernels()
    source = _native(array, "cvtColor", planes=(3,))

    if swap_in:
        source = _swap(source)

    out = (kernels.from_hsv(source, full=full) if inverse
           else kernels.to_hsv(source, full=full))

    return _swap(out) if swap_out else out


def _hls(array, inverse, swap_in, swap_out):
    kernels = _kernels()
    source = _native(array, "cvtColor", planes=(3,))

    if swap_in:
        source = _swap(source)

    out = kernels.from_hls(source) if inverse else kernels.to_hls(source)

    return _swap(out) if swap_out else out


def _lab(array, inverse, swap_in, swap_out):
    kernels = _kernels()
    source = _native(array, "cvtColor", planes=(3,))

    if swap_in:
        source = _swap(source)

    out = kernels.from_lab(source) if inverse else kernels.to_lab(source)

    return _swap(out) if swap_out else out


def _xyz(array, inverse, swap_in, swap_out):
    kernels = _kernels()
    source = _native(array, "cvtColor", planes=(3,))

    if swap_in:
        source = _swap(source)

    out = kernels.from_xyz(source) if inverse else kernels.to_xyz(source)

    return _swap(out) if swap_out else out


def _luma_chroma(array, yuv, inverse, swap_in, swap_out):
    kernels = _kernels()
    source = _native(array, "cvtColor", planes=(3,))

    if swap_in:
        source = _swap(source)

    out = (kernels.from_ycrcb(source, yuv) if inverse
           else kernels.to_ycrcb(source, yuv))

    return _swap(out) if swap_out else out


def _luv(array, inverse, swap_in, swap_out):
    kernels = _kernels()
    source = _native(array, "cvtColor", planes=(3,))

    if swap_in:
        source = _swap(source)

    out = kernels.from_luv(source) if inverse else kernels.to_luv(source)

    return _swap(out) if swap_out else out


def _unimplemented_colour(name):
    def convert(_array):
        raise error(
            "cvtColor: {} is not implemented".format(name))

    return convert


#: Code to conversion. The swaps are the whole point of the table: a kernel
#: takes RGB, and half of OpenCV's codes name BGR.
_CONVERSIONS = {
    COLOR_BGR2RGB: _swap,                        # and RGB2BGR, the same code
    COLOR_BGRA2BGR: lambda a: np.ascontiguousarray(np.asarray(a)[..., :3]),
    COLOR_BGRA2RGB: lambda a: _swap(np.asarray(a)[..., :3]),
    COLOR_BGR2GRAY: lambda a: _to_gray(_swap(a)),
    COLOR_RGB2GRAY: _to_gray,
    COLOR_GRAY2BGR: lambda a: _kernels().to_rgb(_native(a, "cvtColor")),
    COLOR_BGR2HSV: lambda a: _hsv(a, False, False, True, False),
    COLOR_RGB2HSV: lambda a: _hsv(a, False, False, False, False),
    COLOR_HSV2BGR: lambda a: _hsv(a, False, True, False, True),
    COLOR_HSV2RGB: lambda a: _hsv(a, False, True, False, False),
    COLOR_BGR2HSV_FULL: lambda a: _hsv(a, True, False, True, False),
    COLOR_RGB2HSV_FULL: lambda a: _hsv(a, True, False, False, False),
    COLOR_HSV2BGR_FULL: lambda a: _hsv(a, True, True, False, True),
    COLOR_HSV2RGB_FULL: lambda a: _hsv(a, True, True, False, False),
    COLOR_BGR2HLS: lambda a: _hls(a, False, True, False),
    COLOR_RGB2HLS: lambda a: _hls(a, False, False, False),
    COLOR_HLS2BGR: lambda a: _hls(a, True, False, True),
    COLOR_HLS2RGB: lambda a: _hls(a, True, False, False),
    COLOR_BGR2LAB: lambda a: _lab(a, False, True, False),
    COLOR_RGB2LAB: lambda a: _lab(a, False, False, False),
    COLOR_LAB2BGR: lambda a: _lab(a, True, False, True),
    COLOR_LAB2RGB: lambda a: _lab(a, True, False, False),
    COLOR_BGR2XYZ: lambda a: _xyz(a, False, True, False),
    COLOR_RGB2XYZ: lambda a: _xyz(a, False, False, False),
    COLOR_XYZ2BGR: lambda a: _xyz(a, True, False, True),
    COLOR_XYZ2RGB: lambda a: _xyz(a, True, False, False),
    COLOR_BGR2LUV: lambda a: _luv(a, False, True, False),
    COLOR_RGB2LUV: lambda a: _luv(a, False, False, False),
    COLOR_LUV2BGR: lambda a: _luv(a, True, False, True),
    COLOR_LUV2RGB: lambda a: _luv(a, True, False, False),
    COLOR_BGR2YUV: lambda a: _luma_chroma(a, True, False, True, False),
    COLOR_RGB2YUV: lambda a: _luma_chroma(a, True, False, False, False),
    COLOR_YUV2BGR: lambda a: _luma_chroma(a, True, True, False, True),
    COLOR_YUV2RGB: lambda a: _luma_chroma(a, True, True, False, False),
    COLOR_BGR2YCR_CB: lambda a: _luma_chroma(a, False, False, True, False),
    COLOR_RGB2YCR_CB: lambda a: _luma_chroma(a, False, False, False, False),
    COLOR_YCR_CB2BGR: lambda a: _luma_chroma(a, False, True, False, True),
    COLOR_YCR_CB2RGB: lambda a: _luma_chroma(a, False, True, False, False),
    COLOR_BGR2HLS_FULL: _unimplemented_colour("COLOR_BGR2HLS_FULL"),
    COLOR_RGB2HLS_FULL: _unimplemented_colour("COLOR_RGB2HLS_FULL"),
    COLOR_HLS2BGR_FULL: _unimplemented_colour("COLOR_HLS2BGR_FULL"),
    COLOR_HLS2RGB_FULL: _unimplemented_colour("COLOR_HLS2RGB_FULL"),
}


def cvtColor(src, code, dst=None, dstCn=0):
    """`cv2.cvtColor`.

    The `dst` positional is honoured, because mmcv converts in place:
    `cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)` is how `imnormalize_` gets to
    RGB without a copy.
    """
    if int(code) not in _CONVERSIONS:
        raise error("cvtColor: no conversion {}".format(code))

    source = np.asarray(src)
    result = _CONVERSIONS[int(code)](source)

    return _out(_restore(result, source), dst)


# ---------------------------------------------------------------------------
# Per-pixel arithmetic
# ---------------------------------------------------------------------------

def _arithmetic(name, operation, src1, src2, dst, dtype, widen):
    """One of OpenCV's per-pixel operations, in the working type it uses.

    `widen` is not a style choice, it is a measurement. On a float32 image
    with a float64 per-channel scalar -- which is exactly what
    `mmcv.image.imnormalize_` passes -- `cv2.add` and `cv2.subtract` agree to
    the last bit with the scalar **narrowed to float32** and the arithmetic
    done there, and disagree by an ULP with a float64 intermediate;
    `cv2.multiply` is the other way round. Both were checked over three
    random images each, and the pair of answers is stable.
    """
    first = np.asarray(src1)
    planes = _planes(first)
    second = _operand(src2, planes, first.shape)

    target = first.dtype if dtype in (None, -1) else _depth_dtype(dtype, name)

    working = np.dtype(np.float64)

    if not widen and target.kind == "f" and target.itemsize <= 4:
        working = target

    values = operation(first.astype(working),
                       np.asarray(second, dtype=working))

    return _out(_saturate(values, target), dst)


def _depth_dtype(depth, what):
    table = {CV_8U: np.uint8, CV_16U: np.uint16, CV_16S: np.int16,
             CV_32S: np.int32, CV_32F: np.float32, CV_64F: np.float64}

    if int(depth) not in table:
        raise error("{}: no depth {}".format(what, depth))

    return np.dtype(table[int(depth)])


def add(src1, src2, dst=None, mask=None, dtype=-1):
    """`cv2.add`, saturating."""
    _refuse_mask(mask, "add")
    return _arithmetic("add", np.add, src1, src2, dst, dtype, widen=False)


def subtract(src1, src2, dst=None, mask=None, dtype=-1):
    """`cv2.subtract`, saturating."""
    _refuse_mask(mask, "subtract")
    return _arithmetic("subtract", np.subtract, src1, src2, dst, dtype,
                       widen=False)


def multiply(src1, src2, dst=None, scale=1.0, dtype=-1):
    """`cv2.multiply`, saturating, with OpenCV's extra `scale`."""
    return _arithmetic("multiply",
                       lambda a, b: a * b * np.float64(scale),
                       src1, src2, dst, dtype, widen=True)


def divide(src1, src2, dst=None, scale=1.0, dtype=-1):
    """`cv2.divide`, saturating. A zero divisor gives zero, as OpenCV's does."""
    def operation(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(b == 0.0, 0.0, np.float64(scale) * a / b)
        return out

    return _arithmetic("divide", operation, src1, src2, dst, dtype, widen=True)


def _refuse_mask(mask, what):
    if mask is not None:
        raise error("{}: the mask argument is not implemented".format(what))


def addWeighted(src1, alpha, src2, beta, gamma, dst=None, dtype=-1):
    """`cv2.addWeighted`: `src1 * alpha + src2 * beta + gamma`, saturated.

    Goes through the kernel, which reproduces OpenCV's fused multiply-add and
    its half-to-even rounding (finding 2.57), whenever both images share one
    of the three native types. A mixed or wider pair is done in float64,
    which is what OpenCV's own widening does.

    Exact on 8- and 16-bit images. On float32 it is within an ULP -- about
    3e-05 at a pixel value of 255 -- which is where the kernel stands and not
    something this wrapper adds.
    """
    first = np.asarray(src1)
    second = np.asarray(src2)

    if (first.dtype == second.dtype and first.dtype in _NATIVE and
            first.shape == second.shape and dtype in (None, -1)):
        result = _kernels().add_weighted(
            np.ascontiguousarray(first), float(alpha),
            np.ascontiguousarray(second), float(beta), float(gamma))
        return _out(result, dst)

    target = first.dtype if dtype in (None, -1) else _depth_dtype(
        dtype, "addWeighted")
    values = (first.astype(np.float64) * np.float64(alpha) +
              second.astype(np.float64) * np.float64(beta) +
              np.float64(gamma))

    return _out(_saturate(values, target), dst)


def LUT(src, lut, dst=None):
    """`cv2.LUT`: look each byte up in a 256 entry table.

    The table may be one entry per channel, which is how
    `mmcv.image.adjust_color` gets a different curve per plane.
    """
    source = np.asarray(src)

    if source.dtype != np.dtype(np.uint8):
        raise error("LUT: takes 8-bit input, got {}".format(source.dtype))

    table = np.asarray(lut)

    if table.size % 256:
        raise error("LUT: the table has {} entries, not 256".format(table.size))

    if table.size == 256:
        return _out(table.reshape(256)[source], dst)

    table = table.reshape(256, -1)
    planes = _planes(source)

    if table.shape[1] != planes:
        raise error("LUT: {} tables for {} planes".format(
            table.shape[1], planes))

    result = np.empty(source.shape, dtype=table.dtype)

    for plane in range(planes):
        result[..., plane] = table[source[..., plane], plane]

    return _out(result, dst)


def split(m, mv=None):
    """`cv2.split`: a list of single channel planes."""
    array = np.asarray(m)

    if array.ndim == 2:
        return [array]

    return [np.ascontiguousarray(array[..., plane])
            for plane in range(array.shape[2])]


def merge(mv, dst=None):
    """`cv2.merge`: single channel planes back into one array."""
    planes = [np.asarray(plane) for plane in mv]
    planes = [plane[..., 0] if plane.ndim == 3 else plane for plane in planes]

    return _out(np.ascontiguousarray(np.stack(planes, axis=-1)), dst)


def normalize(src, dst=None, alpha=1.0, beta=0.0, norm_type=NORM_L2,
              dtype=-1, mask=None):
    """`cv2.normalize`, for `NORM_MINMAX` alone.

    `alpha` and `beta` are the two ends of the target range in either order,
    which is OpenCV's convention for this norm and nobody else's.
    """
    _refuse_mask(mask, "normalize")

    if int(norm_type) != NORM_MINMAX:
        raise error(
            "normalize: only NORM_MINMAX is implemented, not {}".format(
                norm_type))

    source = np.asarray(src)
    low, high = sorted((float(alpha), float(beta)))
    result = _kernels().normalize(_native(source, "normalize"), low, high)

    return _out(_restore(result, source), dst)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def resize(src, dsize, dst=None, fx=0.0, fy=0.0, interpolation=INTER_LINEAR):
    """`cv2.resize`. `dsize` is `(width, height)`, or None with `fx`/`fy`."""
    source = np.asarray(src)
    height, width = source.shape[:2]

    if dsize in (None, 0) or tuple(dsize) == (0, 0):
        if not fx or not fy:
            raise error("resize: neither a size nor both scale factors")
        out_width = int(round(width * float(fx)))
        out_height = int(round(height * float(fy)))

        # OpenCV maps coordinates by the **factor** when it is given one, and
        # by the output size when it is given a size. The kernels only take a
        # size, so the two agree only when the factor lands on an integer
        # number of pixels. Nothing in mmcv, mmdet or imgaug reaches here --
        # they all compute the size themselves -- and a caller that does gets
        # told rather than quietly given a slightly different image.
        if (abs(width * float(fx) - out_width) > 1e-9 or
                abs(height * float(fy) - out_height) > 1e-9):
            raise error(
                "resize: a scale factor that does not land on whole pixels "
                "({} by {} at {}, {}) cannot be reproduced; pass the size "
                "instead".format(width, height, fx, fy))
    else:
        out_width, out_height = int(dsize[0]), int(dsize[1])

    if out_width <= 0 or out_height <= 0:
        raise error("resize: empty target size {}".format(
            (out_width, out_height)))

    how = _interpolation(interpolation, "resize")

    # int32 and the like: OpenCV supports them for **nearest neighbour only**,
    # and so does this. `imgaug` upscales a coarse int32 dropout mask that
    # way. The mapping is taken from the kernel rather than rewritten -- two
    # index planes go through the same nearest resize and come back as the
    # indices to gather -- so there is one nearest-neighbour rule in the tree
    # and not two that can drift.
    if source.dtype not in _NATIVE and source.dtype != np.dtype(np.float64):
        if how != "nearest":
            raise error(
                "resize: {} pixels are supported for nearest neighbour only, "
                "as cv2 supports them".format(source.dtype))

        kernels = _kernels()
        columns = np.broadcast_to(
            np.arange(width, dtype=np.float32), (height, width))
        rows = np.broadcast_to(
            np.arange(height, dtype=np.float32).reshape(-1, 1),
            (height, width))
        take_x = kernels.resize(np.ascontiguousarray(columns), out_width,
                                out_height, interpolation="nearest")
        take_y = kernels.resize(np.ascontiguousarray(rows), out_width,
                                out_height, interpolation="nearest")

        return _out(source[take_y.astype(np.intp), take_x.astype(np.intp)],
                    dst)

    result = _kernels().resize(
        _native(source, "resize"), out_width, out_height, interpolation=how)

    return _out(_restore(result, source), dst)


def flip(src, flipCode, dst=None):
    """`cv2.flip`: 0 about the x axis, positive about the y, negative both."""
    array = np.asarray(src)
    code = int(flipCode)

    if code == 0:
        result = array[::-1]
    elif code > 0:
        result = array[:, ::-1]
    else:
        result = array[::-1, ::-1]

    return _out(np.ascontiguousarray(result), dst)


def copyMakeBorder(src, top, bottom, left, right, borderType, dst=None,
                   value=None):
    """`cv2.copyMakeBorder`."""
    source = np.asarray(src)
    result = _kernels().make_border(
        _native(source, "copyMakeBorder"), int(top), int(bottom), int(left),
        int(right), _scalar_constant(value, _planes(source)),
        _border(borderType, "copyMakeBorder"))

    return _out(_restore(result, source), dst)


def getRotationMatrix2D(center, angle, scale):
    """`cv2.getRotationMatrix2D`: the 2 by 3 affine, as float64."""
    from viame.utilities import geometry

    return geometry.rotation_matrix_2d(
        (float(center[0]), float(center[1])), float(angle), float(scale))


def invertAffineTransform(M, iM=None):
    """`cv2.invertAffineTransform`."""
    from viame.utilities import geometry

    return _out(geometry.invert_affine(np.asarray(M, dtype=np.float64)), iM)


def getPerspectiveTransform(src, dst, solveMethod=None):
    """`cv2.getPerspectiveTransform`: the homography through four points."""
    from viame.utilities import geometry

    return geometry.four_point_homography(
        np.asarray(src, dtype=np.float64).reshape(-1, 2),
        np.asarray(dst, dtype=np.float64).reshape(-1, 2))


def perspectiveTransform(src, m, dst=None):
    """`cv2.perspectiveTransform`: points through a homography.

    OpenCV takes and returns an `(n, 1, 2)` array; the shape it was handed
    comes back, because callers index it.
    """
    from viame.utilities import geometry

    points = np.asarray(src, dtype=np.float64)
    moved = geometry.apply_homography(
        np.asarray(m, dtype=np.float64), points.reshape(-1, 2))

    return _out(moved.reshape(points.shape).astype(points.dtype, copy=False),
                dst)


def _warp(kernel, src, M, dsize, dst, flags, borderMode, borderValue, what):
    source = np.asarray(src)
    matrix = np.asarray(M, dtype=np.float64)

    # OpenCV's flag says the matrix already maps destination to source, which
    # is the direction the kernels sample in anyway. Passing it straight
    # through is not the same as inverting it here and letting the kernel
    # invert it back: the round trip costs a grey level here and there.
    inverse = bool(int(flags) & WARP_INVERSE_MAP)

    if dsize in (None, 0) or tuple(dsize) == (0, 0):
        out_width, out_height = 0, 0
    else:
        out_width, out_height = int(dsize[0]), int(dsize[1])

    result = getattr(_kernels(), kernel)(
        _native(source, what), matrix, out_width, out_height,
        interpolation=_interpolation(flags, what),
        border=_border(borderMode, what),
        constant=_scalar_constant(borderValue, _planes(source)),
        inverse=inverse)

    return _out(_restore(result, source), dst)


def _scalar_constant(value, planes):
    """OpenCV's `Scalar` border value, per plane.

    The trap: a bare number is **not** that number in every channel.
    `cv::Scalar(7)` is `(7, 0, 0, 0)`, so `copyMakeBorder(..., value=7)` on a
    three channel image gives a border of `(7, 0, 0)` -- verified against cv2,
    because it is not what anyone writing `value=7` expects. A caller that
    wants grey passes a triple, and OpenCV's arithmetic functions read a bare
    number the other way, which `_operand` handles separately.
    """
    if value is None:
        return 0.0

    if np.isscalar(value):
        return float(value) if planes == 1 else (
            [float(value)] + [0.0] * (planes - 1))

    flat = [float(component)
            for component in np.asarray(value, dtype=np.float64).reshape(-1)]

    if not flat:
        return 0.0

    if planes == 1:
        return flat[0]

    return (flat + [0.0] * planes)[:planes]


def warpAffine(src, M, dsize, dst=None, flags=INTER_LINEAR,
               borderMode=BORDER_CONSTANT, borderValue=None):
    """`cv2.warpAffine`."""
    return _warp("warp_affine", src, M, dsize, dst, flags, borderMode,
                 borderValue, "warpAffine")


def warpPerspective(src, M, dsize, dst=None, flags=INTER_LINEAR,
                    borderMode=BORDER_CONSTANT, borderValue=None):
    """`cv2.warpPerspective`."""
    return _warp("warp_perspective", src, M, dsize, dst, flags, borderMode,
                 borderValue, "warpPerspective")


def remap(src, map1, map2, interpolation=INTER_LINEAR, dst=None,
          borderMode=BORDER_CONSTANT, borderValue=None):
    """`cv2.remap`, with both maps given as float planes."""
    source = np.asarray(src)
    result = _kernels().remap(
        _native(source, "remap"),
        np.ascontiguousarray(map1, dtype=np.float32),
        np.ascontiguousarray(map2, dtype=np.float32),
        interpolation=_interpolation(interpolation, "remap"),
        border=_border(borderMode, "remap"),
        constant=_scalar_constant(borderValue, _planes(source)))

    return _out(_restore(result, source), dst)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter2D(src, ddepth, kernel, dst=None, anchor=None, delta=0,
             borderType=BORDER_DEFAULT):
    """`cv2.filter2D`: correlation with a 2-D kernel, not convolution."""
    if anchor not in (None, (-1, -1), [-1, -1]):
        raise error("filter2D: only the centred anchor is implemented")

    if float(delta) != 0.0:
        raise error("filter2D: a non-zero delta is not implemented")

    source = np.asarray(src)

    if ddepth not in (None, -1) and _depth_dtype(
            ddepth, "filter2D") != source.dtype:
        raise error(
            "filter2D: only ddepth -1 (the input's own depth) is implemented")

    result = _kernels().filter_2d(
        _native(source, "filter2D"),
        np.ascontiguousarray(kernel, dtype=np.float64),
        border=_border(borderType, "filter2D"))

    return _out(_restore(result, source), dst)


def _square(size, what):
    """One odd side from OpenCV's `(width, height)`, which the kernels take."""
    if np.isscalar(size):
        return int(size)

    width, height = int(size[0]), int(size[1])

    if width != height:
        raise error(
            "{}: a {} by {} kernel is not implemented; the kernels take one "
            "odd size for both sides".format(what, width, height))

    return width


def blur(src, ksize, dst=None, anchor=None, borderType=BORDER_DEFAULT):
    """`cv2.blur`: the box filter."""
    source = np.asarray(src)
    result = _kernels().box_blur(
        _native(source, "blur"), _square(ksize, "blur"),
        border=_border(borderType, "blur"))

    return _out(_restore(result, source), dst)


def GaussianBlur(src, ksize, sigmaX, dst=None, sigmaY=None,
                 borderType=BORDER_DEFAULT):
    """`cv2.GaussianBlur`.

    A zero `ksize` derives the size from sigma the way OpenCV does -- three
    sigma either side for 8-bit, four for anything wider, rounded and forced
    odd.
    """
    source = np.asarray(src)
    sigma = float(sigmaX)

    if sigmaY not in (None, 0, 0.0) and float(sigmaY) != sigma:
        raise error(
            "GaussianBlur: a different sigma per axis is not implemented")

    size = _square(ksize, "GaussianBlur")

    if size == 0:
        if sigma <= 0.0:
            raise error("GaussianBlur: neither a size nor a sigma")
        spread = 3 if source.dtype == np.dtype(np.uint8) else 4
        size = int(round(sigma * spread * 2 + 1)) | 1

    result = _kernels().gaussian_blur(
        _native(source, "GaussianBlur"), size, sigma,
        border=_border(borderType, "GaussianBlur"))

    return _out(_restore(result, source), dst)


def bilateralFilter(src, d, sigmaColor, sigmaSpace, dst=None,
                    borderType=BORDER_DEFAULT):
    """`cv2.bilateralFilter`: a blur that does not cross an edge.

    Exact for three planes. A single plane is within a count on about half its
    pixels, where OpenCV's one-channel body adds the same twenty-odd floats in
    a different order.
    """
    source = np.asarray(src)
    result = _kernels().bilateral_blur(
        _native(source, "bilateralFilter"), int(d), float(sigmaColor),
        float(sigmaSpace), _border(borderType, "bilateralFilter"))

    return _out(_restore(result, source), dst)


def pyrDown(src, dst=None, dstsize=None, borderType=BORDER_DEFAULT):
    """`cv2.pyrDown`. Exact, including at odd sizes."""
    source = np.asarray(src)
    result = _kernels().pyramid_down(_native(source, "pyrDown"))

    if dstsize is not None and tuple(dstsize) not in (
            (0, 0), (result.shape[1], result.shape[0])):
        raise error(
            "pyrDown: only the derived size {} is implemented, not "
            "{}".format((result.shape[1], result.shape[0]), tuple(dstsize)))

    return _out(_restore(result, source), dst)


def pyrUp(src, dst=None, dstsize=None, borderType=BORDER_DEFAULT):
    """`cv2.pyrUp`. Exact, including where the target is one short of double."""
    source = np.asarray(src)
    width, height = (0, 0) if dstsize is None else (int(dstsize[0]),
                                                   int(dstsize[1]))
    result = _kernels().pyramid_up(_native(source, "pyrUp"), width, height)

    return _out(_restore(result, source), dst)


def pyrMeanShiftFiltering(src, sp, sr, dst=None, maxLevel=0,
                          termcrit=None):
    """`cv2.pyrMeanShiftFiltering`, **one level only**.

    Exact at `maxLevel=0`, over a real 270 by 480 frame as well as random
    blocks. A higher level is refused rather than approximated: OpenCV
    combines the pyramid's levels through a mask whose rule this does not
    reproduce, and `design/lite-findings.md` 2.75 records what was
    established -- `pyrDown`, `pyrUp` and the mean-shift core are each exact,
    so what is missing is the combination and not the arithmetic.

    Note that `maxLevel` **defaults to 0 here and to 1 in cv2**. A caller that
    wants OpenCV's default is asking for the level combination and gets told.
    """
    if int(maxLevel) != 0:
        raise error(
            "pyrMeanShiftFiltering: only maxLevel=0 is implemented. The "
            "pyramid combination is not reproduced -- finding 2.75 -- and "
            "guessing at it would be a different filter wearing the same "
            "name")

    iterations, epsilon = 5, 1.0

    if termcrit is not None:
        kind, count, precision = termcrit
        if int(kind) & TERM_CRITERIA_MAX_ITER:
            iterations = int(count)
        if int(kind) & TERM_CRITERIA_EPS:
            epsilon = float(precision)

    source = np.asarray(src)
    result = _kernels().mean_shift_blur(
        _native(source, "pyrMeanShiftFiltering", planes=(3,)), float(sp),
        float(sr), iterations, epsilon)

    return _out(_restore(result, source), dst)


def medianBlur(src, ksize, dst=None):
    """`cv2.medianBlur`."""
    source = np.asarray(src)
    result = _kernels().median_blur(_native(source, "medianBlur"), int(ksize))

    return _out(_restore(result, source), dst)


def Canny(image, threshold1, threshold2, edges=None, apertureSize=3,
          L2gradient=False):
    """`cv2.Canny`: a single plane edge map, 0 or 255.

    A multi-plane input is handled OpenCV's way, which is worth knowing: each
    pixel independently takes the plane whose gradient is strongest. A colour
    edge map is therefore neither the union of three grey ones nor the edge
    map of the luma.
    """
    return _out(_kernels().canny(
        _native(image, "Canny", planes=(1, 2, 3, 4)), float(threshold1),
        float(threshold2), int(apertureSize), bool(L2gradient)), edges)


def getStructuringElement(shape, ksize, anchor=None):
    """`cv2.getStructuringElement`.

    The kernels take a shape name and a size rather than an element array, so
    what comes back is the request itself -- `dilate` and `erode` below
    recognise it. An element built by hand elsewhere would not be honoured,
    and saying so is better than accepting one and ignoring it.
    """
    names = {MORPH_RECT: "rect", MORPH_CROSS: "cross", MORPH_ELLIPSE: "disk"}

    if int(shape) not in names:
        raise error("getStructuringElement: no shape {}".format(shape))

    return _Element(names[int(shape)], int(ksize[0]), int(ksize[1]))


class _Element:
    """A structuring element as the kernels describe one: a name and a size."""

    __slots__ = ("shape", "width", "height")

    def __init__(self, shape, width, height):
        self.shape = shape
        self.width = width
        self.height = height


def _element(kernel, what):
    if kernel is None:
        return _Element("rect", 3, 3)

    if isinstance(kernel, _Element):
        return kernel

    array = np.asarray(kernel)

    if array.ndim != 2:
        raise error("{}: a structuring element is two dimensional".format(what))

    height, width = array.shape

    if np.all(array != 0):
        return _Element("rect", width, height)

    raise error(
        "{}: only a rectangular element, or one from "
        "getStructuringElement, is implemented".format(what))


def dilate(src, kernel, dst=None, anchor=None, iterations=1,
           borderType=BORDER_CONSTANT, borderValue=None):
    """`cv2.dilate`."""
    return _morphology("dilate", src, kernel, dst, iterations)


def erode(src, kernel, dst=None, anchor=None, iterations=1,
          borderType=BORDER_CONSTANT, borderValue=None):
    """`cv2.erode`."""
    return _morphology("erode", src, kernel, dst, iterations)


def _morphology(name, src, kernel, dst, iterations):
    element = _element(kernel, name)
    source = np.asarray(src)
    result = _native(source, name)

    for _pass in range(max(int(iterations), 1)):
        result = getattr(_kernels(), name)(
            result, element.shape, element.width, element.height)

    return _out(_restore(result, source), dst)


def morphologyEx(src, op, kernel, dst=None, anchor=None, iterations=1,
                 borderType=BORDER_CONSTANT, borderValue=None):
    """`cv2.morphologyEx`, for opening and closing."""
    names = {MORPH_OPEN: "open", MORPH_CLOSE: "close"}

    if int(op) in (MORPH_ERODE, MORPH_DILATE):
        return _morphology("erode" if int(op) == MORPH_ERODE else "dilate",
                           src, kernel, dst, iterations)

    if int(op) not in names:
        raise error(
            "morphologyEx: only erode, dilate, open and close are "
            "implemented, not {}".format(op))

    element = _element(kernel, "morphologyEx")
    source = np.asarray(src)
    result = _kernels().morphology(
        _native(source, "morphologyEx"), names[int(op)], element.shape,
        element.width, element.height, int(iterations))

    return _out(_restore(result, source), dst)


def equalizeHist(src, dst=None):
    """`cv2.equalizeHist`."""
    return _out(_kernels().equalize(
        _native(src, "equalizeHist", planes=(1,))), dst)


def createCLAHE(clipLimit=40.0, tileGridSize=(8, 8)):
    """`cv2.createCLAHE`: an object whose `apply` runs the kernel."""
    return _CLAHE(float(clipLimit), tileGridSize)


class _CLAHE:
    """What `cv2.createCLAHE` returns, with the `apply` its callers use."""

    def __init__(self, clip_limit, tiles):
        self._clip_limit = clip_limit
        self._tiles = (int(tiles[0]), int(tiles[1]))

    def apply(self, src, dst=None):
        return _out(_kernels().clahe(
            _native(src, "CLAHE.apply", planes=(1,)), self._clip_limit,
            self._tiles[0], self._tiles[1]), dst)

    def setClipLimit(self, clip_limit):
        self._clip_limit = float(clip_limit)

    def getClipLimit(self):
        return self._clip_limit

    def setTilesGridSize(self, tiles):
        self._tiles = (int(tiles[0]), int(tiles[1]))

    def getTilesGridSize(self):
        return self._tiles

    def collectGarbage(self):
        """Nothing is cached, so there is nothing to collect."""


def calcHist(images, channels, mask, histSize, ranges, hist=None,
             accumulate=False):
    """`cv2.calcHist`, for the one-channel 256-bin case its callers use.

    `imgaug.augmenters.pillike`'s `Equalize` and `Autocontrast` are the
    callers, and both ask the same question -- how many of each byte value
    are in this plane -- which is a `bincount`. A general N-dimensional
    histogram is not implemented, and says so rather than answering a
    different question.
    """
    if len(channels) != 1 or len(histSize) != 1:
        raise error("calcHist: only a one dimensional histogram is implemented")

    if int(histSize[0]) != 256 or tuple(ranges) != (0, 256):
        raise error(
            "calcHist: only 256 bins over [0, 256) is implemented, not {} "
            "bins over {}".format(histSize[0], tuple(ranges)))

    array = np.asarray(images[0])

    if array.dtype != np.dtype(np.uint8):
        raise error("calcHist: takes 8-bit input, got {}".format(array.dtype))

    plane = array if array.ndim == 2 else array[..., int(channels[0])]

    if mask is not None:
        plane = plane[np.asarray(mask).astype(bool)]

    counts = np.bincount(np.asarray(plane).reshape(-1), minlength=256)

    # A flat 256, which is what cv2 5.0 returns. OpenCV 4 gave a 256 by 1
    # column; every caller here indexes or sums it, so either works, and this
    # is the one the comparison was made against.
    return counts.astype(np.float32)


#: `cv2.Laplacian`'s kernels, read out of cv2 by filtering a delta rather
#: than derived: aperture 1 is the five-point stencil, and 3 and 5 are the
#: ones `getDerivKernels` builds from the separable Sobel pair, which put the
#: weight on the **corners** at 3 and not on the edges.
_LAPLACIAN = {
    1: np.array([[0.0, 1.0, 0.0],
                 [1.0, -4.0, 1.0],
                 [0.0, 1.0, 0.0]]),
    3: np.array([[2.0, 0.0, 2.0],
                 [0.0, -8.0, 0.0],
                 [2.0, 0.0, 2.0]]),
    5: np.array([[2.0, 4.0, 4.0, 4.0, 2.0],
                 [4.0, 0.0, -8.0, 0.0, 4.0],
                 [4.0, -8.0, -24.0, -8.0, 4.0],
                 [4.0, 0.0, -8.0, 0.0, 4.0],
                 [2.0, 4.0, 4.0, 4.0, 2.0]]),
}


def Laplacian(src, ddepth, dst=None, ksize=1, scale=1, delta=0,
              borderType=BORDER_DEFAULT):
    """`cv2.Laplacian`, for a 3 by 3 aperture.

    Done in numpy rather than through `filter_2d`, because its one caller --
    `imgaug.augmenters.artistic`'s Laplacian edge finder -- asks for
    `CV_64F` on a float64 image, and the kernels are uint8, uint16 and
    float32. A 3 by 3 correlation over a reflected border is small enough
    that doing it here costs nothing and keeps the precision the caller
    asked for.
    """
    if int(ksize) not in _LAPLACIAN:
        raise error(
            "Laplacian: only apertures 1, 3 and 5 are implemented, not "
            "{}".format(ksize))

    source = np.asarray(src)

    if source.ndim not in (2, 3):
        raise error("Laplacian: wants a 2-D or 3-D image")

    target = (np.float64 if ddepth in (CV_64F, None)
              else _depth_dtype(ddepth, "Laplacian"))
    kernel = _LAPLACIAN[int(ksize)] * float(scale)
    radius = kernel.shape[0] // 2
    border = _border(borderType, "Laplacian")
    padded = _pad_for_kernel(source.astype(np.float64), radius, border)

    values = np.zeros(source.shape, dtype=np.float64)

    for row in range(kernel.shape[0]):
        for column in range(kernel.shape[1]):
            weight = kernel[row, column]
            if weight == 0.0:
                continue
            window = padded[row:row + source.shape[0],
                            column:column + source.shape[1]]
            values += weight * window

    return _out(_saturate(values + float(delta), target), dst)


def _pad_for_kernel(array, radius, border):
    """One border rule's padding, as `np.pad` spells it."""
    modes = {"constant": "constant", "replicate": "edge",
             "reflect": "symmetric", "reflect_101": "reflect",
             "wrap": "wrap"}
    width = [(radius, radius), (radius, radius)] + \
            [(0, 0)] * (array.ndim - 2)

    return np.pad(array, width, mode=modes[border])


#: OpenCV's `fastAtan2` polynomial, in degrees per term.
#:
#: Reproduced rather than replaced with `numpy.arctan2`, and the reason is
#: consistency rather than fidelity for its own sake. `warpPolar` uses this
#: angle to build its map, and `imgaug`'s `WithPolarWarping` then puts
#: **keypoints** through `cartToPolar` and expects them to land where the
#: pixels went. An accurate `arctan2` in one and OpenCV's third-of-a-degree
#: approximation in the other would pull the two apart by a fraction of a
#: pixel everywhere, which is exactly the class of bug a coordinate
#: transform should not have.
_ATAN_P1 = np.float32(0.9997878412794807) * np.float32(180.0 / np.pi)
_ATAN_P3 = np.float32(-0.3258083974640975) * np.float32(180.0 / np.pi)
_ATAN_P5 = np.float32(0.1555786518463281) * np.float32(180.0 / np.pi)
_ATAN_P7 = np.float32(-0.04432655554792128) * np.float32(180.0 / np.pi)

#: `DBL_EPSILON` narrowed to float, which is the guard OpenCV divides by.
_ATAN_GUARD = np.float32(np.finfo(np.float64).eps)


def _fast_atan2(y, x):
    """OpenCV's `fastAtan2`, in degrees over `[0, 360)`."""
    down = np.asarray(y, dtype=np.float32)
    across = np.asarray(x, dtype=np.float32)

    flat = np.abs(across)
    steep = np.abs(down)
    shallow = flat >= steep

    ratio = np.where(shallow, steep / (flat + _ATAN_GUARD),
                     flat / (steep + _ATAN_GUARD)).astype(np.float32)
    squared = (ratio * ratio).astype(np.float32)

    degrees = ((((_ATAN_P7 * squared + _ATAN_P5) * squared + _ATAN_P3) *
                squared + _ATAN_P1) * ratio).astype(np.float32)
    degrees = np.where(shallow, degrees,
                       np.float32(90.0) - degrees).astype(np.float32)
    degrees = np.where(across < 0.0, np.float32(180.0) - degrees,
                       degrees).astype(np.float32)
    degrees = np.where(down < 0.0, np.float32(360.0) - degrees,
                       degrees).astype(np.float32)

    return degrees


def cartToPolar(x, y, magnitude=None, angle=None, angleInDegrees=False):
    """`cv2.cartToPolar`: `(magnitude, angle)`, the angle in `[0, 2pi)`.

    The angle is OpenCV's polynomial rather than `numpy.arctan2`, because
    `warpPolar` below is built on the same one -- see `_fast_atan2`.
    """
    first = np.asarray(x, dtype=np.float32)
    second = np.asarray(y, dtype=np.float32)

    radius = np.hypot(first, second).astype(np.float32)
    theta = _fast_atan2(second, first)

    if not angleInDegrees:
        theta = np.deg2rad(theta).astype(np.float32)

    return _out(radius, magnitude), _out(theta, angle)


def warpPolar(src, dsize, center, maxRadius, flags):
    """`cv2.warpPolar`, linear only, forward and inverse.

    Forward: the output's columns are radius and its rows are angle, sampled
    at `rho * maxRadius / width` and `phi * 2pi / height` -- **no half-pixel
    offset**, which is not what the obvious reading of OpenCV's source
    suggests and is what a search over the four placements found. A `dsize` of
    zero takes OpenCV's derived size, `round(maxRadius)` by
    `round(maxRadius * pi)`.

    Inverse: the polar image is padded by one row top and bottom with
    **wrap**, because the angle axis is a circle and a sample between the last
    row and the first has to see both. That padding is OpenCV's too.

    Semi-log (`WARP_POLAR_LOG`) is not implemented and says so.
    """
    if int(flags) & WARP_POLAR_LOG:
        raise error("warpPolar: the semi-log mapping is not implemented")

    source = np.asarray(src)
    inverse = bool(int(flags) & WARP_INVERSE_MAP)
    interpolation = _interpolation(flags, "warpPolar")
    radius_limit = float(maxRadius)
    centre_x, centre_y = float(center[0]), float(center[1])

    if not inverse:
        width, height = (int(dsize[0]), int(dsize[1])) if dsize else (0, 0)

        if width <= 0:
            width = int(np.rint(radius_limit))
        if height <= 0:
            height = int(np.rint(radius_limit * np.pi))

        rho = np.arange(width, dtype=np.float64) * (radius_limit / width)
        phi = np.arange(height, dtype=np.float64) * (2.0 * np.pi / height)

        map_x = (rho[None, :] * np.cos(phi)[:, None] + centre_x)
        map_y = (rho[None, :] * np.sin(phi)[:, None] + centre_y)

        return remap(source, map_x.astype(np.float32),
                     map_y.astype(np.float32), interpolation=int(flags),
                     borderMode=BORDER_CONSTANT)

    width, height = int(dsize[0]), int(dsize[1])

    if width <= 0 or height <= 0:
        raise error("warpPolar: the inverse map needs a target size")

    polar_height, polar_width = source.shape[:2]
    bordered = copyMakeBorder(source, 1, 1, 0, 0, BORDER_WRAP)

    rows, columns = np.mgrid[0:height, 0:width].astype(np.float64)
    across = columns - centre_x
    down = rows - centre_y

    map_x = np.hypot(across, down) * (polar_width / radius_limit)
    map_y = np.deg2rad(_fast_atan2(down, across).astype(np.float64)) * \
        (polar_height / (2.0 * np.pi)) + 1.0

    return remap(bordered, map_x.astype(np.float32), map_y.astype(np.float32),
                 interpolation=int(flags), borderMode=BORDER_CONSTANT)


def polarToCart(magnitude, angle, x=None, y=None, angleInDegrees=False):
    """`cv2.polarToCart`."""
    radius = np.asarray(magnitude, dtype=np.float32)
    theta = np.asarray(angle, dtype=np.float32)

    if angleInDegrees:
        theta = np.deg2rad(theta)

    return (_out((radius * np.cos(theta)).astype(np.float32), x),
            _out((radius * np.sin(theta)).astype(np.float32), y))


def convertMaps(map1, map2, dstmap1type, dstmap2=None, nninterpolation=False):
    """`cv2.convertMaps`, which here converts nothing.

    OpenCV packs a pair of float maps into a fixed-point `CV_16SC2` pair so
    that `remap` can index a precomputed weight table -- whole pixels in one
    map, a fifth of a bit of fraction in the other. `remap` here takes the
    float maps directly, so the pair is handed back as it came and the
    sampling is done at full precision.

    That is a **difference, not an optimisation removed**: OpenCV's fixed
    point quantises a source coordinate to 1/32 of a pixel before it
    interpolates, so a remap through converted maps is not the same image as
    a remap through the float ones. Ours is the float one.
    """
    if int(dstmap1type) not in (CV_16SC2, CV_32FC1, CV_32FC2):
        raise error("convertMaps: no map type {}".format(dstmap1type))

    return (np.ascontiguousarray(map1, dtype=np.float32),
            np.ascontiguousarray(map2, dtype=np.float32))


#: What `setRNGSeed` last set, for `kmeans` to draw from. OpenCV has a global
#: RNG and `imgaug` sets it precisely to make `kmeans` reproducible; this is
#: the same contract without the global state anything else could reach.
_rng_seed = 0


def kmeans(data, K, bestLabels, criteria, attempts, flags, centers=None):
    """`cv2.kmeans`: `(compactness, labels, centres)`.

    **Not a port, and cannot be one.** `imgaug`'s own comment at the call
    site says why: cv2's k-means draws from OpenCV's global RNG, is not
    deterministic without `setRNGSeed`, and gives no way to read the state
    back. What it asks for is a *reproducible* quantisation, and this gives
    exactly that -- k-means++ seeding from `setRNGSeed`'s value, then Lloyd's
    algorithm to the same termination criteria.
    """
    samples = np.asarray(data, dtype=np.float64).reshape(len(data), -1)
    clusters = int(K)

    if clusters < 1 or clusters > len(samples):
        raise error(
            "kmeans: {} clusters for {} samples".format(clusters,
                                                        len(samples)))

    kind, max_iterations, epsilon = criteria
    max_iterations = (int(max_iterations) if int(kind) & TERM_CRITERIA_MAX_ITER
                      else 100)
    epsilon = float(epsilon) if int(kind) & TERM_CRITERIA_EPS else 0.0

    best = None

    for attempt in range(max(int(attempts), 1)):
        rng = np.random.default_rng(_rng_seed + attempt)
        middles = _kmeans_pp(samples, clusters, rng)
        labels = np.zeros(len(samples), dtype=np.int32)

        for _step in range(max(max_iterations, 1)):
            distances = ((samples[:, None, :] - middles[None, :, :]) ** 2
                         ).sum(axis=2)
            labels = np.argmin(distances, axis=1).astype(np.int32)
            moved = 0.0
            for cluster in range(clusters):
                members = samples[labels == cluster]
                if not len(members):
                    continue
                centre = members.mean(axis=0)
                moved = max(moved,
                            float(((centre - middles[cluster]) ** 2).sum()))
                middles[cluster] = centre
            if moved <= epsilon * epsilon:
                break

        distances = ((samples[:, None, :] - middles[None, :, :]) ** 2
                     ).sum(axis=2)
        labels = np.argmin(distances, axis=1).astype(np.int32)
        compactness = float(distances[np.arange(len(samples)), labels].sum())

        if best is None or compactness < best[0]:
            best = (compactness, labels, middles)

    compactness, labels, middles = best

    return (compactness, labels.reshape(-1, 1),
            _out(middles.astype(np.float32), centers))


def _kmeans_pp(samples, clusters, rng):
    """k-means++ seeding: each centre drawn in proportion to its distance."""
    middles = np.empty((clusters, samples.shape[1]), dtype=np.float64)
    middles[0] = samples[rng.integers(len(samples))]

    for chosen in range(1, clusters):
        distances = ((samples[:, None, :] - middles[None, :chosen, :]) ** 2
                     ).sum(axis=2).min(axis=1)
        total = distances.sum()
        if total <= 0.0:
            middles[chosen] = samples[rng.integers(len(samples))]
            continue
        middles[chosen] = samples[rng.choice(len(samples), p=distances / total)]

    return middles


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def connectedComponentsWithStats(image, labels=None, stats=None,
                                 centroids=None, connectivity=8,
                                 ltype=CV_32S):
    """`cv2.connectedComponentsWithStats`.

    Returns `(count, labels, stats, centroids)`, with `stats` columns
    `left, top, width, height, area` and `centroids` the mean `(x, y)` of
    each label, background included as label 0.

    **The numbering is ours, not OpenCV's.** `label_components` partitions the
    mask identically but does not always give a component the same number, so
    a caller comparing label values against a recording of cv2 would be
    disappointed. A caller asking which component is the largest, which is
    what `mmdet`'s visualisation does, gets the same answer.
    """
    count, numbered = _kernels().label_components(
        _native(image, "connectedComponentsWithStats", planes=(1,)),
        int(connectivity))

    height, width = numbered.shape[:2]
    flat = np.asarray(numbered).reshape(-1)
    rows = np.repeat(np.arange(height), width)
    columns = np.tile(np.arange(width), height)

    areas = np.bincount(flat, minlength=count)
    sums_x = np.bincount(flat, weights=columns.astype(np.float64),
                         minlength=count)
    sums_y = np.bincount(flat, weights=rows.astype(np.float64),
                         minlength=count)

    # One pass for each extreme, rather than one pass per label: a mask can
    # have thousands of components and `numbered == label` in a loop is
    # quadratic in the number of them.
    left = np.full(count, width, dtype=np.int64)
    top = np.full(count, height, dtype=np.int64)
    right = np.full(count, -1, dtype=np.int64)
    bottom = np.full(count, -1, dtype=np.int64)
    np.minimum.at(left, flat, columns)
    np.minimum.at(top, flat, rows)
    np.maximum.at(right, flat, columns)
    np.maximum.at(bottom, flat, rows)

    out_stats = np.zeros((count, 5), dtype=np.int32)
    present = areas > 0
    out_stats[present, CC_STAT_LEFT] = left[present]
    out_stats[present, CC_STAT_TOP] = top[present]
    out_stats[present, CC_STAT_WIDTH] = (right - left + 1)[present]
    out_stats[present, CC_STAT_HEIGHT] = (bottom - top + 1)[present]
    out_stats[:, CC_STAT_AREA] = areas

    out_centroids = np.zeros((count, 2), dtype=np.float64)

    with np.errstate(invalid="ignore", divide="ignore"):
        out_centroids[:, 0] = np.where(present, sums_x / areas, 0.0)
        out_centroids[:, 1] = np.where(present, sums_y / areas, 0.0)

    return count, numbered, out_stats, out_centroids


def connectedComponents(image, labels=None, connectivity=8, ltype=CV_32S):
    """`cv2.connectedComponents`: `(count, labels)`."""
    return _kernels().label_components(
        _native(image, "connectedComponents", planes=(1,)), int(connectivity))


def findContours(image, mode, method, contours=None, hierarchy=None,
                 offset=None):
    """`cv2.findContours`, for `RETR_EXTERNAL` and `RETR_CCOMP`.

    Contours come back OpenCV's shape -- a tuple of `(n, 1, 2)` int32 arrays
    -- with a `(1, n, 4)` hierarchy beside them. Under `RETR_CCOMP` the
    hierarchy is the real two-level tree: a hole's `parent` column names the
    component it sits inside.

    **The order of the borders is not OpenCV 5's.** The points are identical
    and so is every hole's parent; the sequence differs when a component has
    more than one hole, because OpenCV 5 reimplemented the flattening.
    Finding 2.74 has the measurement.

    **`RETR_EXTERNAL` here is every component's outer border, where OpenCV's
    is only the outermost.** `find_contours` walks the connected components,
    so a component sitting inside another component's hole -- an island in a
    ring -- gets a contour here and does not in OpenCV, which stops at the top
    level of its tree. On random noise masks that is about one mask in eight.
    Nothing in the vendored packages asks for `RETR_EXTERNAL`; they all want
    `RETR_CCOMP`, where the two agree.
    """
    if offset not in (None, (0, 0)):
        raise error("findContours: the offset argument is not implemented")

    if int(method) not in (CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE):
        raise error("findContours: no chain method {}".format(method))

    kernels = _kernels()
    mask = _native(image, "findContours", planes=(1,))
    simple = int(method) == CHAIN_APPROX_SIMPLE

    if int(mode) == RETR_EXTERNAL:
        traced = [(_opencv_order(points), False)
                  for points in kernels.find_contours(mask)]
    elif int(mode) in (RETR_LIST, RETR_CCOMP):
        traced = list(kernels.find_borders(mask))
    else:
        raise error(
            "findContours: only RETR_EXTERNAL, RETR_LIST and RETR_CCOMP are "
            "implemented, not {}".format(mode))

    if not traced:
        return (), None

    out = []
    holes = []

    for points, is_hole in traced:
        chain = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if simple:
            chain = np.asarray(kernels.approx_poly(chain, 0.0),
                               dtype=np.float64).reshape(-1, 2)
        out.append(np.ascontiguousarray(
            np.rint(chain).astype(np.int32)).reshape(-1, 1, 2))
        holes.append(bool(is_hole))

    return tuple(out), _hierarchy(holes, int(mode))


def _opencv_order(points):
    """One `find_contours` trace in the order and form OpenCV gives.

    Two differences, both of them the kernel's documented behaviour rather
    than a bug: `find_contours` **closes** its contours, repeating the first
    point, and it traces the opposite way round from `cv::findContours`.
    Reversing the closed chain and dropping the duplicate gives OpenCV's
    sequence exactly -- verified over some seven thousand contours.
    """
    chain = np.asarray(points, dtype=np.float64).reshape(-1, 2)

    if len(chain) > 1 and np.array_equal(chain[0], chain[-1]):
        return chain[::-1][:-1]

    return chain


def _hierarchy(holes, mode):
    """OpenCV's `(1, n, 4)` table of `next, previous, first_child, parent`.

    `RETR_LIST` has no tree, so every border is a sibling of the next; under
    `RETR_CCOMP` a hole belongs to the top-level border that precedes it,
    which is the order `find_borders` flattens in.
    """
    count = len(holes)
    table = np.full((count, 4), -1, dtype=np.int32)

    if mode == RETR_LIST:
        for index in range(count):
            if index + 1 < count:
                table[index, 0] = index + 1
            if index:
                table[index, 1] = index - 1
        return table.reshape(1, count, 4)

    parent = -1
    siblings = {-1: []}

    for index, is_hole in enumerate(holes):
        if not is_hole:
            parent = index
            siblings.setdefault(-1, []).append(index)
        else:
            table[index, 3] = parent
            siblings.setdefault(parent, []).append(index)
            if table[parent, 2] < 0:
                table[parent, 2] = index

    for owner, group in siblings.items():
        for position, index in enumerate(group):
            if position + 1 < len(group):
                table[index, 0] = group[position + 1]
            if position:
                table[index, 1] = group[position - 1]

    return table.reshape(1, count, 4)


def contourArea(contour, oriented=False):
    """`cv2.contourArea`, unsigned.

    `oriented` is refused rather than faked: the kernel returns the magnitude
    and the sign is the winding, which it does not report. `moments` gives a
    signed `m00` for a caller that wants one.
    """
    if oriented:
        raise error(
            "contourArea: an oriented area is not implemented; the kernel "
            "returns the magnitude, and moments()['m00'] is signed")

    return _kernels().contour_area(
        np.asarray(contour, dtype=np.float64).reshape(-1, 2))


def arcLength(curve, closed):
    """`cv2.arcLength`."""
    return _kernels().arc_length(
        np.asarray(curve, dtype=np.float64).reshape(-1, 2), bool(closed))


def boundingRect(array):
    """`cv2.boundingRect`: `(x, y, width, height)`."""
    return _kernels().bounding_rect(
        np.asarray(array, dtype=np.float64).reshape(-1, 2))


def minAreaRect(points):
    """`cv2.minAreaRect`: `((cx, cy), (w, h), angle)`."""
    rect = _kernels().min_area_rect(
        np.asarray(points, dtype=np.float64).reshape(-1, 2))

    return (tuple(rect["centre"]), tuple(rect["size"]), float(rect["angle"]))


def boxPoints(box, points=None):
    """`cv2.boxPoints`: the four corners of a rotated rectangle, float32.

    Built from the rectangle rather than read out of `min_area_rect`, so that
    a rectangle assembled by hand -- which is what `mmdeploy` passes -- gives
    the corners OpenCV would give it.
    """
    (centre_x, centre_y), (width, height), angle = box
    radians = np.deg2rad(float(angle))
    cosine, sine = np.cos(radians), np.sin(radians)
    half_w, half_h = float(width) / 2.0, float(height) / 2.0

    # OpenCV's order: bottom left, top left, top right, bottom right, in the
    # image's own axes.
    corners = np.array([[-half_w, half_h], [-half_w, -half_h],
                        [half_w, -half_h], [half_w, half_h]],
                       dtype=np.float64)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)

    moved = corners @ rotation.T + np.array([float(centre_x), float(centre_y)])

    return _out(moved.astype(np.float32), points)


def moments(array, binaryImage=False):
    """`cv2.moments` on a contour, through the second order."""
    return _kernels().moments(
        np.asarray(array, dtype=np.float64).reshape(-1, 2))


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _colour(value, planes):
    """A drawing colour the kernels will take: a scalar or one per plane."""
    if np.isscalar(value):
        return float(value)

    flat = [float(component) for component in np.asarray(value).reshape(-1)]

    return flat[0] if planes == 1 else flat[:planes]


def rectangle(img, pt1, pt2, color, thickness=1, lineType=LINE_8, shift=0):
    """`cv2.rectangle`, drawn in place. The image is returned, as OpenCV's is."""
    array = np.asarray(img)
    _kernels().draw_rect(array, int(pt1[0]), int(pt1[1]), int(pt2[0]),
                         int(pt2[1]), _colour(color, _planes(array)),
                         int(thickness))

    return img


def circle(img, center, radius, color, thickness=1, lineType=LINE_8, shift=0):
    """`cv2.circle`, drawn in place."""
    array = np.asarray(img)
    _kernels().draw_circle(array, int(center[0]), int(center[1]), int(radius),
                           _colour(color, _planes(array)), int(thickness))

    return img


def line(img, pt1, pt2, color, thickness=1, lineType=LINE_8, shift=0):
    """`cv2.line`, drawn in place."""
    array = np.asarray(img)
    _kernels().draw_line(array, int(pt1[0]), int(pt1[1]), int(pt2[0]),
                         int(pt2[1]), _colour(color, _planes(array)),
                         int(thickness))

    return img


def polylines(img, pts, isClosed, color, thickness=1, lineType=LINE_8,
              shift=0):
    """`cv2.polylines`, drawn in place."""
    array = np.asarray(img)

    for chain in pts:
        _kernels().draw_polyline(
            array, np.asarray(chain, dtype=np.float64).reshape(-1, 2),
            _colour(color, _planes(array)), bool(isClosed), int(thickness))

    return img


def fillPoly(img, pts, color, lineType=LINE_8, shift=0, offset=None):
    """`cv2.fillPoly`, drawn in place."""
    array = np.asarray(img)

    for chain in pts:
        _kernels().fill_polygon(
            array, np.asarray(chain, dtype=np.float64).reshape(-1, 2),
            _colour(color, _planes(array)))

    return img


def putText(img, text, org, fontFace, fontScale, color, thickness=1,
            lineType=LINE_8, bottomLeftOrigin=False):
    """`cv2.putText`, drawn in place -- **in a different font**.

    `draw_text` has VIAME's own 5 by 7 bitmap face and an integer scale; a
    Hershey outline font is not in the tree and a caller asking for
    `FONT_HERSHEY_COMPLEX` gets legible text of about the right size in the
    right place, not OpenCV's glyphs. The one caller is
    `mmcv.visualization.imshow_det_bboxes`, which is labelling a picture for
    a person to look at.
    """
    array = np.asarray(img)
    scale = max(int(round(float(fontScale) * 2.0)), 1)
    _kernels().draw_text(array, str(text), int(org[0]), int(org[1]),
                         _colour(color, _planes(array)), scale)

    return img


def getTextSize(text, fontFace, fontScale, thickness):
    """`cv2.getTextSize`: `((width, height), baseline)`."""
    width, height = _kernels().text_size(
        str(text), max(int(round(float(fontScale) * 2.0)), 1))

    return (int(width), int(height)), 0


# ---------------------------------------------------------------------------
# Files
#
# These four keep OpenCV's BGR convention, because mmcv's own API is BGR and
# mmdet reads mmcv's arrays. The swap happens here, once, at the boundary.
# ---------------------------------------------------------------------------

def _read_flags(flags):
    """OpenCV's read flags, as `(grayscale, unchanged)`."""
    flags = int(flags)

    if flags < 0:
        return False, True

    grayscale = (flags & (IMREAD_COLOR | IMREAD_ANYCOLOR)) == 0
    unchanged = bool(flags & IMREAD_ANYDEPTH)

    return grayscale, unchanged


def imread(filename, flags=IMREAD_COLOR):
    """`cv2.imread`: BGR, or None when the file will not decode.

    **A grayscale read is not bit identical to OpenCV's, and cannot be.**
    `cv2.imread(..., IMREAD_GRAYSCALE)` converts inside libpng or libjpeg, with
    `png_set_rgb_to_gray` and its gamma handling; it differs from OpenCV's own
    `cvtColor(imread(path), COLOR_BGR2GRAY)` by a count on about half the
    pixels of a random image. What this gives is that second answer -- decode,
    then convert with the kernel -- which agrees with `cv2.cvtColor` on every
    pixel and with `cv2.imread`'s grey on about half of them, each by one.
    Reproducing libpng's conversion is not worth building for the one config
    key that asks for it.
    """
    from viame.utilities import imageops

    grayscale, unchanged = _read_flags(flags)

    try:
        if unchanged or int(flags) < 0:
            array = imageops.read_unchanged(str(filename))
        else:
            array = imageops.read_image(str(filename))
    except Exception:
        return None

    if grayscale and array.ndim == 3:
        array = imageops.to_gray(array[..., :3])

    return array if array.ndim == 2 else _swap(array)


def imwrite(filename, img, params=None):
    """`cv2.imwrite`, taking BGR. True on success, as OpenCV's is."""
    from viame.utilities import imageops

    array = np.asarray(img)

    try:
        imageops.write_image(str(filename),
                             array if array.ndim == 2 else _swap(array))
    except Exception:
        return False

    return True


def imdecode(buf, flags=IMREAD_COLOR):
    """`cv2.imdecode`: BGR, or None when the bytes will not decode.

    A grayscale decode carries `imread`'s caveat, for the same reason.
    """
    from viame.utilities import imageops

    grayscale, unchanged = _read_flags(flags)
    data = np.asarray(buf, dtype=np.uint8).tobytes()

    try:
        if unchanged or int(flags) < 0:
            array = imageops.decode_unchanged(data)
        else:
            array = imageops.decode_image(data)
    except Exception:
        return None

    if grayscale and array.ndim == 3:
        array = imageops.to_gray(array[..., :3])

    return array if array.ndim == 2 else _swap(array)


def imencode(ext, img, params=None):
    """`cv2.imencode`: `(ok, buffer)`, taking BGR."""
    from viame.utilities import imageops

    array = np.asarray(img)
    quality = None

    if params:
        values = list(params)
        for index in range(0, len(values) - 1, 2):
            if int(values[index]) == IMWRITE_JPEG_QUALITY:
                quality = int(values[index + 1])

    try:
        data = imageops.encode_image(
            array if array.ndim == 2 else _swap(array), str(ext),
            quality=quality)
    except Exception:
        return False, np.zeros(0, dtype=np.uint8)

    return True, np.frombuffer(data, dtype=np.uint8).copy()


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

def VideoWriter_fourcc(*codes):
    """`cv2.VideoWriter_fourcc`: four characters packed into an int."""
    value = 0

    for shift, code in enumerate(codes[:4]):
        value |= (ord(code) & 0xFF) << (8 * shift)

    return value


def _fourcc_name(value):
    """A packed fourcc back to the four characters, for `VideoWriter`."""
    value = int(value)

    return "".join(chr((value >> (8 * shift)) & 0xFF) for shift in range(4))


class VideoCapture:
    """`cv2.VideoCapture` over `viame.video_io.frames.FrameReader`.

    Frames come back **BGR**, as OpenCV's do, from a decode that is otherwise
    the one a VIAME pipeline performs. Only the properties `mmcv.VideoReader`
    reads are answered; an unknown one gives 0.0 rather than pretending.
    """

    def __init__(self, filename=None, apiPreference=None):
        self._reader = None
        self._path = None

        if filename is not None:
            self.open(filename)

    def open(self, filename, apiPreference=None):
        from viame.video_io import frames

        self.release()
        self._path = str(filename)

        try:
            self._reader = frames.FrameReader(self._path)
        except Exception:
            self._reader = None

        return self._reader is not None

    def isOpened(self):
        return self._reader is not None

    def read(self):
        """`(ok, frame)`, the frame BGR."""
        if self._reader is None:
            return False, None

        frame = self._reader.read()

        if frame is None:
            return False, None

        return True, _swap(frame)

    def grab(self):
        return self.read()[0]

    def retrieve(self):
        return self.read()

    def get(self, propId):
        if self._reader is None:
            return 0.0

        reader = self._reader
        table = {
            CAP_PROP_FRAME_WIDTH: lambda: float(reader.width),
            CAP_PROP_FRAME_HEIGHT: lambda: float(reader.height),
            CAP_PROP_FPS: lambda: float(reader.frame_rate),
            CAP_PROP_FRAME_COUNT: lambda: float(reader.frame_count),
            CAP_PROP_POS_FRAMES: lambda: float(reader.position),
            CAP_PROP_POS_MSEC: lambda: (
                1000.0 * reader.position / reader.frame_rate
                if reader.frame_rate else 0.0),
            # There is no container fourcc to report, and OpenCV's own value
            # for a stream it cannot name is 0.
            CAP_PROP_FOURCC: lambda: 0.0,
        }

        return table.get(int(propId), lambda: 0.0)()

    def set(self, propId, value):
        if self._reader is None:
            return False

        if int(propId) == CAP_PROP_POS_FRAMES:
            self._reader.seek(int(value))
            return True

        if int(propId) == CAP_PROP_POS_MSEC:
            rate = self._reader.frame_rate
            if not rate:
                return False
            self._reader.seek(int(round(float(value) * rate / 1000.0)))
            return True

        return False

    def release(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


class VideoWriter:
    """`cv2.VideoWriter` over `viame.video_io.frames.FrameWriter`.

    Takes BGR frames, as OpenCV's does. The fourcc chooses the codec by name
    where PyAV knows it -- `mp4v`, `avc1`, `h264`, `x264` all mean H.264 here
    -- and anything else falls through to the writer's own preference order.
    """

    _CODECS = {"avc1": "libx264", "h264": "libx264", "x264": "libx264",
               "mp4v": "libx264", "xvid": "mpeg4", "mjpg": "mjpeg"}

    def __init__(self, filename=None, fourcc=0, fps=0.0, frameSize=(0, 0),
                 isColor=True):
        self._writer = None
        self._request = None

        if filename is not None:
            self.open(filename, fourcc, fps, frameSize, isColor)

    def open(self, filename, fourcc=0, fps=0.0, frameSize=(0, 0),
             isColor=True):
        from viame.video_io import frames

        self.release()
        codec = self._CODECS.get(_fourcc_name(fourcc).lower())

        try:
            self._writer = frames.FrameWriter(
                str(filename), int(frameSize[0]), int(frameSize[1]),
                float(fps), codec=codec)
        except Exception:
            self._writer = None

        return self._writer is not None

    def isOpened(self):
        return self._writer is not None

    def write(self, image):
        if self._writer is None:
            return

        array = np.asarray(image)
        self._writer.write(array if array.ndim == 2 else _swap(array))

    def release(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Threads, and the windows there are none of
# ---------------------------------------------------------------------------

def setNumThreads(nthreads):
    """`cv2.setNumThreads`, over the kernels' own worker budget.

    Its callers -- `mmdet.utils.setup_env` and `imgaug.multicore` -- call it
    with 0 or 1 to stop a library spawning a thread per core inside a
    dataloader worker that is already one of eight. OpenCV's three cases are
    kept: a negative count restores the default, **0 runs everything on the
    calling thread**, and a positive one is a ceiling. It is a ceiling and not
    a request -- the kernels' pool is built once, from `VIAME_NUM_THREADS` --
    so asking for more than the process was started with does nothing.
    """
    count = int(nthreads)
    _kernels().set_kernel_thread_count(0 if count < 0 else max(count, 1))


def getNumThreads():
    """`cv2.getNumThreads`."""
    return int(_kernels().kernel_thread_count())


def setRNGSeed(seed):
    """`cv2.setRNGSeed`, for the one function here that draws at all.

    OpenCV's global RNG is the reason `grabCut` is not a function of its own
    input (finding 2.71), and every kernel here that needs randomness takes
    its seed instead. `kmeans` is the exception, because its caller reaches
    for this to make it reproducible -- so this records the seed and `kmeans`
    uses it. Nothing else in this module has state to set.
    """
    global _rng_seed

    _rng_seed = int(seed) & 0xFFFFFFFF


def _no_windows(name):
    def missing(*_arguments, **_keywords):
        raise error(
            "{}: there is no window system here. VIAME's tools display "
            "through matplotlib; a pipeline writes an image or a "
            "video.".format(name))

    return missing


imshow = _no_windows("imshow")
namedWindow = _no_windows("namedWindow")
destroyWindow = _no_windows("destroyWindow")
destroyAllWindows = _no_windows("destroyAllWindows")
getWindowProperty = _no_windows("getWindowProperty")
setWindowProperty = _no_windows("setWindowProperty")
resizeWindow = _no_windows("resizeWindow")


def waitKey(delay=0):
    """`cv2.waitKey`: nothing was pressed, because there is no window.

    Not an error, unlike the window functions: `imshow` above has already
    refused by the time anything calls this, and a caller that reaches it
    another way is polling, which -1 answers honestly.
    """
    return -1
