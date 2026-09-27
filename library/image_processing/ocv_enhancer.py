# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Image enhancement, on `viame.image_kernels`.

the `opencv` plugin's `enhance_images.cxx` in python. Thirteen shipped
pipelines select `ocv_enhancer`, which makes it the most-used filter of its
kind in the tree. It stayed python through phase 3 because of one step --
`cv::fastNlMeansDenoisingColored`, a few hundred lines of patch comparison
with its own lookup tables -- and it stays python now that
`image_kernels.denoise_colour` exists, because python is where the six
toggles read clearly and none of the steps is hot.

Everything happens on an **RGB** array. The C++ worked in BGR, and so did
this file until the denoiser was ours: `fastNlMeansDenoisingColored` converts
to CIELAB internally and assumes BGR, so the channel order was load bearing
for exactly one step. It is not any more -- `denoise_colour` takes RGB -- and
every other step is either per channel or symmetric in the order, so the
result is the same array it always was. The conversions that are *not*
symmetric, L*a*b* and HSV, are simply asked for the RGB spelling.

`vxl_enhancer` is registered as a second name for this class. The two were
already the same code at runtime before phase 3: the `vxl` plugin and
the `opencv` plugin each defined `viame::enhance_images` with identical mangled
symbols, so the loader bound one definition for both factories. See
`design/STATUS.md`, P3-T05.
"""

import logging

import numpy as np

from viame import image_kernels as kernels
from viame.algo import ImageFilter
from viame.types import Image, ImageContainer

logger = logging.getLogger(__name__)

# `cv::createCLAHE`'s default tile grid, which the C++ never changed.
CLAHE_TILE_GRID = (8, 8)


def _as_bool(value):
    return str(value).strip().lower() in ("true", "yes", "on", "1")


def _config_bool(value):
    return "true" if value else "false"


def _config_double(value):
    return "%g" % float(value)


def _planar(array):
    """The array as the steps want it: two dimensions when it is grey."""
    if array.ndim == 3 and array.shape[2] == 1:
        return np.ascontiguousarray(array[:, :, 0])

    return np.ascontiguousarray(array)


def _scale_abs(plane, alpha, beta):
    """`cv2.convertScaleAbs`: scale, shift, absolute value, then a byte.

    The absolute value comes *before* the rounding and the saturation, so a
    negative result folds up rather than clamping to zero -- which is what
    the shift in the CLAHE float path relies on not happening, since it is
    built to land the minimum on zero exactly.
    """
    scaled = plane.astype(np.float32) * np.float32(alpha) + np.float32(beta)

    return np.clip(np.rint(np.abs(scaled)), 0, 255).astype(np.uint8)


def _scaled(plane, factor):
    """`plane * factor` the way a `cv::Mat` does it: rounded and saturated."""
    top = np.iinfo(plane.dtype).max if np.issubdtype(plane.dtype, np.integer) \
        else None
    scaled = plane.astype(np.float32) * np.float32(factor)

    if top is None:
        return scaled.astype(plane.dtype)

    return np.clip(np.rint(scaled), 0, top).astype(plane.dtype)


def _blur_size(sigma, eight_bit):
    """The kernel `cv2.GaussianBlur` picks when it is given `(0, 0)`.

    Three sigmas either side for a byte image and four for anything wider,
    forced odd -- `cvRound( sigma * (depth == CV_8U ? 3 : 4) * 2 + 1 ) | 1`.
    """
    reach = 3 if eight_bit else 4

    return int(round(sigma * reach * 2 + 1)) | 1


class EnhanceImages(ImageFilter):
    """Smooth, denoise, white balance, equalise, saturate and sharpen."""

    def __init__(self):
        ImageFilter.__init__(self)

        self._apply_smoothing = False
        self._smoothing_kernel = 3
        self._apply_denoising = False
        self._denoise_kernel = 3
        self._denoise_coeff = 3
        self._force_8bit = False
        self._auto_balance = False
        self._apply_clahe = False
        self._clip_limit = 4
        self._saturation = 1.0
        self._apply_sharpening = False
        self._sharpening_kernel = 3
        self._sharpening_weight = 0.5

    # ------------------------------------------------------------------
    # Configuration

    def get_configuration(self):
        cfg = super(ImageFilter, self).get_configuration()
        cfg.set_value("apply_smoothing", _config_bool(self._apply_smoothing))
        cfg.set_value("smoothing_kernel", str(int(self._smoothing_kernel)))
        cfg.set_value("apply_denoising", _config_bool(self._apply_denoising))
        cfg.set_value("denoise_kernel", str(int(self._denoise_kernel)))
        cfg.set_value("denoise_coeff", str(int(self._denoise_coeff)))
        cfg.set_value("force_8bit", _config_bool(self._force_8bit))
        cfg.set_value("auto_balance", _config_bool(self._auto_balance))
        cfg.set_value("apply_clahe", _config_bool(self._apply_clahe))
        cfg.set_value("clip_limit", str(int(self._clip_limit)))
        cfg.set_value("saturation", _config_double(self._saturation))
        cfg.set_value("apply_sharpening",
                      _config_bool(self._apply_sharpening))
        cfg.set_value("sharpening_kernel", str(int(self._sharpening_kernel)))
        cfg.set_value("sharpening_weight",
                      _config_double(self._sharpening_weight))
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)

        self._apply_smoothing = _as_bool(cfg.get_value("apply_smoothing"))
        self._smoothing_kernel = int(float(cfg.get_value("smoothing_kernel")))
        self._apply_denoising = _as_bool(cfg.get_value("apply_denoising"))
        self._denoise_kernel = int(float(cfg.get_value("denoise_kernel")))
        self._denoise_coeff = int(float(cfg.get_value("denoise_coeff")))
        self._force_8bit = _as_bool(cfg.get_value("force_8bit"))
        self._auto_balance = _as_bool(cfg.get_value("auto_balance"))
        self._apply_clahe = _as_bool(cfg.get_value("apply_clahe"))
        self._clip_limit = int(float(cfg.get_value("clip_limit")))
        self._saturation = float(cfg.get_value("saturation"))
        self._apply_sharpening = _as_bool(cfg.get_value("apply_sharpening"))
        self._sharpening_kernel = int(
            float(cfg.get_value("sharpening_kernel")))
        self._sharpening_weight = float(cfg.get_value("sharpening_weight"))

    def check_configuration(self, cfg):
        return True

    # ------------------------------------------------------------------

    def _denoise(self, image):
        """Non-local means, `fastNlMeansDenoisingColored`'s arguments and all.

        The C++ passed the same coefficient for luminance and for colour, and
        a search window three times the template, so those are not defaults
        being taken -- they are the configuration.

        A single channel image reaches this and is refused, which is what
        OpenCV did: `fastNlMeansDenoisingColored` wants three channels or
        four. The refusal is part of the recorded contract.
        """
        return kernels.denoise_colour(
            image, float(self._denoise_coeff), float(self._denoise_coeff),
            self._denoise_kernel, self._denoise_kernel * 3)

    def _balance(self, image):
        """Scale each channel so all three have the mean of the three.

        `cv::sum` over the whole image, divided by the pixel count, and each
        channel multiplied by the mean of the three over its own. The
        multiply saturates, which is what `Mat * double` does.
        """
        illumination = image.reshape(-1, image.shape[2]).sum(axis=0) / (
            image.shape[0] * image.shape[1])

        scale = illumination.sum() / 3.0

        out = np.empty_like(image)

        for channel in range(image.shape[2]):
            if illumination[channel] == 0:
                out[..., channel] = image[..., channel]
                continue

            values = image[..., channel].astype(np.float64) * (
                scale / illumination[channel])
            out[..., channel] = np.clip(
                np.rint(values), 0, np.iinfo(image.dtype).max
                if np.issubdtype(image.dtype, np.integer) else values.max()
            ).astype(image.dtype)

        return out

    def _clahe(self, image):
        """CLAHE on the lightness channel of Lab, as the C++ did.

        A float image is stretched to bytes first, equalised, and stretched
        back by the reciprocal of a *different* scale than it came in by --
        `scale2` is `max / 255` where `scale1` was `255 / (max - min)`, so
        the shift is not undone. Reproduced, not corrected: no shipped
        config feeds this a float image.
        """
        if image.dtype not in (np.uint8, np.float32):
            image = image.astype(np.float32)

        colour = image.ndim == 3 and image.shape[2] == 3

        lab = kernels.to_lab(image) if colour else image

        planes = [np.ascontiguousarray(lab[..., k]) for k in range(3)] \
            if lab.ndim == 3 else [lab]

        if image.dtype == np.float32:
            low = float(planes[0].min())
            high = float(planes[0].max())
            scale1 = (255.0 / (high - low)) if high > 0.0 else 1.0
            shift1 = -(low * scale1)
            scale2 = (high / 255.0) if high > 0.0 else 1.0

            as_bytes = _scale_abs(planes[0], scale1, shift1)
            planes[0] = self._equalise(as_bytes).astype(np.float32) * (
                1.0 / scale2)
        else:
            planes[0] = self._equalise(planes[0])

        if not colour:
            return planes[0]

        return kernels.from_lab(np.ascontiguousarray(np.stack(planes, -1)))

    def _equalise(self, plane):
        return kernels.clahe(plane, float(self._clip_limit),
                             CLAHE_TILE_GRID[0], CLAHE_TILE_GRID[1])

    def _saturate(self, image):
        """Scale the S of HSV, which is the one step that cannot be exact.

        `cv2.cvtColor( ..., COLOR_HSV2RGB )` is not a function of its input:
        the same HSV triple converts differently depending on where in the row
        it sits, because the vectorised body and the scalar tail round the
        sixth-of-a-hue interpolation differently. `from_hsv` is one answer, the
        right one, and it differs from a recording of OpenCV's on a few pixels
        by a single level. See lite-findings.md 2.56.
        """
        if image.dtype not in (np.uint8, np.float32):
            # cvtColor has no 16 bit HSV, and refusing here keeps that rather
            # than letting numpy widen a uint16 image into the float overload
            # and quietly change both the dtype and the hue scale.
            raise RuntimeError(
                "Unable to adjust saturation on %s imagery" % image.dtype)

        hsv = kernels.to_hsv(image)
        channels = [np.ascontiguousarray(hsv[..., k]) for k in range(3)]

        # `sat *= c_saturation` on a `cv::Mat`: saturating, and rounding.
        channels[1] = _scaled(channels[1], self._saturation)

        return kernels.from_hsv(np.ascontiguousarray(np.stack(channels, -1)))

    def _sharpen(self, image):
        # `cv::Size( 0, 0 )`: the kernel size comes from the sigma, which is
        # `sharpening_kernel` -- misleadingly named, since it is a sigma and
        # not a size.
        sigma = float(self._sharpening_kernel)
        blurred = kernels.gaussian_blur(
            image, _blur_size(sigma, image.dtype == np.uint8), sigma)

        return kernels.add_weighted(image, 1.0 + self._sharpening_weight,
                                    blurred, -self._sharpening_weight, 0.0)

    def filter(self, image_data):
        if image_data is None:
            return image_data

        image = _planar(image_data.asarray())

        eight_bit = image.dtype == np.uint8

        if self._apply_smoothing:
            if not eight_bit and self._smoothing_kernel > 5:
                # medianBlur's wide window is a byte only histogram, so
                # OpenCV refuses anything else past five.
                raise RuntimeError(
                    "Unable to smooth %s imagery with a kernel wider than "
                    "five" % image.dtype)

            image = kernels.median_blur(image, self._smoothing_kernel)

        if self._apply_denoising and eight_bit:
            image = self._denoise(image)

        if self._auto_balance and image.ndim == 3 and image.shape[2] == 3:
            image = self._balance(image)

        if self._force_8bit and image.dtype != np.uint8:
            # `cv::normalize( ..., 255, 0, NORM_MINMAX )` stretches the range
            # across every channel together, and `convertTo( CV_8U )` then
            # rounds. The stretch keeps the input's own type, so the rounding
            # is the second step's and not this one's.
            normalised = kernels.normalize(image, 0.0, 255.0)
            image = np.clip(np.rint(normalised.astype(np.float64)),
                            0, 255).astype(np.uint8)

        if self._apply_denoising and not eight_bit:
            if image.dtype != np.uint8:
                raise RuntimeError(
                    "Unable to perform denoising on not 8-bit imagery")

            image = self._denoise(image)

        if self._apply_clahe:
            image = self._clahe(image)

        if self._saturation != 1.0 and image.ndim == 3 and \
                image.shape[2] == 3:
            image = self._saturate(image)

        if self._apply_sharpening:
            image = self._sharpen(image)

        return ImageContainer(Image(np.ascontiguousarray(image)))


# `vxl_enhancer` is the same implementation under a second name, which is
# what it already was -- see the module docstring. A python implementation is
# discovered by walking the interface's subclasses, so an alias has to be a
# subclass rather than a second registration of one class, and it has to be
# defined at module scope: `__subclasses__` holds weak references, so a class
# created inside the registration function is collected before discovery
# reaches it.
class VXLEnhancer(EnhanceImages):
    """`vxl_enhancer`, which has been this code since before phase 3."""


def __vital_algorithm_register__():
    from viame.utilities.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        EnhanceImages, "ocv_enhancer",
        "Simple illumination normalization using Lab space and CLAHE")

    register_vital_algorithm(
        VXLEnhancer, "vxl_enhancer",
        "Simple illumination normalization using Lab space and CLAHE")
