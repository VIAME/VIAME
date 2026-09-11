# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Image enhancement, on cv2.

`plugins/opencv/enhance_images.cxx` in python. Thirteen shipped pipelines
select `ocv_enhancer`, which makes it the most-used OpenCV filter in the
tree, and the reason it is python rather than `image_ops` is one step of it:
`cv::fastNlMeansDenoisingColored`. Non-local means is not an imgproc
primitive -- it is a few hundred lines of patch comparison with its own
lookup tables -- and reproducing it would be a large piece of work to keep a
feature every shipped config currently disables. But `apply_denoising` is a
DIVE toggle a user can flip, so dropping it is a visible regression.
Keeping the whole filter in python costs neither.

Everything happens on a **BGR** array, which is what the C++ worked on and
what the bridge handed it. Most of the steps are symmetric in the channel
order and would give the same answer either way; the denoiser is not --
`fastNlMeansDenoisingColored` converts to CIELAB internally and assumes BGR
-- so rather than reason about each step, this converts once on the way in
and once on the way out, as the C++ did.

`vxl_enhancer` is registered as a second name for this class. The two were
already the same code at runtime before phase 3: `plugins/vxl` and
`plugins/opencv` each defined `viame::enhance_images` with identical mangled
symbols, so the loader bound one definition for both factories. See
`design/STATUS.md`, P3-T05.
"""

import logging

import numpy as np

from kwiver.vital.algo import ImageFilter
from kwiver.vital.types import Image, ImageContainer

logger = logging.getLogger(__name__)

# `cv::createCLAHE`'s default tile grid, which the C++ never changed.
CLAHE_TILE_GRID = (8, 8)


def _as_bool(value):
    return str(value).strip().lower() in ("true", "yes", "on", "1")


def _config_bool(value):
    return "true" if value else "false"


def _config_double(value):
    return "%g" % float(value)


def _to_bgr(array):
    """The array as the BGR the C++ worked on, and how to put it back."""
    if array.ndim == 2:
        return array, False

    if array.shape[2] == 1:
        return array[:, :, 0], False

    return np.ascontiguousarray(array[:, :, ::-1]), True


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
        import cv2

        return cv2.fastNlMeansDenoisingColored(
            image, None, float(self._denoise_coeff),
            float(self._denoise_coeff), self._denoise_kernel,
            self._denoise_kernel * 3)

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
        import cv2

        if image.dtype not in (np.uint8, np.float32):
            image = image.astype(np.float32)

        colour = image.ndim == 3 and image.shape[2] == 3

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab) if colour else image

        planes = list(cv2.split(lab)) if lab.ndim == 3 else [lab]

        clahe = cv2.createCLAHE(clipLimit=float(self._clip_limit),
                                tileGridSize=CLAHE_TILE_GRID)

        if image.dtype == np.float32:
            low, high, _, _ = cv2.minMaxLoc(planes[0])
            scale1 = (255.0 / (high - low)) if high > 0.0 else 1.0
            shift1 = -(low * scale1)
            scale2 = (high / 255.0) if high > 0.0 else 1.0

            as_bytes = cv2.convertScaleAbs(planes[0], alpha=scale1,
                                           beta=shift1)
            planes[0] = clahe.apply(as_bytes).astype(np.float32) * (
                1.0 / scale2)
        else:
            planes[0] = clahe.apply(planes[0])

        if not colour:
            return planes[0]

        return cv2.cvtColor(cv2.merge(planes), cv2.COLOR_Lab2BGR)

    def _saturate(self, image):
        import cv2

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        channels = list(cv2.split(hsv))

        # `sat *= c_saturation` on a `cv::Mat`: saturating, and rounding.
        channels[1] = cv2.multiply(channels[1], self._saturation)

        return cv2.cvtColor(cv2.merge(channels), cv2.COLOR_HSV2BGR)

    def _sharpen(self, image):
        import cv2

        # `cv::Size( 0, 0 )`: the kernel size comes from the sigma, which is
        # `sharpening_kernel` -- misleadingly named, since it is a sigma and
        # not a size.
        blurred = cv2.GaussianBlur(image, (0, 0),
                                   float(self._sharpening_kernel))

        return cv2.addWeighted(image, 1.0 + self._sharpening_weight,
                               blurred, -self._sharpening_weight, 0)

    def filter(self, image_data):
        import cv2

        if image_data is None:
            return image_data

        source, was_colour = _to_bgr(image_data.asarray())
        image = source.copy()

        eight_bit = image.dtype == np.uint8

        if self._apply_smoothing:
            image = cv2.medianBlur(image, self._smoothing_kernel)

        if self._apply_denoising and eight_bit:
            image = self._denoise(image)

        if self._auto_balance and image.ndim == 3 and image.shape[2] == 3:
            image = self._balance(image)

        if self._force_8bit and image.dtype != np.uint8:
            # `cv::normalize( ..., 255, 0, NORM_MINMAX )` stretches the range
            # across every channel together, and `convertTo( CV_8U )` then
            # rounds.
            normalised = cv2.normalize(image, None, 255, 0, cv2.NORM_MINMAX)
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

        if was_colour and image.ndim == 3:
            image = np.ascontiguousarray(image[:, :, ::-1])

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
    from viame.core.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        EnhanceImages, "ocv_enhancer",
        "Simple illumination normalization using Lab space and CLAHE")

    register_vital_algorithm(
        VXLEnhancer, "vxl_enhancer",
        "Simple illumination normalization using Lab space and CLAHE")
