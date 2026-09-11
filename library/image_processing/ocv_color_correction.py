# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Colour correction for underwater imagery, on cv2.

`plugins/opencv/apply_color_correction.cxx` in python, beside
`ocv_enhancer` and for the same reason: it is not an imgproc primitive but
an application built out of a dozen of them -- a lookup table, a masked
mean, a morphological backscatter estimate, CLAHE in Lab, an exponential
attenuation model and a three-way fusion. Keeping the chain in python keeps
it one line per step and holds it to `tests/golden/opencv`'s thirteen
recorded cases exactly.

No shipped pipeline selects this filter. It is kept rather than removed
because `examples/image_enhancement/README.rst` documents it with a worked
example, so it is a feature that happens not to be wired into a pipeline.

**`water_type` does nothing.** The C++ has a `set_water_type_presets()` that
would set the three attenuation coefficients from `oceanic`, `coastal` or
`turbid`, and **nothing ever calls it**: the coefficients always come from
`red_attenuation`, `green_attenuation` and `blue_attenuation`. The recording
proves it -- `coastal` and `turbid` produce output byte-identical to the
default. Reproduced rather than fixed, because the README describes the
presets and a user who has been getting oceanic coefficients while asking
for turbid ones has tuned around it. `design/lite-findings.md` records it.

Everything happens on a **BGR** array, as the C++ did: the channel indices
are load-bearing here -- index 0 is blue and index 2 is red throughout the
attenuation model and the gray world balance -- and so are `BGR2GRAY` and
`BGR2Lab`.
"""

import logging

import numpy as np

from kwiver.vital.algo import ImageFilter
from kwiver.vital.types import Image, ImageContainer

logger = logging.getLogger(__name__)

# `cv::createCLAHE` in the fusion path, which sets both explicitly.
FUSION_CLIP_LIMIT = 2.0
FUSION_TILE_GRID = (8, 8)

# The gamma the fusion path's shadow-recovery pass uses, and the three
# weights it fuses with.
FUSION_GAMMA = 0.7
FUSION_WHITE_BALANCE_WEIGHT = 0.4
FUSION_CLAHE_WEIGHT = 0.4
FUSION_GAMMA_WEIGHT = 0.2

# `estimate_auto_gamma` targets middle gray, and clamps.
AUTO_GAMMA_TARGET = 0.5
AUTO_GAMMA_MIN = 0.1
AUTO_GAMMA_MAX = 5.0

# The backscatter estimate's structuring element is a twentieth of the
# longer side, odd, and at least this.
BACKSCATTER_MIN_KERNEL = 5
BACKSCATTER_KERNEL_DIVISOR = 20
BACKSCATTER_WEIGHT = 0.5


def _as_bool(value):
    return str(value).strip().lower() in ("true", "yes", "on", "1")


def _config_bool(value):
    return "true" if value else "false"


def _config_double(value):
    return "%g" % float(value)


class ApplyColorCorrection(ImageFilter):
    """Gamma, gray world balance and underwater compensation."""

    def __init__(self):
        ImageFilter.__init__(self)

        self._apply_gamma = False
        self._gamma = 1.0
        self._gamma_auto = False
        self._apply_gray_world = False
        self._gray_world_sat_threshold = 0.95
        self._apply_underwater = False
        self._underwater_method = "simple"
        self._depth_map_path = ""
        self._use_auto_depth = True
        self._water_type = "oceanic"
        self._red_attenuation = 0.5
        self._green_attenuation = 0.3
        self._blue_attenuation = 0.1
        self._backscatter_removal = True

        self._depth_map = None

    # ------------------------------------------------------------------
    # Configuration

    def get_configuration(self):
        cfg = super(ImageFilter, self).get_configuration()
        cfg.set_value("apply_gamma", _config_bool(self._apply_gamma))
        cfg.set_value("gamma", _config_double(self._gamma))
        cfg.set_value("gamma_auto", _config_bool(self._gamma_auto))
        cfg.set_value("apply_gray_world",
                      _config_bool(self._apply_gray_world))
        cfg.set_value("gray_world_sat_threshold",
                      _config_double(self._gray_world_sat_threshold))
        cfg.set_value("apply_underwater",
                      _config_bool(self._apply_underwater))
        cfg.set_value("underwater_method", self._underwater_method)
        cfg.set_value("depth_map_path", self._depth_map_path)
        cfg.set_value("use_auto_depth", _config_bool(self._use_auto_depth))
        cfg.set_value("water_type", self._water_type)
        cfg.set_value("red_attenuation",
                      _config_double(self._red_attenuation))
        cfg.set_value("green_attenuation",
                      _config_double(self._green_attenuation))
        cfg.set_value("blue_attenuation",
                      _config_double(self._blue_attenuation))
        cfg.set_value("backscatter_removal",
                      _config_bool(self._backscatter_removal))
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)

        self._apply_gamma = _as_bool(cfg.get_value("apply_gamma"))
        self._gamma = float(cfg.get_value("gamma"))
        self._gamma_auto = _as_bool(cfg.get_value("gamma_auto"))
        self._apply_gray_world = _as_bool(cfg.get_value("apply_gray_world"))
        self._gray_world_sat_threshold = float(
            cfg.get_value("gray_world_sat_threshold"))
        self._apply_underwater = _as_bool(cfg.get_value("apply_underwater"))
        self._underwater_method = str(cfg.get_value("underwater_method"))
        self._depth_map_path = str(cfg.get_value("depth_map_path"))
        self._use_auto_depth = _as_bool(cfg.get_value("use_auto_depth"))
        self._water_type = str(cfg.get_value("water_type"))
        self._red_attenuation = float(cfg.get_value("red_attenuation"))
        self._green_attenuation = float(cfg.get_value("green_attenuation"))
        self._blue_attenuation = float(cfg.get_value("blue_attenuation"))
        self._backscatter_removal = _as_bool(
            cfg.get_value("backscatter_removal"))

        self._depth_map = None

        # `set_water_type_presets()` would belong here. It is not called,
        # here or in the C++; see the module docstring.

    def check_configuration(self, cfg):
        valid = True

        if float(cfg.get_value("gamma", _config_double(self._gamma))) <= 0.0:
            logger.error("Gamma value must be positive")
            valid = False

        return valid

    # ------------------------------------------------------------------
    # Gamma

    def _auto_gamma(self, image):
        """The gamma that would bring the mean to middle gray."""
        import cv2

        gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if image.ndim == 3 and image.shape[2] == 3 else image)

        mean = float(np.asarray(gray, dtype=np.float64).mean() / 255.0)

        if mean > 0.001 and abs(mean - AUTO_GAMMA_TARGET) > 0.01:
            gamma = np.log(AUTO_GAMMA_TARGET) / np.log(mean)
            return float(max(AUTO_GAMMA_MIN, min(AUTO_GAMMA_MAX, gamma)))

        return 1.0

    def _gamma_correct(self, image, gamma=None):
        """`apply_gamma_correction`, including the way `gamma_auto` wins.

        The C++ reads `c_gamma` and then overwrites it from the histogram
        whenever `gamma_auto` is set -- so the fusion path, which sets
        `c_gamma = 0.7` around its shadow-recovery pass, does **not** get
        0.7 when `gamma_auto` is on: it gets the automatic gamma of the
        image it is handed. That is not obviously intended and it is what
        the recording shows, so it is what happens here.
        """
        import cv2

        gamma = self._gamma if gamma is None else gamma

        if self._gamma_auto:
            gamma = self._auto_gamma(image)

        if abs(gamma - 1.0) < 0.001:
            return image

        table = np.clip(
            np.rint(np.power(np.arange(256) / 255.0, 1.0 / gamma) * 255.0),
            0, 255).astype(np.uint8)

        source = image

        if source.dtype != np.uint8:
            source = np.clip(
                np.rint(cv2.normalize(source, None, 255, 0,
                                      cv2.NORM_MINMAX).astype(np.float64)),
                0, 255).astype(np.uint8)

        return cv2.LUT(source, table)

    # ------------------------------------------------------------------
    # Gray world

    def _gray_world(self, image):
        """Scale each channel to the mean of the three, ignoring bright pixels."""
        import cv2

        if image.ndim != 3 or image.shape[2] != 3:
            return image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        top = 255.0 if image.dtype == np.uint8 else 1.0
        mask = gray < (self._gray_world_sat_threshold * top)

        if not mask.any():
            means = np.zeros(3)
        else:
            means = np.array([image[..., c][mask].mean() for c in range(3)])

        average = float(means.sum() / 3.0)

        if average < 0.001:
            return image

        out = np.empty_like(image)

        for channel in range(3):
            scale = average / (means[channel] + 0.001)
            values = image[..., channel].astype(np.float64) * scale
            out[..., channel] = np.clip(np.rint(values), 0, 255).astype(
                image.dtype)

        return out

    # ------------------------------------------------------------------
    # Underwater

    def _load_depth_map(self):
        import cv2

        if self._depth_map_path and self._depth_map is None:
            loaded = cv2.imread(self._depth_map_path,
                                cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)

            if loaded is not None and loaded.size:
                self._depth_map = cv2.normalize(
                    loaded, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

        return self._depth_map

    def _relative_depth(self, image):
        """Depth from the blue over red ratio: red attenuates faster."""
        import cv2

        if image.ndim != 3 or image.shape[2] != 3:
            return np.full(image.shape[:2], 0.5, dtype=np.float32)

        blue = image[..., 0].astype(np.float32)
        red = image[..., 2].astype(np.float32)

        ratio = cv2.divide(blue, red + 1.0)

        return cv2.normalize(ratio, None, 0, 1, cv2.NORM_MINMAX)

    def _remove_backscatter(self, image):
        """Erode to estimate the scattered component, then subtract half of it."""
        import cv2

        if image.ndim != 3 or image.shape[2] != 3:
            return image

        size = max(image.shape[0], image.shape[1]) // BACKSCATTER_KERNEL_DIVISOR
        size = max(BACKSCATTER_MIN_KERNEL, size | 1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

        as_float = image.astype(np.float32)
        backscatter = cv2.erode(as_float, kernel)
        backscatter = cv2.GaussianBlur(backscatter, (0, 0), size / 2.0)

        result = as_float - backscatter * BACKSCATTER_WEIGHT
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)

        return np.clip(np.rint(result.astype(np.float64)), 0,
                       255).astype(np.uint8)

    def _underwater_simple(self, image):
        import cv2

        if image.ndim != 3 or image.shape[2] != 3:
            return image

        depth_map = self._load_depth_map()

        if depth_map is not None and depth_map.size:
            if depth_map.shape[:2] != image.shape[:2]:
                depth = cv2.resize(depth_map,
                                   (image.shape[1], image.shape[0]))
            else:
                depth = depth_map.copy()
        elif self._use_auto_depth:
            depth = self._relative_depth(image)
        else:
            depth = np.full(image.shape[:2], 0.5, dtype=np.float32)

        if self._backscatter_removal:
            image = self._remove_backscatter(image)

        attenuations = (self._blue_attenuation, self._green_attenuation,
                        self._red_attenuation)

        out = np.empty_like(image)

        for channel, attenuation in enumerate(attenuations):
            factor = cv2.exp(depth * np.float32(attenuation))
            values = image[..., channel].astype(np.float32) * factor

            # `cv::threshold( ..., THRESH_TRUNC )`: an upper clamp only.
            values = np.minimum(values, 255.0)

            out[..., channel] = np.clip(
                np.rint(values.astype(np.float64)), 0, 255).astype(np.uint8)

        return cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)

    def _underwater_fusion(self, image):
        """Fuse a white-balanced, a CLAHE and a gamma-lifted version."""
        import cv2

        if image.ndim != 3 or image.shape[2] != 3:
            return image

        balanced = self._gray_world(image.copy())

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        planes = list(cv2.split(lab))

        clahe = cv2.createCLAHE(clipLimit=FUSION_CLIP_LIMIT,
                                tileGridSize=FUSION_TILE_GRID)
        planes[0] = clahe.apply(planes[0])

        enhanced = cv2.cvtColor(cv2.merge(planes), cv2.COLOR_Lab2BGR)

        lifted = self._gamma_correct(image.copy(), FUSION_GAMMA)

        result = cv2.addWeighted(balanced, FUSION_WHITE_BALANCE_WEIGHT,
                                 enhanced, FUSION_CLAHE_WEIGHT, 0)
        result = cv2.addWeighted(result, 1.0, lifted, FUSION_GAMMA_WEIGHT, 0)

        return self._gray_world(result)

    # ------------------------------------------------------------------

    def filter(self, image_data):
        import cv2

        if image_data is None:
            return image_data

        array = image_data.asarray()

        colour = array.ndim == 3 and array.shape[2] >= 3

        if colour:
            image = np.ascontiguousarray(array[:, :, 2::-1])
        elif array.ndim == 3:
            image = np.ascontiguousarray(array[:, :, 0])
        else:
            image = np.ascontiguousarray(array)

        if image.dtype != np.uint8:
            normalised = cv2.normalize(image, None, 255, 0, cv2.NORM_MINMAX)
            image = np.clip(np.rint(normalised.astype(np.float64)), 0,
                            255).astype(np.uint8)

        if self._apply_gamma:
            image = self._gamma_correct(image)

        if self._apply_gray_world:
            image = self._gray_world(image)

        if self._apply_underwater:
            if self._underwater_method == "fusion":
                image = self._underwater_fusion(image)
            else:
                image = self._underwater_simple(image)

        if colour and image.ndim == 3:
            image = np.ascontiguousarray(image[:, :, ::-1])

        return ImageContainer(Image(np.ascontiguousarray(image)))


def __vital_algorithm_register__():
    from viame.core.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        ApplyColorCorrection, "ocv_color_correction",
        "Color correction algorithms: gamma, underwater compensation, gray world white balance")
