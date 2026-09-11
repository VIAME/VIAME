# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""SIFT and SURF feature detection and description, on cv2.

`library/image_processing/feature_detect_extract_{SIFT,SURF}.cxx` in python,
per `lite-removals.md` section 2.4. SIFT and SURF are not `image_ops`
primitives -- each is a scale-space pyramid, an extremum search, an
orientation assignment and a gradient histogram, and both are held to a
recording of what OpenCV produced -- so the algorithm stays OpenCV's and only
the language changes.

Four names, two per algorithm, because vital splits detection from
description: `detect_features:ocv_SIFT` and `extract_descriptors:ocv_SIFT`,
and the same for SURF. Each pair shares its config keys, since each wraps one
cv2 object that does both.

The image conversion is the C++ bridge's: it built a BGR `cv::Mat` and let
the detector take `COLOR_BGR2GRAY` on it, which is the right luminance of
the original RGB. So this converts with `COLOR_RGB2GRAY` and hands over one
channel -- see `_to_cv_image`.

**SURF needs a cv2 built with the non-free modules.** It is
`cv2.xfeatures2d.SURF_create`, which the `opencv-python-headless` wheel does
not carry -- the build VIAME ships does. `ocv_SURF` registers either way, so
the name is never silently missing from the registry, and says so when asked
to run. See `design/lite-findings.md`.
"""

import logging

from kwiver.vital.algo import DetectFeatures, ExtractDescriptors

from viame.image_processing.ocv_feature_types import (OCVFeatureSet,
                                                      descriptors_to_set,
                                                      features_to_keypoints)

logger = logging.getLogger(__name__)


def _config_double(value):
    """Spell a double as `config_block` spelled the C++ one: 1.6, not 1.6."""
    return "%g" % float(value)


def _as_bool(value):
    return str(value).strip().lower() in ("true", "yes", "on", "1")


def _config_bool(value):
    return "true" if value else "false"


# ----------------------------------------------------------------------------
# The two algorithms, as the config keys they carry and the cv2 call they make
# ----------------------------------------------------------------------------

class _SIFT(object):
    """`cv::SIFT`, with the C++ wrapper's key names and defaults."""

    name = "ocv_SIFT"
    description = "OpenCV feature detection via the SIFT algorithm"

    defaults = (
        ("n_features", 0, int),
        ("n_octave_layers", 3, int),
        ("contrast_threshold", 0.04, float),
        ("edge_threshold", 10, int),
        ("sigma", 1.6, float),
    )

    @staticmethod
    def create(values):
        import cv2

        # Positional, in the order `cv::SIFT::create` declares them, which is
        # the order the C++ wrapper passed them in.
        return cv2.SIFT_create(
            values["n_features"],
            values["n_octave_layers"],
            values["contrast_threshold"],
            values["edge_threshold"],
            values["sigma"])


class _SURF(object):
    """`cv::xfeatures2d::SURF`, likewise.

    Note `n_octaves_layers`, with the extra s: that is the key the C++
    wrapper registered, and `registry.json` records it, so it stays.
    """

    name = "ocv_SURF"
    description = "OpenCV feature detection via the SURF algorithm"

    defaults = (
        ("hessian_threshold", 100, float),
        ("n_octaves", 4, int),
        ("n_octaves_layers", 3, int),
        ("extended", False, bool),
        ("upright", False, bool),
    )

    @staticmethod
    def create(values):
        import cv2

        try:
            factory = cv2.xfeatures2d.SURF_create
        except AttributeError:
            raise RuntimeError(
                "ocv_SURF needs a cv2 built with the non-free modules, and "
                "this one has no cv2.xfeatures2d.SURF_create. The "
                "opencv-python-headless wheel is built without them; use "
                "ocv_SIFT, which is free and in every build.")

        return factory(
            values["hessian_threshold"],
            values["n_octaves"],
            values["n_octaves_layers"],
            values["extended"],
            values["upright"])


# ----------------------------------------------------------------------------
# Configuration, shared by the detector and the extractor of each algorithm
# ----------------------------------------------------------------------------

class _ConfiguredByAlgorithm(object):
    """The config half of a wrapper: the keys, and the cv2 object they make.

    `get_configuration` is not here, and cannot be. pybind11 stops an
    override calling its own C++ base by comparing the calling python frame
    with the override it is about to dispatch to -- so `super().method()`
    reaches C++ only when it is written *inside* the override itself. In a
    helper one frame down the guard does not fire and `get_configuration`
    calls itself until the stack runs out. So each interface's own
    `get_configuration` holds the `super()` call, and only the filling in is
    shared.
    """

    algorithm = None

    def _initialize(self):
        self._values = {key: default
                        for key, default, _ in self.algorithm.defaults}
        self._cv = None

    def _fill_configuration(self, cfg):
        for key, _, kind in self.algorithm.defaults:
            value = self._values[key]
            if kind is bool:
                cfg.set_value(key, _config_bool(value))
            elif kind is int:
                cfg.set_value(key, str(int(value)))
            else:
                cfg.set_value(key, _config_double(value))

        return cfg

    def _read_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)

        for key, _, kind in self.algorithm.defaults:
            raw = cfg.get_value(key)
            if kind is bool:
                self._values[key] = _as_bool(raw)
            elif kind is int:
                self._values[key] = int(float(raw))
            else:
                self._values[key] = float(raw)

        # The C++ wrapper rebuilt its cv2 object on every configuration
        # change and on every call, because OpenCV 3 and later have no
        # parameter setters. Rebuilding once here is the same thing, since
        # nothing else can change these values.
        self._cv = None

    def _detector(self):
        if self._cv is None:
            self._cv = self.algorithm.create(self._values)

        return self._cv


def _to_cv_image(image_container):
    """The array a cv2 feature algorithm should see.

    Grey, and converted here rather than left to cv2. The C++ bridge built a
    **BGR** `cv::Mat` and handed that over, and cv2 then took
    `COLOR_BGR2GRAY` on it, which is the correct luminance of the original
    RGB. Handing cv2 the RGB array instead and letting it take BGR2GRAY
    weights red and blue the wrong way round -- not a subtle difference: it
    cost ten of eighty-one SIFT keypoints on the first fixture tried.
    """
    import cv2

    array = image_container.asarray()

    if array.ndim == 3 and array.shape[2] >= 3:
        return cv2.cvtColor(array[:, :, :3], cv2.COLOR_RGB2GRAY)

    if array.ndim == 3:
        return array[:, :, 0]

    return array


def _to_cv_mask(mask, image_container):
    """The mask a cv2 detector should see, or None.

    The C++ took the first channel of the mask image, thresholded it at 128
    and required it to match the image's shape.
    """
    import cv2

    if mask is None or mask.size() == 0:
        return None

    if (mask.width() != image_container.width() or
            mask.height() != image_container.height()):
        raise RuntimeError(
            "OCV detect feature algorithm given a non-zero mask with "
            "mismatched shape compared to input image: image is {}x{}, mask "
            "is {}x{}".format(image_container.width(),
                              image_container.height(),
                              mask.width(), mask.height()))

    array = mask.asarray()
    if array.ndim == 3:
        array = array[:, :, 0]

    _, thresholded = cv2.threshold(array, 128, 255, cv2.THRESH_BINARY)
    return thresholded


# ----------------------------------------------------------------------------
# The four implementations
# ----------------------------------------------------------------------------

class _Detector(_ConfiguredByAlgorithm, DetectFeatures):
    def __init__(self):
        DetectFeatures.__init__(self)
        self._initialize()

    def get_configuration(self):
        return self._fill_configuration(
            super(DetectFeatures, self).get_configuration())

    def set_configuration(self, cfg_in):
        self._read_configuration(cfg_in)

    def check_configuration(self, cfg):
        return True

    def detect(self, image_data, mask=None):
        keypoints = self._detector().detect(
            _to_cv_image(image_data), _to_cv_mask(mask, image_data))

        return OCVFeatureSet(keypoints)


class _Extractor(_ConfiguredByAlgorithm, ExtractDescriptors):
    def __init__(self):
        ExtractDescriptors.__init__(self)
        self._initialize()

    def get_configuration(self):
        return self._fill_configuration(
            super(ExtractDescriptors, self).get_configuration())

    def set_configuration(self, cfg_in):
        self._read_configuration(cfg_in)

    def check_configuration(self, cfg):
        return True

    def extract(self, image_data, features, image_mask=None):
        """Returns `(descriptors, features)`.

        The tuple is the convention for a python implementation of a method
        with an output parameter -- see
        `python/kwiver/vital/algo/trampolines/README.md`. The feature set is
        one of them because `detectAndCompute` with provided keypoints may
        drop any it cannot describe, and a caller left holding the old set
        pairs every descriptor after the first drop with the wrong feature.
        """
        if image_data is None or features is None:
            return None, features

        keypoints = features_to_keypoints(features)

        # `compute` rather than the C++'s `detectAndCompute( ..., True )`:
        # cv2's python binding does not expose `useProvidedKeypoints`, and
        # `Feature2D::compute` is that call with the flag set and a null
        # mask -- which is also what the C++ passed, whatever mask it was
        # given, and why this ignores `image_mask` too.
        keypoints, descriptors = self._detector().compute(
            _to_cv_image(image_data), keypoints)

        return descriptors_to_set(descriptors), OCVFeatureSet(keypoints)


class DetectFeaturesSIFT(_Detector):
    algorithm = _SIFT


class ExtractDescriptorsSIFT(_Extractor):
    algorithm = _SIFT


class DetectFeaturesSURF(_Detector):
    algorithm = _SURF


class ExtractDescriptorsSURF(_Extractor):
    algorithm = _SURF


def __vital_algorithm_register__():
    from viame.core.vital_registration import register_vital_algorithm

    for cls in (DetectFeaturesSIFT, ExtractDescriptorsSIFT,
                DetectFeaturesSURF, ExtractDescriptorsSURF):
        register_vital_algorithm(
            cls, cls.algorithm.name, cls.algorithm.description)
