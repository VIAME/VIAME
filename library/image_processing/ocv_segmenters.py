# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Detection segmentation with GrabCut and watershed, on cv2.

`plugins/opencv/refine_detections_{grabcut,watershed}.cxx` in python, per
`lite-removals.md` section 2.6, which sends both to python by name.
`cv::grabCut` is a Gaussian mixture over the crop, iterated with a min-cut,
and `cv::watershed` is a flood from labelled markers: neither is an
`image_ops` primitive, so the algorithm stays OpenCV's and only the language
changes.

Both set a mask on each detection and change nothing else about it. Both are
held to `tests/golden/opencv`'s `refine` cases, which record the mask of
every detection rather than a digest of it.

The rectangle arithmetic is the C++'s `bbox_to_mask_rect`: floor the minimum
and ceil the maximum, so a box with fractional edges gets the pixels it
touches. A mask is box-sized and its origin is the box's upper left, not the
image's, which is what `get_standard_mask` is about -- a mask that does not
match its box is corner-aligned and padded rather than resized.
"""

import logging

import numpy as np

from kwiver.vital.algo import RefineDetections
from kwiver.vital.types import DetectedObjectSet, Image, ImageContainer

logger = logging.getLogger(__name__)


def _as_bool(value):
    return str(value).strip().lower() in ("true", "yes", "on", "1")


def _config_bool(value):
    return "true" if value else "false"


def _config_double(value):
    return "%g" % float(value)


class Rect(object):
    """A `cv::Rect`: an origin, a size, and the two set operations on them."""

    def __init__(self, x, y, width, height):
        self.x = int(x)
        self.y = int(y)
        self.width = max(0, int(width))
        self.height = max(0, int(height))

    @property
    def empty(self):
        return self.width <= 0 or self.height <= 0

    def __eq__(self, other):
        return (self.x, self.y, self.width, self.height) == \
               (other.x, other.y, other.width, other.height)

    def intersect(self, other):
        """`operator&`: the overlap, empty at the origin when there is none."""
        x = max(self.x, other.x)
        y = max(self.y, other.y)
        right = min(self.x + self.width, other.x + other.width)
        bottom = min(self.y + self.height, other.y + other.height)

        if right <= x or bottom <= y:
            return Rect(0, 0, 0, 0)

        return Rect(x, y, right - x, bottom - y)

    def union(self, other):
        """`operator|`: the smallest rectangle holding both.

        An empty operand is ignored, which is what OpenCV does and what
        makes a union with a rectangle that fell off the image harmless.
        """
        if self.empty:
            return other
        if other.empty:
            return self

        x = min(self.x, other.x)
        y = min(self.y, other.y)
        right = max(self.x + self.width, other.x + other.width)
        bottom = max(self.y + self.height, other.y + other.height)

        return Rect(x, y, right - x, bottom - y)

    def shifted(self, x, y):
        """`rect - point`: the same rectangle in another origin's frame."""
        return Rect(self.x - x, self.y - y, self.width, self.height)

    def slice(self):
        return (slice(self.y, self.y + self.height),
                slice(self.x, self.x + self.width))


def mask_rect(box):
    """`bbox_to_mask_rect`: the pixels a box touches."""
    min_x = int(np.floor(box.min_x()))
    min_y = int(np.floor(box.min_y()))

    return Rect(min_x, min_y,
                int(np.ceil(box.max_x())) - min_x,
                int(np.ceil(box.max_y())) - min_y)


def scale_about_center(box, factor):
    """`kwiver::vital::scale_about_center`, which both use for their crops."""
    from kwiver.vital.types import BoundingBoxD

    centre_x = (box.min_x() + box.max_x()) / 2.0
    centre_y = (box.min_y() + box.max_y()) / 2.0
    half_width = box.width() * factor / 2.0
    half_height = box.height() * factor / 2.0

    return BoundingBoxD(centre_x - half_width, centre_y - half_height,
                        centre_x + half_width, centre_y + half_height)


def standard_mask(detection):
    """`get_standard_mask`: the detection's mask, at its box's size.

    A mask that is already the right size is used as it is; one that is not
    is corner-aligned into a zeroed buffer and padded or cropped. Not
    resized -- a segmentation mask scaled by interpolation would mean
    something different.
    """
    container = detection.mask

    if container is None:
        return None

    array = container.asarray()

    if array.ndim == 3:
        array = array[:, :, 0]

    array = array.astype(np.uint8)

    rect = mask_rect(detection.bounding_box)

    if array.shape == (rect.height, rect.width):
        return np.ascontiguousarray(array)

    out = np.zeros((rect.height, rect.width), dtype=np.uint8)
    height = min(rect.height, array.shape[0])
    width = min(rect.width, array.shape[1])
    out[:height, :width] = array[:height, :width]

    return out


def _to_bgr(image_container):
    """The BGR array both implementations saw.

    The bridge was asked for a `BGR_COLOR` mat, and unlike most callers
    neither of these ever converts it back -- the output is a mask, not an
    image -- so the swap does not cancel and has to happen here.
    `cv::grabCut` and `cv::watershed` both weigh the channels, so it is not
    a formality.
    """
    array = image_container.asarray()

    if array.ndim == 2:
        array = np.dstack([array] * 3)

    if array.shape[2] >= 3:
        return np.ascontiguousarray(array[:, :, 2::-1])

    return np.ascontiguousarray(np.dstack([array[:, :, 0]] * 3))


def _with_mask(detection, mask):
    """A copy of `detection` carrying `mask`."""
    refined = detection.clone()
    refined.mask = ImageContainer(Image(np.ascontiguousarray(mask)))
    return refined


class RefineDetectionsGrabCut(RefineDetections):
    """Set each detection's mask with `cv2.grabCut`."""

    def __init__(self):
        RefineDetections.__init__(self)

        self._context_scale_factor = 2.0
        self._foreground_scale_factor = 0.0
        self._iter_count = 2
        self._seed_with_existing_masks = True

    def get_configuration(self):
        cfg = super(RefineDetections, self).get_configuration()
        cfg.set_value("context_scale_factor",
                      _config_double(self._context_scale_factor))
        cfg.set_value("foreground_scale_factor",
                      _config_double(self._foreground_scale_factor))
        cfg.set_value("iter_count", str(int(self._iter_count)))
        cfg.set_value("seed_with_existing_masks",
                      _config_bool(self._seed_with_existing_masks))
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)

        self._context_scale_factor = float(
            cfg.get_value("context_scale_factor"))
        self._foreground_scale_factor = float(
            cfg.get_value("foreground_scale_factor"))
        self._iter_count = int(float(cfg.get_value("iter_count")))
        self._seed_with_existing_masks = _as_bool(
            cfg.get_value("seed_with_existing_masks"))

    def check_configuration(self, cfg):
        return True

    def refine(self, image_data, detections):
        import cv2

        if image_data is None or detections is None:
            return detections

        image = _to_bgr(image_data)
        image_rect = Rect(0, 0, image.shape[1], image.shape[0])

        result = DetectedObjectSet()

        for detection in detections:
            box = detection.bounding_box

            context = mask_rect(
                scale_about_center(box, self._context_scale_factor))
            context = context.intersect(image_rect)

            rect = mask_rect(box)
            whole = context.union(rect)

            # Everything starts as background, the detection's own box as
            # probable foreground, and the seed as certain foreground.
            mask = np.full((whole.height, whole.width), cv2.GC_BGD,
                           dtype=np.uint8)

            def local(target):
                return target.shifted(whole.x, whole.y).slice()

            inner = rect.intersect(context)
            if not inner.empty:
                mask[local(inner)] = cv2.GC_PR_FGD

            seed = standard_mask(detection)

            if self._seed_with_existing_masks and seed is not None:
                region = mask[local(rect)]
                region[seed.astype(bool)] = cv2.GC_FGD
            else:
                foreground = mask_rect(
                    scale_about_center(box, self._foreground_scale_factor))
                foreground = foreground.intersect(whole)

                if not foreground.empty:
                    mask[local(foreground)] = cv2.GC_FGD

            # Two cases make no sense to run: a box entirely outside the
            # image, which has no foreground, and a box that contains the
            # context, which has no background.
            runnable = (not rect.intersect(image_rect).empty and
                        rect.intersect(context) != context)

            if runnable:
                crop = mask[local(context)]
                refined, _, _ = cv2.grabCut(
                    np.ascontiguousarray(image[context.slice()]),
                    np.ascontiguousarray(crop), None, None, None,
                    self._iter_count, cv2.GC_INIT_WITH_MASK)
                mask[local(context)] = refined

            out = mask[local(rect)]
            out = np.isin(out, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)

            result.add(_with_mask(detection, out))

        return result


class RefineDetectionsWatershed(RefineDetections):
    """Set each detection's mask with `cv2.watershed`."""

    def __init__(self):
        RefineDetections.__init__(self)

        self._seed_scale_factor = 0.2
        self._uncertain_scale_factor = 1.0
        self._seed_with_existing_masks = True

    def get_configuration(self):
        cfg = super(RefineDetections, self).get_configuration()
        cfg.set_value("seed_scale_factor",
                      _config_double(self._seed_scale_factor))
        cfg.set_value("uncertain_scale_factor",
                      _config_double(self._uncertain_scale_factor))
        cfg.set_value("seed_with_existing_masks",
                      _config_bool(self._seed_with_existing_masks))
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)

        self._seed_scale_factor = float(cfg.get_value("seed_scale_factor"))
        self._uncertain_scale_factor = float(
            cfg.get_value("uncertain_scale_factor"))
        self._seed_with_existing_masks = _as_bool(
            cfg.get_value("seed_with_existing_masks"))

    def check_configuration(self, cfg):
        return True

    def refine(self, image_data, detections):
        import cv2

        if image_data is None or detections is None:
            return detections

        image = _to_bgr(image_data)
        image_rect = Rect(0, 0, image.shape[1], image.shape[0])

        # Everything outside every detection's uncertain region is known
        # background, and gets the last label.
        background = np.full(image.shape[:2], 255, dtype=np.uint8)
        markers = np.zeros(image.shape[:2], dtype=np.int32)

        seeds = []
        entries = list(detections)

        for index, detection in enumerate(entries):
            box = detection.bounding_box
            rect = mask_rect(box)

            uncertain = mask_rect(
                scale_about_center(box, self._uncertain_scale_factor))
            uncertain = uncertain.intersect(image_rect)

            if not uncertain.empty:
                background[uncertain.slice()] = 0

            crop = rect.intersect(image_rect)
            already_set = markers[crop.slice()] != 0

            existing = standard_mask(detection)

            if self._seed_with_existing_masks and existing is not None:
                seed = existing.copy()
            else:
                seed = np.zeros((rect.height, rect.width), dtype=np.uint8)
                inner = mask_rect(
                    scale_about_center(box, self._seed_scale_factor))
                inner = inner.intersect(rect)

                if not inner.empty:
                    seed[inner.shifted(rect.x, rect.y).slice()] = 1

            if not crop.empty:
                local = seed[crop.shifted(rect.x, rect.y).slice()].astype(bool)
                region = markers[crop.slice()]

                # The label, and then -1 wherever two detections claimed the
                # same pixel. `cv::max( markers, 0 )` below turns those back
                # into unknown rather than letting either win.
                region[local] = index + 1
                region[np.logical_and(local, already_set)] = -1

            seeds.append(seed)

        markers = np.maximum(markers, 0)
        markers[background.astype(bool)] = len(entries) + 1

        cv2.watershed(image, markers)

        result = DetectedObjectSet()

        for index, detection in enumerate(entries):
            box = detection.bounding_box
            rect = mask_rect(box)
            crop = rect.intersect(image_rect)
            mask = seeds[index]

            if not crop.empty:
                region = mask[crop.shifted(rect.x, rect.y).slice()]
                region[markers[crop.slice()] == index + 1] = 1

            result.add(_with_mask(detection, mask))

        return result


def __vital_algorithm_register__():
    from viame.core.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        RefineDetectionsGrabCut, "ocv_grabcut",
        "Set detection segmentation masks using cv::grabCut")
    register_vital_algorithm(
        RefineDetectionsWatershed, "ocv_watershed",
        "Set detection segmentation masks using cv::watershed")
