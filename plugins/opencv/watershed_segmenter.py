# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
OpenCV Watershed-based point segmentation algorithm.

This module provides a concrete implementation of SegmentViaPoints using
OpenCV's watershed algorithm for interactive point-based segmentation.
User clicks are used to initialize foreground/background seeds for the
watershed algorithm.
"""

import sys

import cv2
import numpy as np
import scriptconfig as scfg

from kwiver.vital.algo import SegmentViaPoints


class WatershedSegmenterConfig(scfg.DataConfig):
    """Configuration for OpenCV watershed segment via points algorithm."""

    # Seed region parameters
    foreground_radius = scfg.Value(
        5, help='Radius of foreground seed regions around click points (pixels)')
    background_radius = scfg.Value(
        3, help='Radius of background seed regions around click points (pixels)')

    # Morphological operations
    morph_kernel_size = scfg.Value(
        3, help='Kernel size for morphological operations (erosion/dilation)')
    morph_iterations = scfg.Value(
        2, help='Number of iterations for morphological operations')

    # Pre-processing
    blur_kernel_size = scfg.Value(
        5, help='Gaussian blur kernel size for pre-processing (0 to disable)')
    use_gradient = scfg.Value(
        'true', help='Use gradient magnitude to guide watershed boundaries')

    # Post-processing
    min_area = scfg.Value(
        100, help='Minimum area in pixels for output segments')
    fill_holes = scfg.Value(
        'true', help='Fill holes in the output mask')

    # Expansion parameters for uncertain region
    expansion_factor = scfg.Value(
        2.0, help='Factor to expand bounding box for uncertain watershed region')


class WatershedSegmenter(SegmentViaPoints):
    """
    OpenCV Watershed-based point segmentation implementation.

    Uses user-provided foreground and background points to seed the
    watershed algorithm for interactive segmentation.
    """

    def __init__(self):
        SegmentViaPoints.__init__(self)
        self._config = WatershedSegmenterConfig()

    def get_configuration(self):
        cfg = super(SegmentViaPoints, self).get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)

        for key in self._config.keys():
            self._config[key] = str(cfg.get_value(key))

        self._config.__post_init__()

        for key, value in self._config.items():
            setattr(self, "_" + key, value)

        return True

    def check_configuration(self, cfg):
        return True

    def _log(self, message):
        print(f"[WatershedSegmenter] {message}", file=sys.stderr, flush=True)

    def _parse_bool(self, value):
        """Parse a string value as boolean."""
        if isinstance(value, bool):
            return value
        return str(value).lower() in ('true', '1', 'yes', 'on')

    def segment(self, image, points, point_labels):
        """
        Perform point-based segmentation using OpenCV watershed.

        Args:
            image: ImageContainer with the image to segment
            points: Vector of Point2d objects indicating prompt locations
            point_labels: Vector of int labels (1=foreground, 0=background)

        Returns:
            DetectedObjectSet containing segmented objects with masks
        """
        from kwiver.vital.types import DetectedObjectSet, DetectedObject, DetectedObjectType
        try:
            from kwiver.vital.types import BoundingBoxD
        except ImportError:
            from kwiver.vital.types import BoundingBox as BoundingBoxD
        from kwiver.vital.types.types import ImageContainer, Image

        # Convert image to numpy array
        img_array = image.image().asarray()

        # Ensure 3-channel BGR format for OpenCV
        if img_array.ndim == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 1:
            img_array = cv2.cvtColor(img_array.squeeze(), cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif img_array.shape[2] == 3:
            # Assume RGB, convert to BGR for OpenCV
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Ensure uint8
        if img_array.dtype != np.uint8:
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)

        # Ensure contiguous memory layout
        img_array = np.ascontiguousarray(img_array)

        h, w = img_array.shape[:2]

        # Convert points to numpy arrays
        fg_points = []
        bg_points = []
        for p, label in zip(points, point_labels):
            x, y = int(p.value[0]), int(p.value[1])
            if label == 1:  # Foreground
                fg_points.append((x, y))
            else:  # Background
                bg_points.append((x, y))

        if not fg_points:
            # No foreground points, return empty result
            return DetectedObjectSet()

        # Calculate bounding box from foreground points with expansion
        fg_array = np.array(fg_points)
        x_min, y_min = fg_array.min(axis=0)
        x_max, y_max = fg_array.max(axis=0)

        # Add padding based on foreground radius and expansion factor
        expansion = int(self._config.expansion_factor * max(
            self._config.foreground_radius, 50))
        x_min = max(0, x_min - expansion)
        y_min = max(0, y_min - expansion)
        x_max = min(w, x_max + expansion)
        y_max = min(h, y_max + expansion)

        # Create markers image (same size as input)
        markers = np.zeros((h, w), dtype=np.int32)

        # Draw foreground markers (label 1)
        for (x, y) in fg_points:
            cv2.circle(markers, (x, y), self._config.foreground_radius, 1, -1)

        # Draw background markers (label 2)
        for (x, y) in bg_points:
            cv2.circle(markers, (x, y), self._config.background_radius, 2, -1)

        # If no background points provided, create automatic background markers
        # at the edges of the expanded bounding box
        if not bg_points:
            # Mark border as background
            border_width = 3
            markers[y_min:y_min+border_width, x_min:x_max] = 2  # Top
            markers[y_max-border_width:y_max, x_min:x_max] = 2  # Bottom
            markers[y_min:y_max, x_min:x_min+border_width] = 2  # Left
            markers[y_min:y_max, x_max-border_width:x_max] = 2  # Right

        # Pre-process image
        img_processed = img_array.copy()
        blur_size = self._config.blur_kernel_size
        if blur_size > 0:
            if blur_size % 2 == 0:
                blur_size += 1
            img_processed = cv2.GaussianBlur(img_processed, (blur_size, blur_size), 0)

        # Apply watershed
        cv2.watershed(img_processed, markers)

        # Create binary mask from watershed result
        # Watershed labels: -1 = boundary, 1 = foreground, 2 = background
        mask = (markers == 1).astype(np.uint8)

        # Post-process mask
        kernel = np.ones((self._config.morph_kernel_size,
                         self._config.morph_kernel_size), np.uint8)

        # Close small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                                iterations=self._config.morph_iterations)

        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                                iterations=1)

        # Fill holes if requested
        if self._parse_bool(self._config.fill_holes):
            # Find contours and fill
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask, contours, -1, 1, -1)

        # Filter by minimum area
        if self._config.min_area > 0:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            mask_filtered = np.zeros_like(mask)
            for cnt in contours:
                if cv2.contourArea(cnt) >= self._config.min_area:
                    cv2.drawContours(mask_filtered, [cnt], -1, 1, -1)
            mask = mask_filtered

        # Create detected object set
        detected_objects = DetectedObjectSet()

        # Get bounding box from mask
        ys, xs = np.where(mask)
        if len(xs) > 0 and len(ys) > 0:
            x1, y1 = float(xs.min()), float(ys.min())
            x2, y2 = float(xs.max()), float(ys.max())

            bbox = BoundingBoxD(x1, y1, x2, y2)

            # Calculate score based on mask quality
            # Use ratio of mask area to bounding box area as a proxy
            mask_area = np.sum(mask)
            bbox_area = (x2 - x1 + 1) * (y2 - y1 + 1)
            score = min(1.0, mask_area / bbox_area) if bbox_area > 0 else 0.5

            dot = DetectedObjectType("object", score)
            detected_obj = DetectedObject(bbox, score, dot)

            # Create mask image (crop to bounding box)
            mask_crop = mask[int(y1):int(y2)+1, int(x1):int(x2)+1].astype(np.uint8)
            detected_obj.mask = ImageContainer(Image(mask_crop))

            detected_objects.add(detected_obj)

        return detected_objects


def __vital_algorithm_register__():
    """Register the OpenCV watershed segment via points algorithm with KWIVER."""
    from kwiver.vital.algo import algorithm_factory

    implementation_name = "ocv_watershed"
    if algorithm_factory.has_algorithm_impl_name(
            WatershedSegmenter.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(
        implementation_name,
        "OpenCV watershed-based point segmentation algorithm",
        WatershedSegmenter)

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
