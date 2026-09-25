# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Finding round blobs, which is what a dot calibration target is made of.

What `cv::SimpleBlobDetector` did, by the same route. **Blobs are lighter
than their surroundings**, because that is the side of the threshold OpenCV
traces; a caller looking for dark blobs inverts the image first, which is
what the dot target detector does and why it does it.

The idea is repetition:
threshold the image at a ladder of levels, trace the shapes at each, keep the
ones that look like blobs, and then believe only the ones that appear at
several levels in the same place. A speck of noise survives one threshold; a
dot survives most of them.

"Looks like a blob" is four independent tests, and the detector applies all
four because each catches something the others miss:

* **area** -- between the bounds asked for, which is the only one with units
  and the only one a caller normally sets;
* **circularity**, `4 pi A / P^2` -- one for a circle, less for anything with
  a longer boundary. This is what rejects a ragged edge;
* **inertia**, the ratio of the smaller second moment to the larger -- one
  for a disc, near zero for a line. This is what rejects a streak, which can
  be perfectly smooth and so perfectly circular by the test above;
* **convexity**, area over the area of the convex hull -- one for a convex
  shape. This is what rejects a crescent or a pair of touching dots, both of
  which can pass all three above.

The details are OpenCV's, because the calibration's dot positions follow from
them. The two worth naming here, because they are the ones that look like
bugs:

* every bound is **exclusive at the top and inclusive at the bottom**: a blob
  of exactly `max_area` is rejected, one of exactly `min_area` is kept.
* a blob's radius is the median distance from its centre to its boundary, and
  the *median of an even count is the mean of the middle two* -- which then
  decides whether two blobs at different thresholds are the same blob.
"""

import numpy as np

from viame import image_kernels


# The threshold ladder, as `cv::SimpleBlobDetector::Params` defaults it.
MIN_THRESHOLD = 50.0
MAX_THRESHOLD = 220.0
THRESHOLD_STEP = 10.0

# How many levels a blob must appear at to be believed.
MIN_REPEATABILITY = 2

# Two blobs nearer than this are the same blob, unless one's radius says
# otherwise. OpenCV's default, in pixels.
MIN_DISTANCE = 10.0


def _grouped(groups, centre, radius, minimum_distance):
    """Add a blob to the group it belongs to, or start a new one.

    OpenCV compares against the **median** member of each group rather than
    its mean or its most recent, and calls it the same blob when it is nearer
    than all three of the minimum distance, that member's radius and this
    blob's own. Keeping each group sorted by radius is what makes the median
    cheap, and is why the insert below walks backwards rather than appending.
    """
    for group in groups:
        middle = group[len(group) // 2]

        gap = float(np.hypot(centre[0] - middle[0][0],
                             centre[1] - middle[0][1]))

        if gap >= minimum_distance and gap >= middle[1] and gap >= radius:
            continue

        group.append((centre, radius))

        at = len(group) - 1
        while at > 0 and radius < group[at - 1][1]:
            group[at] = group[at - 1]
            at -= 1
        group[at] = (centre, radius)

        return

    groups.append([(centre, radius)])


def _inertia_ratio(moments):
    """The ratio of the smaller principal second moment to the larger.

    One for a disc and near zero for a line. A shape with no orientation --
    the denominator below vanishing -- is treated as perfectly round, which
    is what OpenCV does and is the right answer for a circle.
    """
    if moments["m00"] == 0.0:
        return 0.0

    centre_x = moments["m10"] / moments["m00"]
    centre_y = moments["m01"] / moments["m00"]

    mu20 = moments["m20"] / moments["m00"] - centre_x * centre_x
    mu02 = moments["m02"] / moments["m00"] - centre_y * centre_y
    mu11 = moments["m11"] / moments["m00"] - centre_x * centre_y

    denominator = np.sqrt((2.0 * mu11) ** 2 + (mu20 - mu02) ** 2)

    # OpenCV's epsilon, and not a rounding guard: below it the shape has no
    # principal axis to speak of, so there is no ratio to take.
    if denominator <= 1e-2:
        return 1.0

    cosine = np.sqrt((1.0 + (mu20 - mu02) / denominator) / 2.0)
    sine = np.sqrt((1.0 - (mu20 - mu02) / denominator) / 2.0)

    smaller = (mu20 * cosine * cosine + mu11 * 2.0 * cosine * sine +
               mu02 * sine * sine)
    larger = (mu20 * sine * sine - mu11 * 2.0 * cosine * sine +
              mu02 * cosine * cosine)

    if larger == 0.0:
        return 0.0

    ratio = smaller / larger

    return ratio if ratio <= 1.0 else 1.0 / ratio


def _radius(contour, centre):
    """The median distance from the centre to the boundary.

    The median of an even count is the **mean of the middle two**, which is
    the textbook definition and not what several of the medians elsewhere in
    this tree use -- `filter_target_cluster` takes the upper of the two
    because the C++ it reproduces does. This one follows OpenCV.

    The contour's repeated closing point is dropped first: our traced
    contours close and OpenCV's do not, and one duplicated point shifts an
    even-length median onto a different pair.
    """
    points = np.asarray(contour, dtype=np.float64)

    if len(points) > 1 and np.array_equal(points[0], points[-1]):
        points = points[:-1]

    distances = np.sort(np.hypot(points[:, 0] - centre[0],
                                 points[:, 1] - centre[1]))

    return float((distances[(len(distances) - 1) // 2] +
                  distances[len(distances) // 2]) / 2.0)


def _blobs_at(mask, min_area, max_area, min_circularity, min_inertia,
              min_convexity):
    """The blobs of one thresholded image, as (centre, radius)."""
    found = []

    # Every border, holes included, which is what `RETR_LIST` gives OpenCV.
    # The holes are not an edge case here: a dark shape inside a lighter
    # region is found *as* the hole in that region and as nothing else, so
    # tracing only the outer borders misses every dark blob on a light
    # ground -- twelve of them on the chessboard the golden records.
    for contour, _is_hole in image_kernels.find_borders(mask):
        if len(contour) < 3:
            continue

        moments = image_kernels.moments(contour)
        area = moments["m00"]

        # Signed by the winding, and the sign is not the question being
        # asked; `contour_area` is the unsigned one but the moments are
        # needed anyway for the centre and the inertia.
        area = abs(area)

        if area <= 0.0:
            continue

        if min_area is not None and not min_area <= area < max_area:
            continue

        if min_circularity is not None:
            perimeter = image_kernels.arc_length(contour, True)

            if perimeter <= 0.0:
                continue

            circularity = 4.0 * np.pi * area / (perimeter * perimeter)

            if circularity < min_circularity:
                continue

        if min_inertia is not None and _inertia_ratio(moments) < min_inertia:
            continue

        if min_convexity is not None:
            hull = image_kernels.convex_hull(
                np.ascontiguousarray(contour, dtype=np.float64))

            if len(hull) < 3:
                continue

            hull_area = image_kernels.contour_area(hull)

            if hull_area <= 0.0:
                continue

            if area / hull_area < min_convexity:
                continue

        centre = (moments["m10"] / moments["m00"],
                  moments["m01"] / moments["m00"])

        found.append((centre, _radius(contour, centre)))

    return found


def detect_blobs(gray, min_area=25.0, max_area=5000.0,
                 min_circularity=0.8, min_inertia=0.1, min_convexity=0.95,
                 min_threshold=MIN_THRESHOLD, max_threshold=MAX_THRESHOLD,
                 threshold_step=THRESHOLD_STEP,
                 min_repeatability=MIN_REPEATABILITY,
                 min_distance=MIN_DISTANCE):
    """The dark blobs of a single plane image, as an N by 2 array of centres.

    `cv2.SimpleBlobDetector_create(params).detect(image)`, returning the
    keypoint positions rather than keypoints -- which is all either caller in
    this tree reads off them.

    Passing `None` for any of the four shape bounds turns that filter off, as
    clearing the matching `filterBy...` flag did.

    @param gray the image, **light** blobs on a darker ground
    @param min_area the smallest blob kept, in pixels; `max_area` is
           **exclusive**, as OpenCV's is
    @param min_circularity `4 pi A / P^2`, one for a circle
    @param min_inertia the ratio of principal second moments, one for a disc
    @param min_convexity area over convex hull area, one for a convex shape
    @param min_repeatability how many thresholds a blob must survive

    Returns `(centres, diameters)` -- an N by 2 array and an N array -- in
    the order the blobs were first seen. The diameter is twice the **median
    member's** radius, which is the size OpenCV puts on the keypoint: not the
    mean of the group and not the radius at any particular threshold.
    """
    gray = np.ascontiguousarray(gray)

    if gray.ndim != 2:
        raise ValueError("detect_blobs wants a single plane image")

    if threshold_step <= 0.0:
        raise ValueError("detect_blobs wants a positive threshold step")

    groups = []

    level = min_threshold
    while level < max_threshold:
        # **Brighter** than the level, which is what
        # `threshold( ..., THRESH_BINARY )` marks and what OpenCV then traces.
        # So a blob here is a light shape on a darker ground, and a caller
        # with dark blobs inverts first -- which is exactly why
        # `detect_dots` inverts, and the one thing to get right when reading
        # this against `cv::SimpleBlobDetector`.
        mask = (gray > level).astype(np.uint8)

        for centre, radius in _blobs_at(mask, min_area, max_area,
                                        min_circularity, min_inertia,
                                        min_convexity):
            _grouped(groups, centre, radius, min_distance)

        level += threshold_step

    centres = []
    diameters = []

    for group in groups:
        if len(group) < min_repeatability:
            continue

        # The plain mean of the group's positions. OpenCV weights by a
        # per-blob confidence and then sets that confidence to one for every
        # blob it finds, so the weighting has never done anything.
        centres.append((sum(centre[0] for centre, _ in group) / len(group),
                        sum(centre[1] for centre, _ in group) / len(group)))

        # The group is kept sorted by radius, so this is its median.
        diameters.append(group[len(group) // 2][1] * 2.0)

    return (np.array(centres, dtype=np.float64).reshape(-1, 2),
            np.array(diameters, dtype=np.float64))
