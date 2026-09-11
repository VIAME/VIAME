"""What the measurement golden recording covers.

The OpenCV-backed pieces of the stereo measurement chain that can be driven
on their own: the disparity matcher and the calibration target detector.
P7-T06 replaces both, and neither had a recording -- P7-T01 covered the
filters and the detectors and P7-T05 the calibration files, and this is the
gap between them.

What it does not cover, and why, since a reader should not have to guess:

* `calibrate_stereo_cameras`, `ocv_optimize_stereo_cameras` and the
  `pair_stereo_*` processes need a sequence of calibration images or a
  feature track set, not one image each. Recording them means committing a
  fixture set of a size this repository does not otherwise carry.
* the `measurement_*` pipelines the task text names need a GPU detector and,
  for the annotation ones, input track files that are not in the tree. The
  one that runs today, `measure_via_default_fish`, writes zero in its length
  column, so there is nothing in its output to hold a port to.

Both gaps are P7-T06's to close before it ports anything, and
`design/lite-findings.md` says so.
"""

# The stereo pair and the two target boards, from `measurement_fixtures.py`.
STEREO = ("stereo_left", "stereo_right")
TARGETS = ("chessboard", "dot_grid")


# `compute_stereo_depth_map:ocv_stereo_disparity`, over the variants that
# change what it computes rather than how it is packaged.
#
# No `calibration_file`, so it runs unrectified: the fixture is already
# rectified by construction, and giving it a calibration would be recording
# the rectification as well as the matching.
DISPARITY = {
    "ocv_stereo_disparity": [
        ("defaults", {}),
        # Both algorithms, because they are different matchers rather than
        # settings of one: BM is a local block correlator and SGBM is a
        # semi-global one, and the shipped default is SGBM.
        ("bm", {"algorithm": "BM"}),
        ("block_size_3", {"block_size": "3"}),
        ("disparities_64", {"num_disparities": "64"}),
        # A non-zero minimum, which shifts the whole search window and is
        # the setting most easily got wrong by one.
        ("min_disparity_16", {"min_disparity": "16",
                              "num_disparities": "64"}),
        # The two packaged output formats beside the raw one. `float32` is
        # the disparity in pixels and `uint16_scaled` is that times
        # `uint16_scale_factor`, which is where a scale factor gets lost.
        ("float32", {"output_format": "float32"}),
        ("uint16_scaled", {"output_format": "uint16_scaled"}),
        # Speckle filtering, which removes small inconsistent regions and is
        # what makes a disparity map usable on real data.
        ("no_speckle_filter", {"speckle_window_size": "0"}),
    ],
}


# `image_object_detector:ocv_detect_calibration_targets`, which finds a
# calibration board's corners and returns one detection per corner.
CALIBRATION_TARGETS = {
    "ocv_detect_calibration_targets": [
        ("defaults", {}),
        # `checkerboard`, not `chessboard`: that is the string the
        # implementation compares against, and anything else turns the
        # checkerboard path off without saying so -- which is what the
        # `unknown_target_type` case records.
        ("checkerboard", {"target_type": "checkerboard"}),
        ("dots", {"target_type": "dots"}),
        ("unknown_target_type", {"target_type": "chessboard"}),
        # A grid the board does not have: the detector must find nothing
        # rather than return a partial one, and a recording of "nothing" is
        # what says so.
        ("wrong_grid", {"target_width": "9", "target_height": "7"}),
        # A region of interest that cuts the board in half, which is how the
        # shipped configs restrict a target to one side of a stereo frame.
        ("roi_left_half", {"roi_x1": "0", "roi_y1": "0",
                           "roi_x2": "200", "roi_y2": "320"}),
    ],
}


def unstable_reason(impl, variant, name=None):
    """Nothing here is unstable; the signature matches the other case files."""
    return None


def divergence_reason(impl, variant, input_name=None):
    """Nothing here diverges from the recording yet."""
    return None
