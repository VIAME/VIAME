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
        # The WLS filter, which the shipped measurement config turns on and
        # which nothing here covered. It is where finding 1.21 lives: the
        # filter fills the invalid regions, and `raw` writes the result back
        # as sixteenths in an int16, where those fills saturate at 32767.
        #
        # **How these three were recorded.** Not by `record.py` against the
        # C++ -- P7-T06 had already replaced `compute_stereo_disparity.cxx`
        # by the time the gap was found. Instead the python implementation
        # was run against the **reference build of `main` at 8edfd2f66**,
        # which still has the C++, on the same inputs: all three are bit
        # identical, zero difference over the whole map. That is the same
        # contract a recording gives, obtained the only way left. Said out
        # loud here because it is the one exception in this framework.
        ("wls", {"use_wls_filter": "true", "wls_lambda": "8000.0",
                 "wls_sigma": "1.5"}),
        ("wls_float32", {"use_wls_filter": "true", "wls_lambda": "8000.0",
                         "wls_sigma": "1.5", "output_format": "float32"}),
        ("wls_bm", {"algorithm": "BM", "use_wls_filter": "true",
                    "wls_lambda": "8000.0", "wls_sigma": "1.5"}),
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


# ----------------------------------------------------------------------------
# Stereo calibration, end to end
# ----------------------------------------------------------------------------
#
# The shipped calibration pipeline over the synthetic views in
# `measurement_fixtures.py`: the target detector, the track accumulation, the
# calibration itself and the YAML writer, all of it. This is the case that
# holds P7-T06's port together, because none of those four can be replaced
# without the others noticing.
#
# It has something no other case in this framework has: **the right answer**.
# The views are rendered through a rig whose parameters are known exactly, so
# a failure can say not only "this differs from the recording" but "and the
# recording was correct". `CALIBRATION_TOLERANCES` is what the recorded
# implementation achieved, with room -- OpenCV recovered the focal lengths to
# about 0.2% and the baseline to 0.006%.
CALIBRATION_PIPELINE = "stereo_calibrate_cameras_default.pipe"

# The board is 30 mm, and the shipped config says 80; a calibration scales
# its translation by whatever it is told, so the fixture's own value has to
# be passed or the baseline comes out 80/30 too large.
# Per detector rather than `global:square_size`, and that is not a style
# choice: `-s global:square_size=30` does **not** reach
# `$CONFIG{global:square_size}`. The override is appended as a second
# `config global` block after the substitutions have been resolved against
# the first, so the calibration silently uses the shipped 80 and comes out
# with its baseline scaled by 80/30. `design/lite-findings.md` records it.
#
# The grid is given explicitly for the same class of reason: with
# `auto_detect_grid` on, OpenCV's auto detection matched a 6 by 5 sub-grid
# inside the 8 by 5 board on one camera's first view and a correct 8 by 5 on
# the other, and the calibration then refused a pair whose corner counts
# disagreed. A real calibration setup states its board.
def _detector_settings():
    return tuple(
        setting
        for detector in ("detector1", "detector2")
        for setting in (
        "{}:detector:ocv_detect_calibration_targets:"
        "auto_detect_grid=false".format(detector),
        "{}:detector:ocv_detect_calibration_targets:"
        "target_width=8".format(detector),
        "{}:detector:ocv_detect_calibration_targets:"
        "target_height=5".format(detector),
            "{}:detector:ocv_detect_calibration_targets:"
            "square_size=30".format(detector),
        ))


# The variants. `all_frames` is the shipped configuration, which has
# `frame_count_threshold` at 0 and so uses every view. `frames_6` turns the
# k-medians frame selection on, and it is here because that is the only way
# to reach `filter_stereo_feature_tracks::select_points_maximizing_variance`
# at all: at the shipped default it short-circuits, so a port of it would
# otherwise be code no test runs. It is deterministic across runs, checked.
CALIBRATION_VARIANTS = (
    ("all_frames", ()),
    ("frames_6", ("cameras_calibration:frame_count_threshold=6",)),
)


def calibration_settings(variant):
    """Every `-s` a calibration variant needs."""
    for name, extra in CALIBRATION_VARIANTS:
        if name == variant:
            return _detector_settings() + extra

    raise KeyError(variant)


CALIBRATION_SETTINGS = _detector_settings()

# Ground truth, from `measurement_fixtures.py`. Repeated here rather than
# imported so that a case file reads as the specification it is.
CALIBRATION_TRUTH = {
    "fx_left": 600.0, "fy_left": 600.0,
    "cx_left": 319.5, "cy_left": 239.5,
    "fx_right": 610.0, "fy_right": 610.0,
    "cx_right": 319.5, "cy_right": 239.5,
    "baseline": 120.0,
}

# Relative tolerances against the truth above, not against the recording.
# The recorded implementation came within 0.3% on every one of them, so
# these have a factor of three in hand; they are here to catch a port that
# is wrong, not to measure one that is right.
# Two per cent on a focal length rather than one: selecting six of the twelve
# views costs accuracy, as it should -- `all_frames` lands within 0.3% and
# `frames_6` within 0.9%. A port that is broken will be far worse than two
# per cent, so this still catches what it is for.
CALIBRATION_TOLERANCES = {
    "focal": 0.02,
    "centre": 0.005,
    "baseline": 0.01,
}

# And a distortion-free rig, so a calibration that invents distortion is
# telling on itself. The recorded one returns exact zeros.
CALIBRATION_MAX_DISTORTION = 1e-6


def calibration_view_names():
    """The fixture names of the left and right calibration views, in order."""
    import measurement_fixtures

    count = measurement_fixtures.CALIBRATION_VIEWS

    return (["calib_left_%02d" % index for index in range(count)],
            ["calib_right_%02d" % index for index in range(count)])


# ----------------------------------------------------------------------------
# Single camera calibration, end to end
# ----------------------------------------------------------------------------
#
# `ocv_calibrate_single_camera` is the other half of the calibration chain and
# the one a user reaches through DIVE's "calibrate camera" button. It has the
# same right answer available to it as the stereo case -- the left views of
# the same synthetic rig -- so the same two checks apply: against the
# recording, and against the rig.
#
# `square_size` is set per process for the reason the stereo case states:
# `-s global:square_size=30` does not reach `$CONFIG{global:square_size}`.
MONO_CALIBRATION_PIPELINE = "utility_calibrate_single_camera.pipe"

MONO_CALIBRATION_VARIANTS = (
    ("all_frames", ("camera_calibration:frame_count_threshold=0",)),
    # The frame selection, as in the stereo case
    ("frames_6", ("camera_calibration:frame_count_threshold=6",)),
)


def mono_calibration_settings(variant):
    """Every `-s` a single camera calibration variant needs."""
    detector = tuple(
        "detector:detector:ocv_detect_calibration_targets:" + setting
        for setting in ("auto_detect_grid=false", "target_width=8",
                        "target_height=5", "square_size=30"))

    for name, extra in MONO_CALIBRATION_VARIANTS:
        if name == variant:
            return detector + ("camera_calibration:square_size=30",) + extra

    raise KeyError(variant)


# The left half of `CALIBRATION_TRUTH`, and the same tolerances.
MONO_CALIBRATION_TRUTH = {
    "fx": 600.0, "fy": 600.0, "cx": 319.5, "cy": 239.5,
}


def mono_calibration_view_names():
    """The fixture names of the left calibration views, in order."""
    return calibration_view_names()[0]


# ----------------------------------------------------------------------------
# Measurement from annotations, end to end
# ----------------------------------------------------------------------------
#
# `compute_measurements` over `measurement_fixtures.measurement_scene()`: a
# textured plane at a known depth through the rig the calibration fixture
# defines, with five segments of known length drawn on it. The track file the
# pipeline reads is generated from the exact left projections, so what the
# recording measures is the matching, the triangulation and the aggregation
# rather than an annotator's aim.
#
# Like the calibration case it has a **right answer**: the lengths are 300 to
# 400 mm and known to the millimetre. That is what caught finding 1.20, where
# every one of them came back zero.
MEASUREMENT_PIPELINE = "stereo_measure_current_annots_default.pipe"

# The variants, one per matching method that is deterministic. Each reaches a
# different part of the OpenCV surface P7-T06 has to replace:
#
# * `input_pairs_only` takes the right keypoints from the track file, so it
#   is triangulation and nothing else -- and with the true right points in
#   the file it should land on the truth to a fraction of a millimetre.
# * `epipolar_template_matching` samples the epipolar curve in the
#   **unrectified** right image: `projectPoints` and `undistortPoints`, plus
#   NCC on a strip. It is the shipped default.
# * `template_matching` rectifies both images first: `stereoRectify`,
#   `initUndistortRectifyMap` and `remap` as well.
# * `compute_disparity` runs SGBM over the rectified pair and looks the
#   disparity up at each keypoint. It is recorded **broken**: it measures
#   three of the five targets at a tenth of their length and puts their right
#   keypoints hundreds of pixels off the left edge of the image. The cause is
#   upstream and reproduces on the reference build of `main`: the shipped
#   config turns the WLS filter on, and `ocv_stereo_disparity` writes its
#   `raw` output back as sixteenths in an **int16**, where the filter's
#   fill-in values saturate at 32767 -- 2047 pixels of disparity on a rig
#   whose real disparity is 45. `find_corresponding_point_external_disparity`
#   rejects only values at or below zero, so it takes them. Finding 1.21.
# * `depth_projection` uses no image content at all -- it puts the right
#   point where `default_depth` says it would be -- so it is pure projection
#   geometry, and its answers are wrong by the ratio of that default to the
#   real depth. Recorded because a port has to reproduce the projection, not
#   because the numbers are good.
#
# `feature_descriptor` and `ransac_feature` are left out: they go through
# `ocv_flann_based`, which seeds its KD-trees from the clock and returns a
# different number of matches from one run to the next.
MEASUREMENT_VARIANTS = (
    # `detection_pairing_method` as well, and it has to be there: the
    # process deliberately does **not** pair left and right by track id --
    # two independent trackers can reuse one -- so with no pairing method
    # every track is left-only and `input_pairs_only` has nothing to take.
    ("input_pairs_only", ("measurer:matching_methods=input_pairs_only",
                          "measurer:detection_pairing_method="
                          "keypoint_projection")),
    ("epipolar_template_matching",
     ("measurer:matching_methods=epipolar_template_matching",)),
    ("template_matching", ("measurer:matching_methods=template_matching",)),
    ("compute_disparity", ("measurer:matching_methods=compute_disparity",)),
    ("depth_projection", ("measurer:matching_methods=depth_projection",)),
)

# Which variants are given the true right keypoints in the second track file.
# Only the one that consumes them; for the rest the file is empty, so the
# method under test is the only thing that can produce a match.
MEASUREMENT_PAIRED_VARIANTS = ("input_pairs_only",)


# How far a recorded value may move, per variant, as a fraction.
#
# Zero for every method that does not rectify: `input_pairs_only`,
# `epipolar_template_matching` and `depth_projection` reproduce the recording
# **exactly** after the port. The two that rectify cannot, and the reason is
# stated at `library/measurement/projection.cxx`: `cv::stereoRectify` samples
# the image border through `CV_32FC2` points, so the inscribed rectangle it
# scales the rectified focal length to fit carries about seven significant
# digits, and this works in double. That moves the rectified focal length by
# a part in ten million, the rectified images by a ten-thousandth of a pixel,
# and the measured lengths by the amounts here -- 1.4e-6 for
# `template_matching`, and 1.2e-3 for `compute_disparity`, whose disparity
# lookup is an integer index into a map and so quantises the difference up.
# `compute_disparity` is not in this table at all; it is **unstable**, below,
# and the reason is finding 1.21 rather than the port.
MEASUREMENT_ARRAY_TOLERANCE = {
    "template_matching": 1e-5,
}


# The pairing's own, and the same cause: every one of its variants runs
# through `stereo_rectify`, so its 3D positions carry the float-versus-double
# difference of about a part in ten million. 1.5e-5 mm at a reported depth of
# 137, measured.
PAIR_STEREO_ARRAY_TOLERANCE = 1e-6


def array_tolerance(kind, impl, variant):
    """The relative tolerance a case's recorded arrays are held to."""
    if kind == "pair_stereo":
        return PAIR_STEREO_ARRAY_TOLERANCE

    return MEASUREMENT_ARRAY_TOLERANCE.get(variant, 0.0)


def measurement_settings(variant):
    for name, settings in MEASUREMENT_VARIANTS:
        if name == variant:
            return settings

    raise KeyError(variant)


# How far a measured length may be from the true one, per variant, as a
# fraction. These are contracts on the **method**, not on the port -- the
# recording is what holds the port -- so they are set where a method that has
# stopped working fails and a method that is working does not.
#
# `input_pairs_only` is given the exact right points, so half a per cent is
# all the triangulation's own first order correction costs -- it lands within
# 0.12%.
# The matchers land within a couple of per cent. `depth_projection` is not
# checked against truth at all: it is told the wrong depth by construction.
MEASUREMENT_LENGTH_TOLERANCE = {
    "input_pairs_only": 0.005,
    "epipolar_template_matching": 0.10,
    "template_matching": 0.05,
    "compute_disparity": None,
    "depth_projection": None,
}


# ----------------------------------------------------------------------------
# Stereo detection pairing
# ----------------------------------------------------------------------------
#
# `ocv_pair_stereo_detections` over the same synthetic scene: a real
# disparity map from `ocv_stereo_disparity`, rectified through the same
# calibration the pairing reads, and boxes around the five segments in each
# camera. The four pipelines that select the process are add-ons whose
# detector needs a downloaded model, so the detections are read from a file.
#
# The disparity has to be real rather than a stand-in, because `PAIRING_3D`
# is a lookup into it: the process reprojects the left detection's box
# through `Q` and compares the result to the right detection's.
#
# **It is recorded wrong, and the scene is what proves it.** Every
# `stereo3d_z` comes out at about 137 where the plane is at 2200 -- a factor
# of sixteen, which is SGBM's fixed point. `cv::reprojectImageTo3D` documents
# that a 16-bit signed disparity is taken to have no fractional bits, and
# nothing divides by sixteen on the way in. Finding 1.10.
PAIR_STEREO_PIPELINE = "pair_stereo_detections.pipe"

PAIR_STEREO_VARIANTS = (
    ("pairing_3d", ()),
    # The other method: rectify both boxes and compare their overlap, which
    # never touches the disparity values and so is not affected by the
    # defect above.
    ("rectified_iou", ("stereo_pairing:pairing_method="
                       "PAIRING_RECTIFIED_IOU",)),
    # A threshold high enough to reject everything, which is the path that
    # decides a detection has no partner.
    ("strict_iou", ("stereo_pairing:pairing_method=PAIRING_RECTIFIED_IOU",
                    "stereo_pairing:iou_pair_threshold=0.99")),
)


def pair_stereo_settings(variant):
    for name, settings in PAIR_STEREO_VARIANTS:
        if name == variant:
            return settings

    raise KeyError(variant)


# The truth the scene gives: the plane sits at this depth, so a `stereo3d_z`
# should be about this and is about a sixteenth of it. Checked as a ratio
# rather than a value, so the day someone fixes the scaling this says so
# loudly instead of drifting.
PAIR_STEREO_TRUE_DEPTH_MM = 2200.0
PAIR_STEREO_DEPTH_RATIO = 16.0
PAIR_STEREO_DEPTH_RATIO_TOLERANCE = 0.10


# ----------------------------------------------------------------------------
# What cannot be recorded exactly
# ----------------------------------------------------------------------------
#
# `kmedians` seeds itself with `cv::kmeans`, which draws from `cv::theRNG()`
# -- a thread-local generator whose state depends on what else has used
# OpenCV on that thread, and not on anything the algorithm was given. Each
# implementation is deterministic within itself (three runs each, identical
# to nine digits) and they disagree with each other: from the same twelve
# frames the C++ selected {0, 1, 4, 6, 8, 11} and the python port
# {0, 1, 4, 6, 9, 11}, one frame apart. Seeding the generator by hand across
# thirty values gives those two selections and no others.
#
# So a matrix from this case is not a contract on its bytes. It is still a
# contract on two things worth more: that the calibration lands within
# `RELATIVE_TOLERANCE` of the recorded one -- a different sixth frame costs
# 0.22%, and anything actually broken costs far more -- and that it still
# recovers the ground truth, which `check_calibration_truth` checks and
# which does not care how the frames were chosen.
UNSTABLE = {
    # `compute_disparity` reads its depth out of the **saturated** disparity
    # map of finding 1.21: the WLS filter's fill values reach 32767, which is
    # 2047 pixels of disparity on a rig whose real disparity is 45, and
    # neighbouring pixels of such a map hold wildly different numbers. So a
    # rectification that differs in its seventh significant digit -- which is
    # exactly what `library/measurement/projection` differs by, since OpenCV
    # samples its image border in float and this works in double -- lands the
    # lookup on a different pixel and moves a right keypoint by ten.
    #
    # There is no tolerance that makes that a byte contract, and pretending
    # otherwise would be recording noise. What is still contractual is below
    # in `compare_unstable`: the same tracks get a measurement, and their
    # lengths are still an order of magnitude short of the truth, which is
    # the defect this case exists to pin. A method that started working would
    # fail this -- and should, because it would be a change worth noticing.
    ("measurement", MEASUREMENT_PIPELINE, "compute_disparity"):
        "the disparity map it reads is saturated (finding 1.21), so a "
        "rectification differing in the seventh digit moves a keypoint by "
        "ten pixels",
    ("calibration_pipeline", CALIBRATION_PIPELINE, "frames_6"):
        "kmedians seeds from cv::theRNG(), whose state is not an input to "
        "the algorithm, so which six of the twelve frames are selected "
        "differs between implementations",
}

RELATIVE_TOLERANCE = 0.01


def unstable(kind, impl, variant=None):
    """Why this case cannot be compared exactly, or None."""
    return UNSTABLE.get((kind, impl, variant))


# How far an unstable `measurement` member may move. `track_ids` is exact --
# the same tracks have to get a measurement -- and `length` keeps a wide band
# that still separates "the broken method, reproduced" from "the method
# started working", which would be ten times larger. The keypoints and the
# midpoints are checked for shape alone, since those are the values the
# saturated map moves around.
MEASUREMENT_UNSTABLE_TOLERANCE = {
    "track_ids": 0.0,
    "length": 0.3,
}


def compare_unstable(kind, member, got, want):
    """Problems with an unstable case's member, as a list of strings.

    Relative to the member's own scale, since a rotation matrix and a
    projection matrix in the same document are four orders of magnitude
    apart and one absolute tolerance cannot serve both.
    """
    import numpy as np

    if got.shape != want.shape:
        return ["shape {} != recorded {}".format(got.shape, want.shape)]

    if want.size == 0:
        return []

    if kind == "measurement":
        allowed = MEASUREMENT_UNSTABLE_TOLERANCE.get(member)

        if allowed is None:
            return []

        scale = np.maximum(1.0, np.abs(want))
        worst = float((np.abs(got - want) / scale).max())

        if worst > allowed:
            return ["{:.2%} apart at most, more than {:.0%}".format(
                worst, allowed)]

        return []

    scale = max(1e-9, float(np.abs(want).max()))
    relative = float(np.abs(got - want).max()) / scale

    if relative > RELATIVE_TOLERANCE:
        return ["{:.2%} apart at most, more than {:.0%}".format(
            relative, RELATIVE_TOLERANCE)]

    return []


def unstable_reason(impl, variant, name=None):
    """The per-input hook the other case files have; nothing uses it here."""
    return None


def divergence_reason(impl, variant, input_name=None):
    """Nothing here diverges from the recording yet."""
    return None
