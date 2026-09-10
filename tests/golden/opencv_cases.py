"""What the OpenCV golden recording covers.

Phase 7 replaces every implementation here with `image_ops` code registered
under the same name, so what has to be pinned is what each one currently
computes. The config variants are the ones the installed pipelines actually
ask for -- read out of every `.pipe` and `.conf` under the install -- plus
each implementation's own defaults, and, where the shipped configs leave a
path unexercised, one variant that reaches it.

The image_io cases are not here: they are the codecs group. The calibration
readers are not here either: they are the calib group.
"""

# Still fixtures shared with the VXL group.
STILLS = ("rgb8", "gray8", "gray16")

# Frames the temporal implementations see, in order.
SEQUENCE = tuple("seq_{:02d}".format(index) for index in range(12))

# The OpenCV group's own fixtures, from `opencv_fixtures.py`.
CIRCLES = ("circles_rgb",)
HEAT = ("heat",)
BAYER = ("bayer_bg",)


# ----------------------------------------------------------------------------
# image_filter
# ----------------------------------------------------------------------------
IMAGE_FILTERS = {
    # 13 pipelines select it, with every combination below between them.
    "ocv_enhancer": [
        ("defaults", {}),
        ("force_8bit", {"force_8bit": "true"}),
        ("clahe_3", {"apply_clahe": "true", "clip_limit": "3",
                     "apply_smoothing": "false", "apply_denoising": "false"}),
        ("clahe_20", {"apply_clahe": "true", "clip_limit": "20",
                      "apply_smoothing": "false", "apply_denoising": "false"}),
        ("clahe_3_balance", {"apply_clahe": "true", "clip_limit": "3",
                             "auto_balance": "true"}),
        ("saturation_1_20", {"saturation": "1.20"}),
        ("smooth_3", {"apply_smoothing": "true", "smooth_kernel": "3"}),
        ("denoise_3_2", {"apply_denoising": "true", "denoise_kernel": "3",
                         "denoise_coeff": "2"}),
    ],
    # An alias of ocv_enhancer, registered so that the VXL name kept working.
    # Recorded under its own name so that phase 7 cannot break one and not
    # the other.
    "vxl_enhancer": [
        ("defaults", {}),
        ("clahe_3", {"apply_clahe": "true", "clip_limit": "3"}),
    ],
    "ocv_convert_color": [
        ("defaults", {}),
    ],
    "ocv_color_correction": [
        ("defaults", {}),
    ],
    # Values are not reproducible -- see UNSTABLE -- but the shape, the dtype
    # and the config being accepted are.
    "ocv_random_hue_shift": [
        ("defaults", {}),
    ],
}

# Bayer input rather than the stills: a demosaic on an already-colour image
# says nothing.
BAYER_FILTERS = {
    "ocv_debayer": [
        ("bg_force_8bit", {"pattern": "BG", "force_8bit": "true"}),
        ("bg", {"pattern": "BG", "force_8bit": "false"}),
    ],
}

# Sequence input: these carry state from frame to frame.
TEMPORAL_FILTERS = {
    "ocv_optical_flow": [
        ("defaults", {}),
    ],
}


# ----------------------------------------------------------------------------
# split_image
# ----------------------------------------------------------------------------
SPLIT_IMAGES = {
    "ocv": [("defaults", {})],
    "ocv_horizontally": [("defaults", {})],
    "ocv_channels": [("defaults", {})],
    "habcam": [("defaults", {})],
}


# ----------------------------------------------------------------------------
# detect_motion
# ----------------------------------------------------------------------------
DETECT_MOTION = {
    "ocv_3frame_differencing": [
        ("defaults", {}),
        ("shipped", {"frame_separation": "1", "jitter_radius": "1",
                     "max_foreground_fract": "0.15",
                     "max_foreground_fract_thresh": "10"}),
    ],
}


# ----------------------------------------------------------------------------
# image_object_detector
# ----------------------------------------------------------------------------
#
# Recorded as detections rather than as a drawn image: a box moving by a pixel
# is what a golden should say, not a few thousand changed pixels.
DETECTORS = {
    "hough_circle": {
        "inputs": CIRCLES,
        "variants": [
            # The shipped config, plus min_radius: at 0 the accumulator finds
            # nothing on any fixture, which would make the recording empty.
            ("shipped", {"dp": "1", "min_dist": "10", "param1": "200",
                         "param2": "20", "min_radius": "3",
                         "max_radius": "20"}),
        ],
    },
    "detect_heat_map": {
        "inputs": HEAT,
        "variants": [
            ("shipped", {"threshold": "0", "min_area": "100",
                         "max_area": "400000", "min_fill_fraction": "0.05",
                         "force_bbox_width": "-1", "force_bbox_height": "-1",
                         "class_name": "motion"}),
            # The shipped threshold of 0 passes the whole frame and yields one
            # box covering it, so a positive threshold is what actually
            # exercises the connected components and the area filters.
            ("threshold_100", {"threshold": "100", "min_area": "100",
                               "min_fill_fraction": "0.05",
                               "max_area": "400000",
                               "class_name": "motion"}),
            ("threshold_100_small", {"threshold": "100", "min_area": "20",
                                     "min_fill_fraction": "0.05"}),
        ],
    },
}


# Inputs each image_filter case is recorded on, where they are not all of
# STILLS. Keyed by (impl, variant) or by impl. An input left out is one the
# implementation refuses, and REFUSES below says so: the refusal is part of
# the contract too, so a replacement that quietly started accepting a single
# channel image would be a change in behaviour.
FILTER_INPUTS = {
    "ocv_convert_color": ("rgb8",),
    "ocv_random_hue_shift": ("rgb8",),
    ("ocv_enhancer", "denoise_3_2"): ("rgb8",),
}


# (impl, variant, input) -> why the implementation refuses that input.
# Recorded and replayed: the replacement has to refuse it too.
REFUSES = {
    ("ocv_convert_color", "defaults", "gray8"):
        "cvtColor is asked for a three or four channel source",
    ("ocv_convert_color", "defaults", "gray16"):
        "cvtColor is asked for a three or four channel source",
    # Not a clean refusal: it throws only when the trigger draw fires. See
    # UNSTABLE_REFUSAL below and lite-findings.md 1.10.
    ("ocv_random_hue_shift", "defaults", "gray8"):
        "cvtColor BGR2HSV is asked for a three channel source -- but only "
        "when the trigger draw fires, so this throws about half the time",
    ("ocv_random_hue_shift", "defaults", "gray16"):
        "cvtColor BGR2HSV is asked for a three channel source -- but only "
        "when the trigger draw fires, so this throws about half the time",
    ("ocv_enhancer", "denoise_3_2", "gray8"):
        "fastNlMeansDenoisingColored wants CV_8UC3 or CV_8UC4",
    ("ocv_enhancer", "denoise_3_2", "gray16"):
        "the enhancer refuses denoising on anything but 8 bit",
}


# Refusals that are not deterministic, so the replay must not assert them.
# `ocv_random_hue_shift` returns the input untouched when its `trigger_percent`
# draw misses and converts BGR to HSV when it hits, so on a single channel
# image it throws about half the time and passes it through the other half.
# The recording says which inputs it can throw on; nothing can say it will.
UNSTABLE_REFUSAL = {
    ("ocv_random_hue_shift", "defaults"),
}


# ----------------------------------------------------------------------------
# Whole pipelines
# ----------------------------------------------------------------------------
#
# Reader, filters and writer together, which is what catches a config key that
# stops being plumbed through even though every filter computes the right
# thing on its own.
#
# These are the shipped pipelines that use an OpenCV implementation, write
# images, need no GPU model, and are not already recorded by the VXL group --
# recording one twice would only say the same thing twice.
#
# Several of them connect `input.file_name` to the writer's
# `image_file_name`, so they write over their input rather than to a new name.
# `pipeline_runner` digests the fixtures on the way in and keeps the ones
# whose bytes changed, which is what makes them recordable at all.
PIPELINES = (
    "filter_debayer.pipe",
    "filter_debayer_and_enhance.pipe",
    "filter_enhance.pipe",
    "filter_split_and_debayer.pipe",
    "filter_split_left_side.pipe",
    "filter_split_right_side.pipe",
    "train_aug_split.pipe",
)

# Not recorded end to end, with the reason.
PIPELINES_SKIPPED = {
    "filter_draw_dets.pipe":
        "needs a detected_object_set on its input, which the fixture frames "
        "do not carry",
    "register_multimodal_unsync_ocv.pipe":
        "needs two image streams",
    "filter_debayer_and_depth_map.pipe":
        "stereo: the disparity computer waits on a second image stream and "
        "the pipeline never finishes on a single one",
    "filter_stereo_depth_map.pipe": "stereo, as above",
    "measurement_compute_rectified_disparity.pipe":
        "needs a calibration file beside the frames: the disparity computer "
        "refuses to start without `calibration_matrices.npz`",
    "train_aug_split_and_stereo.pipe": "stereo, as above",
    "train_aug_add_motion_and_color_freq.pipe":
        "shipped broken on main; the VXL group records the same failure",
    "train_aug_intensity_hue_motion.pipe": "shipped broken on main, as above",
    "train_aug_enhance_and_add_motion.pipe":
        "driven through input_adapter/output_adapter by the trainer, so "
        "`kwiver runner` has nothing to feed it",
}

# Recorded by the VXL group already, so not repeated here.
PIPELINES_IN_VXL_GROUP = (
    "train_aug_add_color_freq.pipe",
    "train_aug_add_double_motion.pipe",
    "train_aug_add_motion.pipe",
    "train_aug_add_optical_flow.pipe",
    "train_aug_add_optical_flow_adaptive.pipe",
    "train_aug_hue_shifting_only.pipe",
    "train_aug_intensity_color_freq_motion.pipe",
)


# Cases whose values cannot be reproduced, with the reason. Shape and dtype
# are still a contract for these.
UNSTABLE = {
    ("ocv_random_hue_shift", "defaults"):
        "the hue offset is drawn from the global RNG on every call",
}


def filter_inputs(impl, variant):
    if (impl, variant) in FILTER_INPUTS:
        return FILTER_INPUTS[(impl, variant)]
    return FILTER_INPUTS.get(impl, STILLS)


def refusals(impl, variant):
    """{input: reason} for the inputs this case is recorded as refusing."""
    return {name: reason
            for (case_impl, case_variant, name), reason in REFUSES.items()
            if case_impl == impl and case_variant == variant}


def refusal_is_reliable(impl, variant):
    """Whether a recorded refusal can be asserted on replay."""
    return (impl, variant) not in UNSTABLE_REFUSAL


def unstable_reason(impl, variant, input_name=None):
    if input_name is not None:
        reason = UNSTABLE.get((impl, variant, input_name))
        if reason:
            return reason

    return UNSTABLE.get((impl, variant))
