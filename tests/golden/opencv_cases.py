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
        # The shipped one, and the only pair any config selects:
        # `train_aug_intensity_hue_motion.pipe` asks for rgb to hls, which
        # is also the default.
        ("defaults", {}),
        # The pairs P7-T04b keeps beside it, recorded so the port is held to
        # them rather than to reasoning. Round trips as well as forward
        # conversions, since the inverse is where a scale factor goes
        # missing.
        ("rgb_to_hsv", {"input_color_space": "rgb",
                        "output_color_space": "hsv"}),
        ("rgb_to_lab", {"input_color_space": "rgb",
                        "output_color_space": "lab"}),
        ("hls_to_rgb", {"input_color_space": "hls",
                        "output_color_space": "rgb"}),
        ("hsv_to_rgb", {"input_color_space": "hsv",
                        "output_color_space": "rgb"}),
        ("lab_to_rgb", {"input_color_space": "lab",
                        "output_color_space": "rgb"}),
        # BGR in and out, which OpenCV serves with its own constants rather
        # than by swapping, and which a port that only implements RGB has to
        # get right by swapping.
        ("bgr_to_hls", {"input_color_space": "bgr",
                        "output_color_space": "hls"}),
        ("hls_to_bgr", {"input_color_space": "hls",
                        "output_color_space": "bgr"}),
    ],
    "ocv_color_correction": [
        # The defaults turn every stage off, so this one records that the
        # filter leaves an image alone -- which is worth pinning, but says
        # nothing about the four hundred lines behind the switches.
        ("defaults", {}),
        # Every stage, and every branch inside one. The settings are the
        # ones `examples/image_enhancement/README.rst` documents: no
        # shipped pipeline selects this filter, but the README gives it a
        # worked example, so it is a documented feature rather than dead
        # code.
        ("gamma_fixed", {"apply_gamma": "true", "gamma": "1.8"}),
        ("gamma_auto", {"apply_gamma": "true", "gamma_auto": "true"}),
        ("gray_world", {"apply_gray_world": "true"}),
        ("gray_world_loose", {"apply_gray_world": "true",
                              "gray_world_sat_threshold": "0.6"}),
        ("underwater_simple", {"apply_underwater": "true"}),
        ("underwater_no_backscatter", {"apply_underwater": "true",
                                       "backscatter_removal": "false"}),
        ("underwater_coastal", {"apply_underwater": "true",
                                "water_type": "coastal"}),
        ("underwater_turbid", {"apply_underwater": "true",
                               "water_type": "turbid"}),
        ("underwater_manual", {"apply_underwater": "true",
                               "water_type": "custom",
                               "red_attenuation": "0.65",
                               "green_attenuation": "0.35",
                               "blue_attenuation": "0.15"}),
        ("underwater_no_depth", {"apply_underwater": "true",
                                 "use_auto_depth": "false"}),
        ("underwater_fusion", {"apply_underwater": "true",
                               "underwater_method": "fusion"}),
        # The README's own example, which is three stages at once.
        ("readme_example", {"apply_gamma": "true", "gamma_auto": "true",
                            "apply_underwater": "true",
                            "underwater_method": "fusion",
                            "water_type": "coastal"}),
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
    # `ocv_windowed` chips an image and runs another detector on each chip,
    # then maps the boxes back. `example_detector` is the nested one because
    # it is deterministic and returns a fixed box per image, which makes the
    # recording a record of the **chipping geometry** -- where each chip was
    # taken from and how its boxes came back -- rather than of a detector.
    "ocv_windowed": {
        "inputs": ("rgb8",),
        "variants": [
            ("disabled", {"detector:type": "example_detector",
                          "mode": "disabled"}),
            ("chip", {"detector:type": "example_detector", "mode": "chip",
                      "chip_width": "32", "chip_height": "24",
                      "chip_step_width": "16", "chip_step_height": "12"}),
            ("chip_and_original", {"detector:type": "example_detector",
                                   "mode": "chip_and_original",
                                   "chip_width": "32", "chip_height": "24",
                                   "chip_step_width": "16",
                                   "chip_step_height": "12"}),
            ("original_and_resized", {"detector:type": "example_detector",
                                      "mode": "original_and_resized",
                                      "chip_width": "32",
                                      "chip_height": "24"}),
            # A chip that does not divide the image, so the last column and
            # row are partial -- which is where a crop goes wrong.
            ("uneven_chips", {"detector:type": "example_detector",
                              "mode": "chip",
                              "chip_width": "40", "chip_height": "25",
                              "chip_step_width": "35",
                              "chip_step_height": "20"}),
            # Padding rather than stretching a partial chip.
            ("black_pad", {"detector:type": "example_detector",
                           "mode": "chip", "black_pad": "true",
                           "chip_width": "40", "chip_height": "25",
                           "chip_step_width": "35",
                           "chip_step_height": "20"}),
            # A scale other than one, which resizes before chipping.
            ("scaled", {"detector:type": "example_detector", "mode": "chip",
                        "scale": "2", "chip_width": "32",
                        "chip_height": "24", "chip_step_width": "16",
                        "chip_step_height": "12"}),
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

            # The fixed-size path, which no shipped pipeline selects --
            # every one of them sets `force_bbox_width: -1`. Recorded
            # anyway, and before it is ported: it is two hundred lines of
            # window placement, and the only moment there is anything to
            # record it against is while OpenCV is still here.
            #
            # `get_bbox_fixed_size_dense` is not among them: it is in the
            # file and nothing calls it.
            ("forced_24x16", {"force_bbox_width": "24",
                              "force_bbox_height": "16",
                              "bbox_buffer": "4",
                              "threshold": "100",
                              "class_name": "motion"}),
            ("forced_16x12_buffer_2", {"force_bbox_width": "16",
                                       "force_bbox_height": "12",
                                       "bbox_buffer": "2",
                                       "threshold": "100",
                                       "class_name": "motion"}),
            ("forced_20x20_no_threshold", {"force_bbox_width": "20",
                                           "force_bbox_height": "20",
                                           "bbox_buffer": "0",
                                           "max_boxes": "4",
                                           "class_name": "motion"}),
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
# Processes whose output is a detected object set rather than an image, driven
# through a golden-local pipeline because no shipped one can be run here.
#
# `detect_in_subregions` is selected by four arctic-seal add-on pipelines, all
# of which need a downloaded YOLO model and two or three cameras. The local
# pipeline drives it on the fixture frames with `ocv_windowed` over
# `example_detector` for its input regions, which is deterministic and spreads
# half a dozen overlapping boxes across the frame.
PROCESS_PIPELINES = {
    "detect_in_subregions": {
        "pipeline": "detect_in_subregions.pipe",
        "variants": [
            ("detection_box", ()),
            # Square regions about each input box's centre instead
            ("fixed_size", ("subregions:method=fixed_size",
                            "subregions:fixed_size=64")),
            # A fixed size large enough that later centres fall inside an
            # earlier region, which is the de-duplication path
            ("fixed_size_dedup", ("subregions:method=fixed_size",
                                  "subregions:fixed_size=400")),
            ("include_input_dets", ("subregions:include_input_dets=true",)),
            ("max_subregions", ("subregions:max_subregion_count=2",)),
            # Regions that run off the right and bottom edges, so the clip to
            # the image is what decides the crop
            ("edge_clipped", (
                "proposals:detector:ocv_windowed:detector:"
                "example_detector:center_x=150",
                "proposals:detector:ocv_windowed:detector:"
                "example_detector:center_y=120",
                "proposals:detector:ocv_windowed:detector:"
                "example_detector:width=120",
                "proposals:detector:ocv_windowed:detector:"
                "example_detector:height=100")),
        ],
    },
}


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
