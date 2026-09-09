"""What the VXL golden recording covers.

The config variants are the ones the shipped pipelines actually use, taken
from every `.pipe` and `.conf` under the install's `configs/pipelines` (see
`scan_pipelines.py`), plus the implementation's own defaults. A replacement
has to reproduce every one of them.
"""

# Still fixtures every non-temporal filter runs on.
STILLS = ("rgb8", "gray8", "gray16")

# The only input morphology accepts: it works on binary masks.
MASKS = ("mask",)

# Frames a temporal filter runs on, in order.
SEQUENCE = tuple("seq_{:02d}".format(index) for index in range(12))


# Each entry: implementation name -> list of (variant id, config dict).
# "defaults" is the implementation's own configuration, unmodified.
IMAGE_FILTERS = {
    "vxl_convert_image": [
        ("defaults", {}),
        ("byte", {"format": "byte"}),
        ("byte_three_channel", {"format": "byte", "force_three_channel": "true"}),
        ("byte_single_channel", {"format": "byte", "single_channel": "true"}),
        ("byte_scale_0_25", {"format": "byte", "scale_factor": "0.25"}),
        ("byte_scale_0_50", {"format": "byte", "scale_factor": "0.50"}),
        ("byte_scale_0_75", {"format": "byte", "scale_factor": "0.75"}),
        ("byte_scale_255", {"format": "byte", "scale_factor": "255.0"}),
        ("byte_scale_4_single", {"format": "byte", "scale_factor": "4.0",
                                 "single_channel": "true"}),
        # random_grayscale is a training augmentation and draws from the
        # global RNG, so it is recorded only as its config being accepted
        ("byte_three_channel_gray_0_20", {"format": "byte",
                                          "force_three_channel": "true",
                                          "random_grayscale": "0.20"}),
        ("byte_three_channel_gray_0_25", {"format": "byte",
                                          "force_three_channel": "true",
                                          "random_grayscale": "0.25"}),
        ("float", {"format": "float"}),
        ("uint16", {"format": "uint16"}),
        ("percentile_norm", {"format": "byte", "percentile_norm": "0.02"}),
    ],
    "vxl_average": [
        ("defaults", {}),
        ("window_5", {"type": "window", "window_size": "5", "round": "false",
                      "output_variance": "true"}),
        ("window_20", {"type": "window", "window_size": "20", "round": "false",
                       "output_variance": "true"}),
        ("window_30", {"type": "window", "window_size": "30", "round": "false",
                       "output_variance": "true"}),
        ("window_5_mean", {"type": "window", "window_size": "5",
                           "round": "false", "output_variance": "false"}),
        ("window_5_round", {"type": "window", "window_size": "5",
                            "round": "true", "output_variance": "false"}),
        ("cumulative", {"type": "cumulative", "round": "false",
                        "output_variance": "false"}),
        ("exponential", {"type": "exponential", "exp_weight": "0.3",
                         "round": "false", "output_variance": "false"}),
    ],
    "vxl_color_commonality": [
        ("defaults", {}),
        ("res16_hist32", {"color_resolution_per_channel": "16",
                          "intensity_resolution": "32"}),
        ("res16_hist32_scale256", {"color_resolution_per_channel": "16",
                                   "intensity_resolution": "32",
                                   "output_scale": "256"}),
        ("res16_hist32_scale512", {"color_resolution_per_channel": "16",
                                   "intensity_resolution": "32",
                                   "output_scale": "512"}),
        ("grid", {"grid_image": "true", "grid_resolution_width": "2",
                  "grid_resolution_height": "2"}),
    ],
    "vxl_morphology": [
        ("defaults", {}),
        ("open_disk_1", {"morphology": "open", "element_shape": "disk",
                         "kernel_radius": "1"}),
        ("close_disk_4", {"morphology": "close", "element_shape": "disk",
                          "kernel_radius": "4"}),
        ("erode_disk_2", {"morphology": "erode", "element_shape": "disk",
                          "kernel_radius": "2"}),
        ("dilate_disk_2", {"morphology": "dilate", "element_shape": "disk",
                           "kernel_radius": "2"}),
    ],
    "vxl_threshold": [
        ("defaults", {}),
        ("absolute_15", {"type": "absolute", "threshold": "15"}),
        ("absolute_128", {"type": "absolute", "threshold": "128"}),
        ("percentile_0_8", {"type": "percentile", "threshold": "0.8"}),
    ],
    "vxl_enhancer": [
        ("defaults", {}),
        ("clahe", {"apply_clahe": "true", "clip_limit": "3.0"}),
        ("sharpen", {"apply_sharpening": "true", "sharpening_weight": "0.5"}),
        ("smooth", {"apply_smoothing": "true", "smoothing_kernel": "3"}),
        ("auto_balance_8bit", {"auto_balance": "true", "force_8bit": "true"}),
    ],
    "vxl_white_balancing": [
        ("defaults", {}),
        ("scaled", {"white_scale_factor": "0.9", "black_scale_factor": "0.1"}),
        ("res32", {"matrix_resolution": "32"}),
    ],
}

# Temporal filters see the sequence, mask filters the mask, everything else
# the stills.
TEMPORAL = {"vxl_average"}
MASK_ONLY = {"vxl_morphology"}


# image_io reads the committed fixtures back; the replacement has to decode
# them identically.
IMAGE_IO = {
    "vxl": [
        ("defaults", {}),
        ("force_byte", {"force_byte": "true"}),
        ("no_force_byte", {"force_byte": "false"}),
        ("split_channels", {"split_channels": "true"}),
        ("no_split_channels", {"split_channels": "false"}),
        ("auto_stretch", {"auto_stretch": "true"}),
    ],
}


# Whole pipelines recorded end to end: reader, filters and writer together,
# which is what catches a config key that stops being plumbed through even
# though every filter still computes the right thing on its own.
#
# These are the shipped pipelines that both use a vxl filter and write images.
# The detector and tracker pipelines that use vxl_convert_image on their input
# emit detections from GPU models rather than images, and are covered by the
# per filter cases above plus the existing PIPELINES ctests.
PIPELINES = (
    "train_aug_add_color_freq.pipe",
    "train_aug_add_double_motion.pipe",
    "train_aug_add_motion.pipe",
    "train_aug_add_optical_flow.pipe",
    "train_aug_add_optical_flow_adaptive.pipe",
    "train_aug_all_motion.pipe",
    "train_aug_hue_shifting_only.pipe",
    "train_aug_intensity_color_freq_motion.pipe",
    "train_aug_motion_only.pipe",
)

# Not recorded end to end, with the reason:
#
# - the two "broken" ones are shipped broken on main: image_merger declares no
#   image3 port, so they do not bake. tests/baseline/pipes.json records the
#   same failure, which is what would notice a fix
# - the embedded one is driven through input_adapter/output_adapter by the
#   trainer, so `kwiver runner` has nothing to feed it and it blocks forever
PIPELINES_BROKEN = (
    "train_aug_add_motion_and_color_freq.pipe",
    "train_aug_intensity_hue_motion.pipe",
)
PIPELINES_EMBEDDED = (
    "train_aug_enhance_and_add_motion.pipe",
)

# The frames every recorded pipeline runs on, in order.
PIPELINE_INPUTS = tuple("frame_{:02d}".format(index) for index in range(6))


# Cases whose recorded values cannot be reproduced, and why. Keys are
# (impl, variant) for a whole case or (impl, variant, input) for one input.
# The recording still pins the shape, dtype and that the config is accepted;
# only the value comparison is skipped.
#
# The two threshold entries and the commonality one are VXL bugs, not
# tolerable variation: the output buffer is sized and then never written, so
# the result is whatever was on the heap. Both paths are latent, no shipped
# pipeline uses them. A replacement should be correct rather than bug
# compatible, which is why the values are not held to the recording.
UNSTABLE = {
    ("vxl_convert_image", "byte_three_channel_gray_0_20"):
        "random_grayscale draws from the global RNG",
    ("vxl_convert_image", "byte_three_channel_gray_0_25"):
        "random_grayscale draws from the global RNG",
    ("vxl_color_commonality", "grid"):
        "grid mode leaves part of its output buffer uninitialised",
    ("vxl_threshold", "defaults", "rgb8"):
        "percentile mode on a multi-plane image returns an uninitialised "
        "buffer: vil_threshold_above resizes the plane view it is handed",
    ("vxl_threshold", "percentile_0_8", "rgb8"):
        "percentile mode on a multi-plane image returns an uninitialised "
        "buffer: vil_threshold_above resizes the plane view it is handed",
    ("train_aug_hue_shifting_only.pipe", "default"):
        "the pipeline applies vxl_convert_image's random_grayscale, which "
        "draws from the global RNG",
}


def unstable_reason(impl, variant, input_name=None):
    """Why this case's values are not comparable, or None if they are."""
    if input_name is not None:
        reason = UNSTABLE.get((impl, variant, input_name))
        if reason:
            return reason

    return UNSTABLE.get((impl, variant))


def inputs_for(impl):
    if impl in TEMPORAL:
        return SEQUENCE
    if impl in MASK_ONLY:
        return MASKS
    return STILLS
