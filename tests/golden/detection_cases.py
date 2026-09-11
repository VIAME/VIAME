"""What the `detection` golden covers.

One implementation: `darknet`, which is the last thing in `plugins/` that
reaches for OpenCV and is what P7-T08 replaces. It is recorded before the
port, like everything else in phase 7, and it is the first golden here that
needs a **model** as well as a fixture -- `generic_detector.cfg`, `.weights`
and `.lbl`, the class-agnostic object proposal from the `yolo-generic` add-on,
installed under `configs/pipelines/models`. A build without that model, or
without a CUDA device, skips the group rather than failing it; the darknet
fork here is built with CUDA and refuses `gpu_index -1` outright.

What the variants have to cover is the **pre- and post-processing**, not the
network: the port replaces `cv::resize`, `cv::cvtColor`, the region of
interest and the `cv::Mat` handed to `Detector::detect`, and leaves the
inference alone. So every resize option is here, with the scale, the chip
step and the edge filter that change what the chipping produces, plus the
greyscale conversion and the two thresholds that decide what survives.

The detector is deterministic on this machine: three runs of the same
configuration on the same fixture give bit-identical boxes and confidences.
"""

# The model the recording is made against, relative to the install's
# `configs/pipelines` directory. Recorded in the manifest so a replay on a
# different install can tell whether it is comparing like with like.
MODEL = "models/generic_detector"

# The scene is 1000 by 800 so the chipping loop actually runs; see
# `detection_fixtures.py` for why that matters.
SCENE = ("detect_scene",)
GRAY = ("detect_scene_gray",)
BOTH = ("detect_scene", "detect_scene_gray")


DETECTORS = {
    "darknet": {
        "inputs": SCENE,
        "variants": [
            # The implementation's own configuration. `resize_option` is
            # `disabled`, so the 1000 by 800 scene goes to a 704 by 704
            # network whole and darknet does its own letterboxing.
            ("defaults", {}),

            # A lower threshold, which is what every shipped VIAME config
            # using a generic proposal sets: the default 0.24 is tuned for a
            # classifier, not a proposal.
            ("thresh_0_10", {"thresh": "0.10"}),
            ("thresh_0_50", {"thresh": "0.50"}),

            # `maintain_ar` is `scale_image_maintaining_ar`: fit inside the
            # network's input without distorting, then pad the rest with
            # black. This is the branch the port has to get exactly right,
            # because the padding decides where every box comes back to.
            ("maintain_ar", {"resize_option": "maintain_ar"}),

            # `scale` resizes by a factor and hands the whole thing over.
            # Both directions, because `cv::resize` picks a different
            # interpolation for each and the port must too.
            ("scale_0_50", {"resize_option": "scale", "scale": "0.5"}),
            ("scale_1_50", {"resize_option": "scale", "scale": "1.5"}),

            # Chipping, which is what VIAME actually configures this
            # detector for. At a 200 step over a 1000 by 800 scene the loop
            # produces three chip columns and two chip rows.
            ("chip", {"resize_option": "chip", "chip_step": "200"}),
            ("chip_step_400", {"resize_option": "chip",
                               "chip_step": "400"}),

            # The last step the chipping survives. At 700 the final chip
            # starts at 700 in both directions and is 300 by 100, which is
            # the small-trailing-chip corner. **At 800 -- the scene's own
            # height -- it throws an OpenCV assertion out of the region of
            # interest**, and at anything above that too, so there is no
            # variant recording a step larger than the image: see
            # `lite-findings.md`.
            ("chip_step_700", {"resize_option": "chip",
                               "chip_step": "700"}),

            # Chipping a scaled image: the two scale factors compose, and
            # every box comes back through both.
            ("chip_scaled", {"resize_option": "chip", "chip_step": "200",
                             "scale": "0.75"}),

            # Detections within N pixels of a chip edge are dropped. The
            # fixture puts blobs across the chip boundaries on purpose.
            ("chip_edge_filter_20", {"resize_option": "chip",
                                     "chip_step": "200",
                                     "chip_edge_filter": "20"}),

            # The chips plus the whole frame, which is how a VIAME pipeline
            # catches an object too big for one chip.
            ("chip_and_original", {"resize_option": "chip_and_original",
                                   "chip_step": "200"}),

            # `adaptive` picks between `chip_and_original` and `maintain_ar`
            # by pixel count. The threshold is set below this scene's
            # 800,000 pixels here and above it in the next variant, so both
            # sides of the branch are recorded.
            ("adaptive_chips", {"resize_option": "adaptive",
                                "chip_step": "200",
                                "chip_adaptive_thresh": "500000"}),
            ("adaptive_whole", {"resize_option": "adaptive",
                                "chip_adaptive_thresh": "2000000"}),

            # Non-maximum suppression across everything the regions returned.
            ("nms_0_10", {"resize_option": "chip", "chip_step": "200",
                          "nms_threshold": "0.10"}),
            ("nms_0_90", {"resize_option": "chip", "chip_step": "200",
                          "nms_threshold": "0.90"}),
        ],
    },
}

# Variants that run on the single channel fixture instead, for the greyscale
# conversion. `gs_to_rgb` defaults true, and the false case is what a network
# expecting three channels is handed when it is off.
GRAY_VARIANTS = [
    ("gray_defaults", {}),
    ("gray_no_rgb", {"gs_to_rgb": "false"}),
    # The greyscale conversion happens once before chipping and again for the
    # `chip_and_original` whole-frame region, which is two call sites the
    # port has to keep
    ("gray_chip_and_original", {"resize_option": "chip_and_original",
                                "chip_step": "200"}),
]


# An image smaller than the network, which is 704 by 704 here. The chipping
# loop runs while `li < cols - net_width + chip_step`, so on a 96 by 64 frame
# with any ordinary step the bound is negative, the body never runs, no region
# is produced and the detector returns **nothing at all** -- where the same
# frame with resizing disabled gives six detections. Recorded rather than
# left out: it is a defect, and pinning it means a port that quietly started
# returning something would be noticed rather than welcomed.
SMALL = ("rgb8",)

SMALL_VARIANTS = [
    ("small_image_chip", {"resize_option": "chip", "chip_step": "100"}),
    ("small_image_disabled", {"resize_option": "disabled"}),
    ("small_image_maintain_ar", {"resize_option": "maintain_ar"}),
]


# `adaptive` picks its mode from the first frame's pixel count and **keeps
# it**: `detect` is a const method, but it writes the choice back into the
# private state, so every later frame gets whatever the first one decided.
# One detector over two frames is what shows it, and the golden runner
# already drives a case that way -- it creates the algorithm once and walks
# the inputs -- so the recording carries the latch rather than describing it.
#
# The threshold sits between the two fixtures: the 1000 by 800 scene is
# 800,000 pixels and chips, the 900 by 750 crop is 675,000 and would not.
LATCH = ("detect_scene", "detect_scene_medium")
MEDIUM = ("detect_scene_medium",)

LATCH_VARIANTS = [
    ("adaptive_latch", {"resize_option": "adaptive",
                        "chip_adaptive_thresh": "700000",
                        "chip_step": "200"}),
]

MEDIUM_VARIANTS = [
    # The same frame, the same configuration, on a detector that has not seen
    # the larger one. Read against `adaptive_latch`'s second output.
    ("adaptive_medium_alone", {"resize_option": "adaptive",
                               "chip_adaptive_thresh": "700000",
                               "chip_step": "200"}),
]


REPLACEMENTS = {}


def divergence_reason(impl, variant, input_name=None):
    """No documented divergence: this group is one implementation.

    A port of the pre-processing either puts the same pixels in front of the
    network or it does not, and the network is not being replaced.
    """
    return None
