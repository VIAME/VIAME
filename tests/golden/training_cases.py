"""What the training chip golden covers.

The two windowed trainers, `windowed` (plugins/core) and `ocv_windowed`
(plugins/opencv), each cut training images into chips, write them under
`train_directory` with a manifest of the boxes that landed on each, and hand
the list to a nested trainer. P2-T07 merges them, and the detector and refiner
merges of P2-T05 set the rule: both are recorded first, and the survivor has
to replay both recordings.

What is recorded is everything the chipping decides -- which chips exist, their
pixels, and the boxes and categories written against each -- and nothing the
nested trainer does. `frame_diff` is the nested one because it is registered
wherever python is, and its `add_data_from_disk` only keeps the lists.

The scene is the detection group's, whose blobs were placed to straddle chip
edges; the medium crop sits below `chip_adaptive_thresh` where the full scene
is above it, so the adaptive and original_and_resized modes each take both
branches.
"""

import detection_fixtures

INPUTS = ("detect_scene", "detect_scene_medium")

SIZES = {
    "detect_scene": (detection_fixtures.WIDTH, detection_fixtures.HEIGHT),
    "detect_scene_medium": (detection_fixtures.MEDIUM_WIDTH,
                            detection_fixtures.MEDIUM_HEIGHT),
}


def _category(index):
    # Every seventh is the default `ignore_category`, so the ignored path is
    # exercised: chips are still cut around it, and it is left off them
    if index % 7 == 6:
        return "false_alarm"
    return "fish" if index % 2 == 0 else "scallop"


# (min_x, min_y, max_x, max_y, category), from the blobs' centres and radii
TRUTH = tuple(
    (float(cx - max(rx, ry)), float(cy - max(rx, ry)),
     float(cx + max(rx, ry)), float(cy + max(rx, ry)), _category(index))
    for index, (cx, cy, rx, ry, _angle, _colour)
    in enumerate(detection_fixtures.BLOBS)
)

# One chipping thread: synthetic category ids are assigned in the order boxes
# are first seen, which is only an order when there is one thread -- and an
# image reader, which neither trainer defaults: without one the first load
# dereferences nothing. PNG is lossless, so any reader reads the same pixels
COMMON = {
    "trainer:type": "frame_diff",
    "trainer:frame_diff:identifier": "golden",
    "image_reader:type": "ocv",
    "chip_threads": "1",
}

CHIP = {"mode": "chip", "chip_width": "400", "chip_height": "300",
        "chip_step_width": "300", "chip_step_height": "250"}

TRAINERS = {
    impl: [
        ("chip", dict(COMMON, **CHIP)),
        ("chip_and_original", dict(COMMON, **dict(CHIP,
                                                  mode="chip_and_original"))),
        ("original_and_resized", dict(COMMON, **dict(
            CHIP, mode="original_and_resized",
            chip_adaptive_thresh="700000"))),
        ("adaptive", dict(COMMON, **dict(CHIP, mode="adaptive",
                                         chip_adaptive_thresh="700000"))),
        ("uneven_black_pad", dict(COMMON, **dict(
            CHIP, chip_width="430", chip_height="310",
            chip_step_width="350", chip_step_height="260",
            black_pad="true"))),
        # Background chips are shuffled with the per-frame generator, so this
        # is the variant that says both trainers draw the same random chips
        ("gt_only_background", dict(COMMON, **dict(
            CHIP, chips_w_gt_only="true", background_chip_ratio="0.5"))),
        ("overlap_half", dict(COMMON, **dict(CHIP, overlap_required="0.5"))),
        ("disabled_rewrite", dict(COMMON, mode="disabled",
                                  always_write_image="true")),
        # The two whole-image modes four shipped trainer configs use, and
        # `adaptive` on each name's own `original_to_chip_size` default --
        # which is where the two differ: core's is true, opencv's false
        ("scale_half", dict(COMMON, mode="scale", scale="0.5")),
        ("maintain_ar", dict(COMMON, **dict(CHIP, mode="maintain_ar"))),
        ("adaptive_chip_size", dict(COMMON, **dict(
            CHIP, mode="adaptive", chip_adaptive_thresh="700000",
            original_to_chip_size="true"))),
        ("adaptive_no_chip_size", dict(COMMON, **dict(
            CHIP, mode="adaptive", chip_adaptive_thresh="700000",
            original_to_chip_size="false"))),
    ]
    for impl in ("windowed", "ocv_windowed")
}


# The merge kept opencv's parameter list, and with it opencv's default for
# `original_to_chip_size`: false. Core's copy defaulted it to true, so the one
# variant that leaves it unset diverges for `windowed` -- a name no shipped
# config selects -- while `ocv_windowed`, which three of them do, replays.
# `adaptive_chip_size` is core's old default spelled out, and replays for both.
DIVERGENCES = {
    ("windowed", "adaptive"):
        "original_to_chip_size defaults to false since the P2-T07 merge, as it "
        "did for ocv_windowed; windowed defaulted it to true, recorded as "
        "adaptive_chip_size",
}


def divergence_reason(impl, variant, input_name=None):
    return DIVERGENCES.get((impl, variant))
