"""What the detection refinement golden covers.

Three `refine_detections` implementations that reach OpenCV and that P7-T04b
and P7-T07 replace: the two segmenters, which `lite-removals.md` section 2.6
sends to python, and `add_keypoints_from_mask`, which measures a mask and
hangs keypoints off the detection.

The detections are built here rather than read from a file. A refiner takes
an image *and* a detection set, and the set is as much of the input as the
image is -- so it belongs in the case file where a reader can see it, not in
a fixture nobody opens. The boxes are placed over `rgb8.png`'s disc and over
a stretch of its bars, which are the two things in that fixture a segmenter
can find or fail to find.
"""

# The fixture every case runs on.
IMAGE = "rgb8"

# The input detections, by name. A `mask` is an ellipse inscribed in the box,
# which is what a detector that segments would have produced and what
# `seed_with_existing_masks` and `add_keypoints_from_mask` both need.
#
# `disc` sits over the fixture's bright disc, which has a clean boundary.
# `bars` sits over the gradient and bars, which has no object in it at all --
# a segmenter that finds something there is finding texture, and recording
# what it finds is the point.
DETECTIONS = (
    {"name": "disc", "bbox": (18.0, 21.0, 47.0, 49.0), "type": "fish",
     "confidence": 0.9, "mask": "ellipse"},
    {"name": "bars", "bbox": (60.0, 8.0, 88.0, 30.0), "type": "fish",
     "confidence": 0.6, "mask": "ellipse"},
    # Deliberately over the edge: a box the image only partly contains, which
    # is where a crop or a rectangle intersection goes wrong.
    {"name": "edge", "bbox": (80.0, 44.0, 110.0, 70.0), "type": "scallop",
     "confidence": 0.4, "mask": None},
)

# How much of the box an `ellipse` mask fills, as a fraction of its half
# extent. Under one so the mask is inside the box rather than touching it,
# which is what a real segmentation looks like.
ELLIPSE_FILL = 0.8


REFINERS = {
    # `cv::grabCut`: a GMM over the context box, iterated. Seeded from the
    # detection's own mask when it has one, which is the shipped default,
    # and from a scaled-down box when it does not.
    "ocv_grabcut": [
        ("defaults", {}),
        ("no_seed", {"seed_with_existing_masks": "false"}),
        ("iter_5", {"iter_count": "5"}),
        # A context box the same size as the detection, which is the one
        # case the implementation skips `grabCut` entirely -- there is no
        # background to learn from.
        ("no_context", {"context_scale_factor": "1"}),
    ],
    # `cv::watershed`: a seed inside, an uncertain ring, and the boundary
    # found between them.
    "ocv_watershed": [
        ("defaults", {}),
        ("no_seed", {"seed_with_existing_masks": "false"}),
        ("tight_seed", {"seed_scale_factor": "0.1"}),
    ],
    # The two windowed refiners: one chipper each, registered under
    # `windowed` and `ocv_windowed`, which P2-T05 merges into one
    # implementation under both names. `add_fixed` is the nested refiner
    # because it is deterministic and adds one box the size of whatever
    # image it is given -- so what these record is the **chipping
    # geometry**, where each chip was taken from and how its boxes came
    # back, which is exactly what the merge has to preserve. Recorded
    # before the merge, against both copies, so that the survivor can be
    # held to what each of them did.
    "windowed": [
        ("disabled", {"refiner:type": "add_fixed", "mode": "disabled"}),
        ("chip", {"refiner:type": "add_fixed", "mode": "chip",
                  "chip_width": "32", "chip_height": "24",
                  "chip_step_width": "16", "chip_step_height": "12"}),
        ("uneven_chips", {"refiner:type": "add_fixed", "mode": "chip",
                          "chip_width": "40", "chip_height": "25",
                          "chip_step_width": "35", "chip_step_height": "20"}),
        ("black_pad", {"refiner:type": "add_fixed", "mode": "chip",
                       "black_pad": "true",
                       "chip_width": "40", "chip_height": "25",
                       "chip_step_width": "35", "chip_step_height": "20"}),
    ],
    "ocv_windowed": [
        ("disabled", {"refiner:type": "add_fixed", "mode": "disabled"}),
        ("chip", {"refiner:type": "add_fixed", "mode": "chip",
                  "chip_width": "32", "chip_height": "24",
                  "chip_step_width": "16", "chip_step_height": "12"}),
        ("uneven_chips", {"refiner:type": "add_fixed", "mode": "chip",
                          "chip_width": "40", "chip_height": "25",
                          "chip_step_width": "35", "chip_step_height": "20"}),
        ("black_pad", {"refiner:type": "add_fixed", "mode": "chip",
                       "black_pad": "true",
                       "chip_width": "40", "chip_height": "25",
                       "chip_step_width": "35", "chip_step_height": "20"}),
    ],
    # Keypoints from the mask's shape. Five methods, and the clip; the
    # default is `oriented_bbox`.
    "add_keypoints_from_mask": [
        ("defaults", {}),
        ("clip_to_mask", {"clip_to_mask": "true"}),
        ("pca", {"method": "pca"}),
        ("farthest", {"method": "farthest"}),
        ("hull_extremes", {"method": "hull_extremes"}),
        ("skeleton", {"method": "skeleton"}),
    ],
}


def unstable_reason(impl, variant, name=None):
    """Nothing here is unstable: all three are deterministic over three runs."""
    return None


def divergence_reason(impl, variant, input_name=None):
    """Nothing here diverges from the recording yet."""
    return None
