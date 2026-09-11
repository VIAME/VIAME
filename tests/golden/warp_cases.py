"""What the image warping golden covers.

`warp_image:ocv` is `plugins/opencv/warp_image_ocv.cxx`, and it had no
recording at all until P7-T04b -- the interface was not even bound to python,
which is why. It is the only `warp_image` in the tree.

The homographies are stated here rather than derived, so a reader can see
what each case asks for. All four corners of the behaviour are covered: the
plain warp onto a blank destination, a warp onto an existing image (where the
covered region decides what survives), the alpha blend, and a homography that
sends most of the source off the edge.
"""

# The source and the destination, both committed fixtures.
SOURCE = "rgb8"
DESTINATION = "seq_00"

# The mask fixture used for the alpha cases: single channel, and already in
# `inputs/`.
MASK = "mask"

# Homographies, row major. A translation, a scale about the origin, a
# rotation with a shift, and one that pushes the source mostly out of frame.
HOMOGRAPHIES = {
    "identity": ((1.0, 0.0, 0.0),
                 (0.0, 1.0, 0.0),
                 (0.0, 0.0, 1.0)),
    "translate": ((1.0, 0.0, 12.0),
                  (0.0, 1.0, -7.0),
                  (0.0, 0.0, 1.0)),
    "scale_rotate": ((0.9063, -0.4226, 20.0),
                     (0.4226, 0.9063, -5.0),
                     (0.0, 0.0, 1.0)),
    # Perspective, so the warp is not an affine one: a real homography is
    # what this algorithm exists for and the only case with a non-zero
    # bottom row.
    "perspective": ((1.05, 0.08, -6.0),
                    (0.02, 0.97, 4.0),
                    (0.0006, -0.0003, 1.0)),
    "off_frame": ((1.0, 0.0, 80.0),
                  (0.0, 1.0, 50.0),
                  (0.0, 0.0, 1.0)),
}

# (variant, homography, has destination, has alpha mask)
WARPS = (
    ("identity", "identity", True, False),
    ("translate", "translate", True, False),
    ("translate_no_dest", "translate", False, False),
    ("scale_rotate", "scale_rotate", True, False),
    ("perspective", "perspective", True, False),
    ("off_frame", "off_frame", True, False),
    ("translate_alpha", "translate", True, True),
    ("perspective_alpha", "perspective", True, True),
)

IMPLEMENTATION = "ocv"


def unstable_reason(impl, variant, name=None):
    return None


# **The recorded alpha blend is wrong**, and the port does not reproduce it.
#
# `dest_float.mul( 1.0 - weights )` looks like "one minus the weight", and in
# OpenCV a `double` on the left of a `Mat` becomes `Scalar( 1, 0, 0, 0 )`.
# Subtracting a Scalar from a three channel Mat is per channel, so
# `1.0 - weights` is `[ 1 - w, -w, -w ]`: only the first channel blends, and
# the other two compute `warped * w - dest * w`. On a colour image that is
# visible garbage, not a rounding difference -- 80 counts of mean error
# against a correct blend.
#
# Latent: the only caller of `warp_image` in the tree,
# `plugins/core/warp_image_process`, never passes an alpha mask. So this is
# the same situation as `vxl_threshold`'s percentile mode in P3 -- a bug on a
# path nothing uses -- and takes the same answer: the replacement is correct
# rather than bug compatible, and the recording stays as evidence of what the
# old one did. `design/lite-findings.md` records it.
DIVERGENCES = {
    ("ocv", "translate_alpha"):
        "the recorded alpha blend subtracts the weight from channels 1 and 2 "
        "instead of blending them, because OpenCV reads `1.0 - weights` as "
        "`Scalar( 1, 0, 0, 0 ) - weights`; the port blends every channel",
    ("ocv", "perspective_alpha"):
        "the recorded alpha blend subtracts the weight from channels 1 and 2 "
        "instead of blending them, because OpenCV reads `1.0 - weights` as "
        "`Scalar( 1, 0, 0, 0 ) - weights`; the port blends every channel",
}


def divergence_reason(impl, variant, input_name=None):
    return DIVERGENCES.get((impl, variant))
