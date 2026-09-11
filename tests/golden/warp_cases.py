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


def divergence_reason(impl, variant, input_name=None):
    return None
