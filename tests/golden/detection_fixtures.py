"""The input images the darknet golden needs.

`darknet`'s chip geometry is derived from the **network's** input size, which
for `generic_detector.cfg` is 704 by 704, and its chipping loop is

    for( li = 0; li < cols - net_width + chip_step; li += chip_step )

so on any of the 96 by 64 fixtures the loop body never runs, no region is
produced, and a `chip` recording would be a recording of nothing. Chipping is
most of what VIAME configures this detector for, so it needs a fixture it can
actually chip: at 1000 by 800 with a 200 pixel step there are three chip
columns and two chip rows, six regions plus the original.

The scene is built rather than photographed so it stays in the repository and
stays the same. What it has to be is *detectable*: `generic_detector` is a
class-agnostic object proposal, and it fires on compact things that stand
apart from their surroundings, so the scene is a textured seabed with
twenty-odd blobs of varied size, contrast and shape scattered over it -- some
inside one chip, some deliberately straddling a chip boundary, which is what
`chip_edge_filter` and the non-maximum suppression across regions exist for.
"""

import numpy as np

WIDTH = 1000
HEIGHT = 800
SEED = 20260911

# Where the chip boundaries fall for chip_step 200 and a 704 wide net, so the
# blobs can be placed across them on purpose
CHIP_STEP = 200


def _background(rng):
    """A mottled seabed: low frequency shading plus grain.

    Smooth enough that a blob stands out, busy enough that a proposal network
    has something to reject.
    """
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH]

    shade = (
        118.0
        + 26.0 * np.sin(xs / 140.0) * np.cos(ys / 190.0)
        + 14.0 * np.sin((xs + ys) / 73.0)
        + 9.0 * np.cos((xs - 2 * ys) / 51.0)
    )

    grain = rng.normal(0.0, 7.0, (HEIGHT, WIDTH))

    # A coarser mottle, so the texture has more than one scale
    coarse = rng.normal(0.0, 30.0, (HEIGHT // 16 + 1, WIDTH // 16 + 1))
    coarse = np.repeat(np.repeat(coarse, 16, axis=0), 16, axis=1)
    coarse = coarse[:HEIGHT, :WIDTH]

    field = shade + grain + 0.5 * coarse

    # Three channels with a blue-green cast, the way underwater imagery sits
    image = np.dstack([field * 0.82, field * 1.00, field * 0.93])

    return image


def _blob(image, centre_x, centre_y, radius_x, radius_y, angle, colour,
          rng, edge=2.5):
    """One soft-edged ellipse, drawn over whatever is there."""
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH]

    dx = xs - centre_x
    dy = ys - centre_y

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    u = (dx * cos_a + dy * sin_a) / radius_x
    v = (-dx * sin_a + dy * cos_a) / radius_y

    distance = np.sqrt(u * u + v * v)

    # A soft edge rather than a hard one: a hard ellipse edge is an artefact
    # the network can latch onto, and the point of the fixture is the shape,
    # not the aliasing
    alpha = np.clip((1.0 - distance) / (edge / max(radius_x, radius_y)), 0.0, 1.0)
    alpha = alpha[:, :, None]

    tint = np.array(colour, dtype=float)[None, None, :]
    speckle = rng.normal(0.0, 5.0, image.shape)

    return image * (1.0 - alpha) + (tint + speckle) * alpha


# (centre x, centre y, radius x, radius y, angle, colour)
#
# The first group sits well inside a single chip. The second straddles a chip
# boundary at 200, 400, 600 or 704, which is where a chipped detector has to
# either merge two partial detections or filter them.
BLOBS = (
    (90, 110, 26, 18, 0.4, (232, 226, 210)),
    (300, 95, 15, 15, 0.0, (44, 52, 58)),
    (505, 150, 34, 21, 1.1, (208, 198, 176)),
    (760, 105, 12, 20, 0.3, (38, 60, 72)),
    (905, 210, 22, 22, 0.0, (226, 214, 198)),
    (140, 330, 19, 31, 2.0, (46, 44, 40)),
    (420, 300, 27, 27, 0.0, (240, 236, 228)),
    (640, 360, 16, 12, 0.8, (30, 48, 55)),
    (880, 430, 30, 19, 2.7, (214, 206, 190)),
    (110, 560, 23, 23, 0.0, (36, 40, 46)),
    (350, 620, 33, 20, 0.6, (236, 228, 214)),
    (560, 555, 14, 14, 0.0, (42, 58, 66)),
    (820, 660, 25, 34, 1.5, (222, 212, 196)),
    (250, 730, 18, 18, 0.0, (34, 46, 52)),
    (690, 745, 28, 17, 2.2, (230, 220, 204)),

    # Straddling a chip edge on purpose
    (200, 200, 24, 24, 0.0, (245, 240, 230)),
    (400, 480, 20, 30, 0.9, (28, 38, 44)),
    (600, 200, 30, 18, 1.8, (238, 230, 216)),
    (704, 600, 22, 22, 0.0, (32, 44, 50)),
    (200, 640, 26, 16, 0.5, (244, 238, 226)),
    (400, 704, 18, 26, 1.2, (30, 42, 48)),
)


def detect_scene(rng):
    """A 1000 by 800 scene with blobs a proposal network finds."""
    image = _background(rng)

    for centre_x, centre_y, radius_x, radius_y, angle, colour in BLOBS:
        image = _blob(image, centre_x, centre_y, radius_x, radius_y, angle,
                      colour, rng)

    return np.clip(image, 0, 255).astype(np.uint8)


def detect_scene_gray(rng):
    """The same scene in one channel, for the `gs_to_rgb` variants.

    Built from the colour one by the usual luma weights rather than generated
    again, so the two recordings can be read against each other.
    """
    colour = detect_scene(np.random.default_rng(SEED))

    luma = (0.299 * colour[:, :, 0]
            + 0.587 * colour[:, :, 1]
            + 0.114 * colour[:, :, 2])

    return np.clip(luma, 0, 255).astype(np.uint8)


# The top-left 900 by 750 of the scene: 675,000 pixels against the full
# scene's 800,000. With `chip_adaptive_thresh` set between the two, one is
# large enough for `adaptive` to chip and the other is not, which is what
# shows that the choice is made once and kept.
MEDIUM_WIDTH = 900
MEDIUM_HEIGHT = 750


def detect_scene_medium(rng):
    """The scene cropped below the adaptive threshold the cases use."""
    return detect_scene(np.random.default_rng(SEED))[
        :MEDIUM_HEIGHT, :MEDIUM_WIDTH]


def build():
    """Return {name: image} for the detection group's own fixtures."""
    rng = np.random.default_rng(SEED)

    return {
        "detect_scene": detect_scene(rng),
        "detect_scene_gray": detect_scene_gray(rng),
        "detect_scene_medium": detect_scene_medium(rng),
    }
