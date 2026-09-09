"""Deterministic input images for the golden recordings.

The fixtures are generated once and committed as PNGs beside the recordings,
so replaying a golden never depends on this file or on a JPEG decoder. It is
kept so that a new fixture set can be regenerated the same way.

Content is deliberately varied: a gradient so scaling and format conversion
show up, a disc and bars so morphology has edges to eat, seeded noise so
histogram, percentile and threshold paths see a real distribution, and a
moving disc across the sequence so temporal averaging has something to
average.
"""

import os

import numpy as np

WIDTH = 96
HEIGHT = 64
SEQUENCE_LENGTH = 12
SEED = 20260909


def _base(width=WIDTH, height=HEIGHT):
    """Gradient, bars and a disc, as float in [0, 1]."""
    ys, xs = np.mgrid[0:height, 0:width]

    image = 0.15 + 0.55 * (xs / (width - 1.0)) * (1.0 - 0.4 * ys / (height - 1.0))
    image += 0.18 * ((xs // 8) % 2 == 0)

    disc = (xs - width * 0.34) ** 2 + (ys - height * 0.55) ** 2
    image[disc < (min(width, height) * 0.22) ** 2] = 0.93

    image[height // 8:height // 8 + 3, :] = 0.02

    return np.clip(image, 0.0, 1.0)


def _noise(rng, shape, amount=0.06):
    return rng.normal(0.0, amount, size=shape)


def rgb8(rng):
    """8-bit, three channel, with the channels visibly different."""
    base = _base()
    channels = [
        base,
        np.clip(base * 0.72 + 0.14, 0.0, 1.0),
        np.clip(1.0 - base * 0.85, 0.0, 1.0),
    ]
    stacked = np.stack(channels, axis=-1) + _noise(rng, (HEIGHT, WIDTH, 3))
    return np.clip(stacked * 255.0, 0, 255).astype(np.uint8)


def gray8(rng):
    """8-bit, single channel."""
    image = _base() + _noise(rng, (HEIGHT, WIDTH))
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def gray16(rng):
    """16-bit, single channel, using the top of the range so that 16 to 8 bit
    conversion and percentile stretching have something to do."""
    image = _base() + _noise(rng, (HEIGHT, WIDTH), amount=0.03)
    return np.clip(image * 62000.0 + 1500.0, 0, 65535).astype(np.uint16)


def mask(rng):
    """A binary mask, which is the only input the morphology filter accepts.

    Shapes of several sizes with holes and specks, so erode and dilate at the
    radii the pipelines use each change something.
    """
    base = _base()
    image = base > 0.62

    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH]
    image |= ((xs - WIDTH * 0.75) ** 2 + (ys - HEIGHT * 0.3) ** 2) < 10 ** 2
    image &= ~(((xs - WIDTH * 0.75) ** 2 + (ys - HEIGHT * 0.3) ** 2) < 4 ** 2)

    specks = rng.random((HEIGHT, WIDTH)) < 0.01
    image |= specks

    return image


def sequence(rng):
    """A short 8-bit RGB sequence with a disc moving left to right.

    Temporal filters need more than one frame, and a moving object is what
    makes a windowed average differ from the frames it averages.
    """
    frames = []
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH]

    for index in range(SEQUENCE_LENGTH):
        base = _base()
        centre_x = WIDTH * (0.15 + 0.7 * index / (SEQUENCE_LENGTH - 1.0))
        disc = (xs - centre_x) ** 2 + (ys - HEIGHT * 0.35) ** 2
        base[disc < (HEIGHT * 0.14) ** 2] = 0.05

        channels = [
            base,
            np.clip(base * 0.72 + 0.14, 0.0, 1.0),
            np.clip(1.0 - base * 0.85, 0.0, 1.0),
        ]
        stacked = np.stack(channels, axis=-1) + _noise(rng, (HEIGHT, WIDTH, 3))
        frames.append(np.clip(stacked * 255.0, 0, 255).astype(np.uint8))

    return frames


# Frames the whole-pipeline recordings run on. They are downscaled copies of
# the fish sequence in pipelines_test_data, committed so that a replay needs
# neither that download nor a JPEG decoder, and small so that a dozen
# pipelines' worth of output frames stay a reasonable size in the repository.
PIPELINE_FRAME_COUNT = 6
PIPELINE_FRAME_SIZE = (480, 270)


def pipeline_frames(test_data_dir):
    """Return [(name, array)] for the whole-pipeline fixture frames."""
    from PIL import Image

    source_dir = os.path.join(test_data_dir, "images", "fish")
    frames = []

    for index in range(PIPELINE_FRAME_COUNT):
        name = "fish_1_seq_{:02d}.jpg".format(index + 1)
        path = os.path.join(source_dir, name)

        if not os.path.exists(path):
            raise FileNotFoundError(
                "pipeline fixtures need {}; download the pipelines test data "
                "first".format(path))

        image = Image.open(path).convert("RGB").resize(
            PIPELINE_FRAME_SIZE, Image.BILINEAR)
        frames.append(("frame_{:02d}".format(index), np.array(image)))

    return frames


def build():
    """Return {name: image} for the still fixtures and the sequence frames."""
    rng = np.random.default_rng(SEED)

    images = {
        "rgb8": rgb8(rng),
        "gray8": gray8(rng),
        "gray16": gray16(rng),
        "mask": mask(rng),
    }

    for index, frame in enumerate(sequence(rng)):
        images["seq_{:02d}".format(index)] = frame

    return images
