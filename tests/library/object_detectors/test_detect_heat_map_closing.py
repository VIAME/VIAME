"""`detect_heat_map`'s `closing_radius`, which the port dropped.

`configs/pipelines/detector_motion_three_frame_diff.pipe` sets this key, and
for a while the implementation did not declare it -- so the process refused
the key and that pipeline could not be configured at all. Nothing caught it:
`pipe-check` resolves names, it does not construct a process and call
`_configure`, which is finding 2.78 and why this test exists.

So the first thing here is that the key is *accepted*. The second is that it
does something, because a parameter accepted and ignored would pass the
first check and still be wrong -- silently, which is worse than the refusal
was. The element is a disk of the given radius, settled against the v0.23.3
release rather than guessed; see finding 2.95.

Run just these:  ctest -R "unit:object_detectors:heat_map_closing"
"""

import numpy as np
import pytest


@pytest.fixture(scope="module", autouse=True)
def modules():
    from viame.modules import load_known_modules
    load_known_modules()


def detector(closing_radius=None):
    """A `detect_heat_map` configured to report every blob it can see."""
    from viame.algo import ImageObjectDetector
    from viame.config import empty_config

    algo = ImageObjectDetector.create("detect_heat_map")
    assert algo is not None, \
        "image_object_detector 'detect_heat_map' is not registered"

    cfg = empty_config()
    cfg.set_value("threshold", "10")
    cfg.set_value("min_area", "1")
    cfg.set_value("max_area", "1000000")
    cfg.set_value("min_fill_fraction", "0")
    cfg.set_value("class_name", "blob")

    if closing_radius is not None:
        cfg.set_value("closing_radius", str(closing_radius))

    assert algo.check_configuration(cfg), \
        "detect_heat_map refused this configuration"
    algo.set_configuration(cfg)
    return algo


def two_bars_with_a_gap():
    """Two bars two pixels apart, which a closing of radius 2 bridges."""
    from viame.types import Image, ImageContainer

    image = np.zeros((40, 40), dtype=np.uint8)
    image[10:30, 10:18] = 200
    image[10:30, 20:28] = 200
    return ImageContainer(Image(image))


def count(closing_radius=None):
    return len(detector(closing_radius).detect(two_bars_with_a_gap()))


def test_the_key_is_accepted():
    # The failure this guards is a refusal at configure time, not a wrong
    # answer: "not required or desired: closing_radius".
    cfg = detector(2).get_configuration()
    assert "closing_radius" in cfg.available_values()


def test_zero_is_the_default_and_leaves_the_gap():
    assert count() == count(0) == 2, \
        "two bars with a gap between them are two regions"


def test_a_closing_bridges_the_gap():
    # The point of the parameter. Accepted-and-ignored leaves this at two,
    # which is the failure mode worth holding the line on.
    assert count(2) == 1


@pytest.mark.parametrize("radius", [1, 2, 4])
def test_a_closing_never_loses_the_region(radius):
    assert count(radius) >= 1
