"""What the python `hough_circle` guarantees beyond the golden replay.

`tests/golden/test_golden.py` holds the detector to the recording made while
the implementation was C++: the same circles, in the same order, with the
same boxes. What is here is the part of the contract the recording cannot
reach.

`tests/baseline/registry.json` is that contract, and `compare_registry.py`
normally enforces it. It cannot for this name any more: the registry dump
cannot introspect a python implementation's config -- the pybind trampoline
returns the non-copyable config_block by copy -- so the entry carries an
`error` instead of keys and the comparison skips it. The same thing happened
to the video readers in P4-T05, and the same remedy applies.

Run just these:  ctest -R "unit:object_detectors"
"""

import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.abspath(
    os.path.join(HERE, "..", "..", "baseline", "registry.json"))


@pytest.fixture(scope="module", autouse=True)
def modules():
    from kwiver.vital.modules import load_known_modules
    load_known_modules()


def detector():
    from kwiver.vital.algo import ImageObjectDetector
    algo = ImageObjectDetector.create("hough_circle")
    assert algo is not None, "image_object_detector 'hough_circle' is not registered"
    return algo


def recorded_config():
    with open(REGISTRY) as handle:
        entry = json.load(handle)["algorithms"]["image_object_detector"]["hough_circle"]

    return {key: item["default"] for key, item in entry["config"].items()}


def test_the_name_keeps_its_config():
    cfg = detector().get_configuration()
    have = {key: cfg.get_value(key) for key in cfg.available_values()}

    for key, default in sorted(recorded_config().items()):
        assert key in have, "'hough_circle' lost config key '{}'".format(key)
        assert have[key] == default, (
            "'hough_circle' config key '{}' defaults to '{}', recorded "
            "'{}'".format(key, have[key], default))


def test_configuration_round_trips():
    """A value set is a value read back, in the spelling it was given."""
    from kwiver.vital.config import empty_config

    algo = detector()
    cfg = empty_config()
    cfg.set_value("dp", "2")
    cfg.set_value("min_dist", "5.5")
    cfg.set_value("param1", "80")
    cfg.set_value("param2", "15")
    cfg.set_value("min_radius", "4")
    cfg.set_value("max_radius", "12")
    algo.set_configuration(cfg)

    out = algo.get_configuration()
    assert out.get_value("dp") == "2"
    assert out.get_value("min_dist") == "5.5"
    assert out.get_value("param1") == "80"
    assert out.get_value("param2") == "15"
    assert out.get_value("min_radius") == "4"
    assert out.get_value("max_radius") == "12"


def circle_image(width=64, height=64, centre=(32, 32), radius=12, depth=3):
    """A filled white disc on black, which the transform finds easily."""
    yy, xx = np.mgrid[0:height, 0:width]
    inside = ((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2) <= radius * radius

    image = np.zeros((height, width, depth), dtype=np.uint8)
    for plane in range(depth):
        image[:, :, plane] = inside * 255

    return image


def container(array):
    from kwiver.vital.types import Image, ImageContainer
    return ImageContainer(Image(array))


def configured(**values):
    from kwiver.vital.config import empty_config

    algo = detector()
    cfg = empty_config()
    for key, value in values.items():
        cfg.set_value(key, str(value))
    algo.set_configuration(cfg)
    return algo


def test_finds_a_disc_and_boxes_it_by_the_radius():
    algo = configured(dp=1, min_dist=10, param1=200, param2=20,
                      min_radius=8, max_radius=16)

    found = algo.detect(container(circle_image()))
    assert len(found) >= 1

    box = list(found)[0].bounding_box
    # The box is the centre plus and minus the radius, so it is square and
    # its centre is the circle's. The transform is not exact on a rasterised
    # disc, so this checks the shape rather than the pixel.
    assert box.width() == pytest.approx(box.height(), abs=1e-6)
    assert box.center()[0] == pytest.approx(32, abs=3)
    assert box.center()[1] == pytest.approx(32, abs=3)
    assert box.width() / 2 == pytest.approx(12, abs=4)


def test_every_detection_is_a_circle_with_confidence_one():
    algo = configured(dp=1, min_dist=10, param1=200, param2=20,
                      min_radius=8, max_radius=16)

    for detection in algo.detect(container(circle_image())):
        assert detection.confidence == 1.0
        assert detection.type.score("circle") == 1.0


def test_a_grey_image_is_detected_the_same_as_its_rgb_form():
    """The C++ went through a BGR mat and BGR2GRAY; a grey image took the
    same path. A one-plane image must not be refused or read differently."""
    algo = configured(dp=1, min_dist=10, param1=200, param2=20,
                      min_radius=8, max_radius=16)

    rgb = circle_image()
    grey = rgb[:, :, :1].copy()

    from_rgb = [d.bounding_box for d in algo.detect(container(rgb))]
    from_grey = [d.bounding_box for d in algo.detect(container(grey))]

    assert len(from_rgb) == len(from_grey)
    for a, b in zip(from_rgb, from_grey):
        assert a.min_x() == pytest.approx(b.min_x(), abs=1e-6)
        assert a.min_y() == pytest.approx(b.min_y(), abs=1e-6)


def test_an_empty_image_detects_nothing():
    algo = configured(dp=1, min_dist=10, param1=200, param2=20,
                      min_radius=8, max_radius=16)

    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    assert len(algo.detect(container(blank))) == 0
