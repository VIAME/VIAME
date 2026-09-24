"""A drawn box reaches the model as a box prompt and bounds the mask."""
import numpy as np
import pytest

kwiver = pytest.importorskip("kwiver.vital.types")
from viame.core import interactive_segmentation as seg  # noqa: E402
from viame.core.segmentation_utils import clip_mask_to_box  # noqa: E402


class FakeImage:
    def width(self):
        return 200

    def height(self):
        return 100


class OversizedModel:
    """Whatever the prompt, answers with a mask far larger than any box."""

    def __init__(self):
        self.calls = []

    def segment(self, image, points, labels):
        from kwiver.vital.types import (
            BoundingBoxD, DetectedObject, DetectedObjectSet, DetectedObjectType, ImageContainer, Image)
        self.calls.append(([(p.value[0], p.value[1]) for p in points], list(labels)))
        det = DetectedObject(BoundingBoxD(0, 0, 199, 99), 0.9, DetectedObjectType("object", 0.9))
        det.mask = ImageContainer(Image(np.ones((100, 200), dtype=np.uint8)))
        out = DetectedObjectSet()
        out.add(det)
        return out


def test_clip_mask_to_box_keeps_only_the_box():
    mask = np.ones((10, 10), dtype=bool)
    clipped = clip_mask_to_box(mask, (2.4, 3.6, 5.2, 6.1))
    ys, xs = np.where(clipped)
    assert (xs.min(), ys.min(), xs.max(), ys.max()) == (2, 3, 6, 7)


def test_box_prompt_is_passed_as_corners_and_bounds_the_mask():
    model = OversizedModel()
    service = seg.InteractiveSegmentationService(segment_via_points_algo=model)
    service._current_image_path = "img.png"
    service._current_image_container = FakeImage()
    response = service.handle_predict({
        "image_path": "img.png", "points": [], "point_labels": [], "box": [20, 10, 60, 40],
    })
    assert response["success"]
    assert model.calls == [([(20.0, 10.0), (60.0, 40.0)], [2, 3])]
    xs = [p[0] for p in response["polygon"]]
    ys = [p[1] for p in response["polygon"]]
    assert min(xs) >= 20 and max(xs) <= 61 and min(ys) >= 10 and max(ys) <= 41
