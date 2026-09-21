"""All components of a fish contribute to its keypoint mask."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    'segmentation_utils', ROOT / 'plugins/core/segmentation_utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


def square(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def components():
    return [
        {'exterior': square(10, 20, 20, 30), 'holes': [square(12, 22, 18, 28)]},
        {'exterior': square(25, 20, 35, 30), 'holes': []},
        {'exterior': square(14, 24, 16, 26), 'holes': []},
    ]


def test_rasterizes_disconnected_components_holes_and_nested_island():
    mask, bounds = utils.polygons_to_mask(components())
    assert bounds == (10, 20, 35, 30)
    assert mask[1, 1] == 255
    assert mask[3, 3] == 0  # Hole.
    assert mask[5, 5] == 255  # Separate island inside the hole.
    assert mask[5, 12] == 0  # Gap between components stays empty.
    assert mask[5, 20] == 255  # Second disconnected component.


def test_passes_one_combined_mask_to_keypoint_algorithm(monkeypatch):
    captured = []

    class Detection:
        def __init__(self, bbox, confidence, kind, mask):
            captured.append((bbox, mask))

    monkeypatch.setitem(sys.modules, 'kwiver.vital.types', SimpleNamespace(
        DetectedObject=Detection, DetectedObjectSet=list,
        BoundingBoxD=lambda *bounds: bounds,
        ImageContainer=lambda image: image, Image=lambda array: array,
    ))
    result = SimpleNamespace(keypoints={
        'head': SimpleNamespace(value=[11, 25]),
        'tail': SimpleNamespace(value=[34, 25]),
    })
    monkeypatch.setattr(utils, 'polygon_keypoint_algo', lambda: SimpleNamespace(
        refine=lambda image, detections: [result]))
    assert utils.polygons_to_keypoints(components()) == ([11.0, 25.0], [34.0, 25.0])
    assert len(captured) == 1
    expected, bounds = utils.polygons_to_mask(components())
    assert captured[0][0] == bounds
    np.testing.assert_array_equal(captured[0][1], expected)


def test_legacy_polygon_uses_the_same_extraction(monkeypatch):
    captured = []
    monkeypatch.setattr(utils, 'polygons_to_keypoints', lambda polygons: captured.append(polygons))
    polygon = square(0, 0, 10, 10)
    utils.polygon_to_keypoints(polygon)
    assert captured == [[{'exterior': polygon, 'holes': []}]]


def test_rejects_invalid_rings():
    with pytest.raises(ValueError, match='finite'):
        utils.polygons_to_mask([{'exterior': [[0, 0], [1, 1], [float('nan'), 2]]}])
    assert utils.polygons_to_mask([]) is None
