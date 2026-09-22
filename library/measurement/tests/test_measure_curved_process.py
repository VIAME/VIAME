"""Centerline transfer and curved measurement without viame or a stereo model."""
import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def module(monkeypatch):
    for name in ('viame', 'viame.processes', 'viame.measurement'):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    pipeline = types.ModuleType('viame.pipeline')
    pipeline.datum, pipeline.process = Mock(), Mock()
    process_base = types.ModuleType('viame.processes.base')
    process_base.ViameProcess = type('ViameProcess', (), {})
    kwtypes = types.ModuleType('viame.types')

    class Image:
        def __init__(self, a): self.a = a
        def asarray(self): return self.a

    class Container:
        def __init__(self, a): self.a = a
        def image(self): return self.a
    kwtypes.Image, kwtypes.ImageContainer = Image, Container
    monkeypatch.setitem(sys.modules, 'viame.pipeline', pipeline)
    monkeypatch.setitem(sys.modules, 'viame.processes', types.ModuleType('viame.processes'))
    monkeypatch.setitem(sys.modules, 'viame.processes.base', process_base)
    monkeypatch.setitem(sys.modules, 'viame.types', kwtypes)
    loaded = {}
    for name in ('curved_measurement', 'measure_curved_process'):
        spec = importlib.util.spec_from_file_location('viame.measurement.' + name, ROOT / 'library/measurement' / (name + '.py'))
        loaded[name] = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, loaded[name])
        spec.loader.exec_module(loaded[name])
    return loaded['measure_curved_process']


class IdentityGrid:
    """Raw images already form the rectified grid; disparity sampled bilinearly."""

    def prepare(self, width, height): pass
    def rectify_image(self, image, right): return image
    def rectify_points(self, points, right): return np.asarray(points, dtype=float).reshape(-1, 2)
    def unrectify_points(self, points, right): return np.asarray(points, dtype=float).reshape(-1, 2)
    def intrinsics(self): return dict(fx=100., fy=100., cx_left=100., cx_right=100., cy=100., baseline=1.)

    def match_grid_points(self, disparity, points):
        from scipy.ndimage import map_coordinates
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        d = map_coordinates(disparity, points.T[::-1], order=1, mode='constant', cval=np.nan)
        return np.column_stack([points[:, 0] - d, points[:, 1]])


class ConstantStereo:
    def __init__(self, forward, reverse=None):
        self.maps, self.calls = [forward, forward if reverse is None else reverse], 0

    def compute(self, left, right):
        d = self.maps[self.calls]
        self.calls += 1
        # Reverse inference sees flipped images; return the map in that frame.
        return Mock(image=lambda: Mock(asarray=lambda: d if self.calls == 1 else d[:, ::-1]))


def options(mode):
    return dict(mode=mode, samples=32, smoothing=0.5, consistency_px=1.5,
                centerline_tolerance_px=5.0, max_length_disagreement=0.1, max_depth_ratio=2.0)


def test_centerline_ordering_and_names(module):
    keypoints = {'tail': (5, 0), 'spine_010': (4, 0), 'head': (0, 0), 'spine_002': (2, 0), 'fin': (9, 9)}
    assert module.centerline_names(keypoints) == ['head', 'spine_002', 'spine_010', 'tail']
    np.testing.assert_array_equal(module.centerline_from_keypoints(keypoints),
                                  [[0, 0], [2, 0], [4, 0], [5, 0]])
    assert module.centerline_from_keypoints({'head': (0, 0)}) is None
    assert module.centerline_keypoint_names(4) == ['head', 'spine_001', 'spine_002', 'tail']


def test_transfer_requires_round_trip_agreement(module):
    forward = np.full((200, 200), 10.)
    reverse = np.full((200, 200), 10.)
    reverse[:, :80] = 20.
    grid = IdentityGrid()
    vertices = [[100., 50.], [150., 60.], [85., 70.]]
    matched = module.transfer_vertices(vertices, forward, None, grid, 1.5)
    np.testing.assert_allclose(matched, [[90, 50], [140, 60], [75, 70]])
    matched = module.transfer_vertices(vertices, forward, reverse, grid, 1.5)
    np.testing.assert_allclose(matched[:2], [[90, 50], [140, 60]])
    assert np.isnan(matched[2]).all()


def test_bidirectional_measurement_maps_vertices_both_ways(module):
    measurer = module.CurvedStereoMeasurer(
        ConstantStereo(np.full((200, 200), 10.)), IdentityGrid(), options('bidirectional'))
    measurer.set_frame(np.zeros((200, 200, 3), np.uint8), np.zeros((200, 200, 3), np.uint8))
    assert measurer.bidirectional and measurer._reverse.shape == (200, 200)
    result, mapped = measurer.measure([[80, 80], [110, 90], [140, 80]])
    assert result['success']
    # Spline smoothing rounds the apex, so the arc runs slightly past the polyline.
    assert 2 * np.hypot(3, 1) < result['curved_length'] < 1.05 * 2 * np.hypot(3, 1)
    assert result['straight_length'] == pytest.approx(6)
    assert result['relative_length_disagreement'] == pytest.approx(0)
    np.testing.assert_allclose(mapped, [[70, 80], [100, 90], [130, 80]])


def test_left_mode_keeps_existing_right_centerline(module):
    measurer = module.CurvedStereoMeasurer(
        ConstantStereo(np.full((200, 200), 10.)), IdentityGrid(), options('left'))
    measurer.set_frame(np.zeros((200, 200, 3), np.uint8), np.zeros((200, 200, 3), np.uint8))
    assert measurer._reverse is None
    result, _ = measurer.measure([[80, 80], [140, 80]], [[70, 80], [130, 80]])
    assert result['success'] and result['curved_length'] == pytest.approx(6)
    result, _ = measurer.measure([[80, 80], [140, 80]], [[70, 120], [130, 120]])
    assert not result['success']


def test_inconsistent_reverse_disparity_withholds_length(module):
    reverse = np.full((200, 200), 14.)
    measurer = module.CurvedStereoMeasurer(
        ConstantStereo(np.full((200, 200), 10.), reverse), IdentityGrid(), options('bidirectional'))
    measurer.set_frame(np.zeros((200, 200, 3), np.uint8), np.zeros((200, 200, 3), np.uint8))
    result, mapped = measurer.measure([[80, 80], [140, 80]])
    assert not result['success'] and mapped is None


class FakeBox:
    def __init__(self, x0, y0, x1, y1): self.b = (x0, y0, x1, y1)
    def min_x(self): return self.b[0]
    def min_y(self): return self.b[1]
    def max_x(self): return self.b[2]
    def max_y(self): return self.b[3]


class FakeDet:
    def __init__(self, box, mask=None, polygons=()):
        self.bounding_box, self.polygons = box, list(polygons)
        self.mask = None if mask is None else Mock(asarray=lambda: mask)
    def get_flattened_polygons(self): return self.polygons


def test_detection_mask_from_crop_and_polygons(module):
    crop = np.zeros((4, 6), np.uint8); crop[1:3, 2:5] = 1
    full = module.detection_mask(FakeDet(FakeBox(10.4, 20.6, 16, 24), crop), (30, 40))
    assert full.sum() == 6 and full[21:23, 12:15].all()
    poly = module.detection_mask(FakeDet(FakeBox(0, 0, 1, 1), polygons=[[5, 5, 15, 5, 15, 9, 5, 9]]), (20, 20))
    assert poly[7, 10] and not poly[12, 10]
    assert module.detection_mask(FakeDet(FakeBox(0, 0, 1, 1)), (20, 20)) is None


def bar():
    mask = np.zeros((60, 120), bool)
    mask[25:35, 10:110] = True
    return mask


def test_mask_polyline_follows_the_ridge_between_trusted_endpoints(module):
    pytest.importorskip('skimage')
    path = module.mask_polyline(bar(), [[12, 30], [108, 30]], 8, 4, True)
    assert path.shape == (8, 2)
    np.testing.assert_allclose(path[0], [12, 30]); np.testing.assert_allclose(path[-1], [108, 30])
    assert np.all(np.diff(path[:, 0]) > 0) and np.abs(path[:, 1] - 30).max() < 2


def test_mask_polyline_computes_ends_where_the_trunk_leaves_the_mask(module):
    pytest.importorskip('skimage')
    path = module.mask_polyline(bar(), None, 8, 4)
    assert path[0, 0] > 107 and path[-1, 0] < 12
    assert np.abs(path[:, 1] - 29.5).max() < 2


def test_mask_polyline_bridges_polygons_but_keeps_vertices_inside(module):
    pytest.importorskip('skimage')
    mask = bar()
    mask[:, 50:62] = False
    mask[5:8, 5:8] = True
    path = module.mask_polyline(mask, None, 9, 4)
    assert path[0, 0] > 107 and path[-1, 0] < 12
    assert mask[np.rint(path[1:-1, 1]).astype(int), np.rint(path[1:-1, 0]).astype(int)].all()


def test_mask_polyline_model_endpoints(module):
    pytest.importorskip('skimage')
    short = module.mask_polyline(bar(), [[40, 30], [70, 30]], 8, 4)
    assert short[0, 0] < 12 and short[-1, 0] > 107
    with pytest.raises(ValueError, match='off the mask'):
        module.mask_polyline(bar(), [[108, 30], [12, 55]], 8, 4)
