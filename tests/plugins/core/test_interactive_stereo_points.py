"""Strict bidirectional point transfer, without downloading a stereo model."""
import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import Mock
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]

@pytest.fixture
def service(monkeypatch):
    core = types.ModuleType('viame.core')
    core._measurement = Mock()
    monkeypatch.setitem(sys.modules, 'viame', types.ModuleType('viame'))
    monkeypatch.setitem(sys.modules, 'viame.core', core)
    for name in ('curved_measurement', 'interactive_stereo'):
        spec = importlib.util.spec_from_file_location('viame.core.' + name, ROOT / 'plugins/core' / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
    algo = Mock()
    algo.get_configuration().has_value.return_value = False
    result = module.InteractiveStereoService(algo)
    result._enabled = True
    result._disparity_ready = True
    result._current_left_path, result._current_right_path = 'left.png', 'right.png'
    result._current_disparity = np.full((200, 200), 10.)
    class Image:
        def __init__(self, a): self.a = a
        def asarray(self): return self.a
    class Container:
        def __init__(self, a): self.a = a
        def image(self): return self.a
    kwtypes = types.ModuleType('kwiver.vital.types')
    kwtypes.Image, kwtypes.ImageContainer = Image, Container
    monkeypatch.setitem(sys.modules, 'kwiver', types.ModuleType('kwiver'))
    monkeypatch.setitem(sys.modules, 'kwiver.vital', types.ModuleType('kwiver.vital'))
    monkeypatch.setitem(sys.modules, 'kwiver.vital.types', kwtypes)
    left = np.tile(np.arange(200, dtype=np.uint8), (200, 1))
    right = left + 1
    result._load_image = lambda path, time: Container(Image(left if path == 'left.png' else right))
    def compute(a, b):
        np.testing.assert_equal(a.image().asarray(), right[:, ::-1])
        np.testing.assert_equal(b.image().asarray(), left[:, ::-1])
        return Container(Image(np.full((200, 200), 2560, np.uint16)))
    algo.compute.side_effect = compute
    return result


def request(points, side='left', **extra):
    return dict(command='transfer_points', strict=True, source_camera=side, points=points,
                left_image_path='left.png', right_image_path='right.png', **extra)


def test_dense_directions_and_subpixel_precision(service):
    left = service.handle_request(request([[50.5, 60.25]]))
    assert left['success'] and left['valid_matches'] == [True]
    assert left['transferred_points'] == [[40.5, 60.25]]
    right = service.handle_request(request([[40.5, 60.25]], 'right'))
    assert right['transferred_points'] == [[50.5, 60.25]]
    assert right['disparity_values'] == [10]


def test_reverse_cache_is_per_frame(service):
    service.handle_request(request([[40, 60]], 'right'))
    service.handle_request(request([[45, 65]], 'right'))
    assert service._stereo_algo.compute.call_count == 1
    service._current_frame_time = 2
    service.handle_request(request([[40, 60]], 'right', frame_time=2))
    assert service._stereo_algo.compute.call_count == 2


@pytest.mark.parametrize('point', [[-1, 60], [5, 60], [200, 60], [50, 200]])
def test_invalid_locations_are_not_clamped(service, point):
    response = service.handle_request(request([point]))
    assert not response['success']
    assert response['valid_matches'] == [False]
    assert response['transferred_points'] == [None]


def test_zero_disparity_does_not_become_annotation(service):
    service._current_disparity[60, 50] = 0
    response = service.handle_request(request([[50, 60]]))
    assert not response['success']


def test_stale_frame_rejected(service):
    with pytest.raises(ValueError, match='current frame'):
        service.handle_request(request([[50, 60]], frame_time=1))


def test_nonfinite_input_rejected(service):
    with pytest.raises(ValueError, match='finite'):
        service.handle_request(request([[np.nan, 60]]))


def test_epipolar_reverse_inverts_camera_model_without_mutating_forward(service):
    class Matcher:
        _K_left = np.eye(3)
        _K_right = np.diag([2., 2., 1.])
        _R = np.eye(3)
        _T = np.array([-1., 0, 0])
        _dino_available = True
        def match_point(self, source, target, p):
            np.testing.assert_equal(self._T, [1, 0, 0])
            np.testing.assert_equal(self._K_left, np.diag([2., 2., 1.]))
            assert source[0, 0] == 2 and target[0, 0] == 1
            assert not self._dino_available
            return [p[0] + 7, p[1]]
    service._use_epipolar = True
    service._epipolar_matcher = Matcher()
    service._left_gray = np.ones((200, 200))
    service._right_gray = np.full((200, 200), 2)
    result = service.handle_request(request([[50, 60]], 'right'))
    assert result['transferred_points'] == [[57, 60]]
    np.testing.assert_equal(service._epipolar_matcher._T, [-1, 0, 0])
    assert service._epipolar_matcher._dino_available
