"""Multi-point measurement waits for its own frame's dense disparity."""
import importlib.util
from pathlib import Path
import sys
import threading
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
        spec = importlib.util.spec_from_file_location('viame.core.' + name,
                                                     ROOT / 'plugins/core' / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
    algo = Mock()
    algo.get_configuration().has_value.return_value = False
    result = module.InteractiveStereoService(algo)
    result._enabled = True
    result._current_left_path = 'left.mp4'
    result._current_right_path = 'right.mp4'
    result._current_frame_time = 0
    result._dense_measurement = lambda p, d, q, e: dict(
        length=float(np.linalg.norm(np.asarray(p) - q) / 10), stereo_rms=0)
    return result


def request():
    return dict(id='curve-1', command='measure_line',
                left_line=[[80, 80], [110, 110], [140, 80]],
                right_line=[[70, 80], [85, 95], [100, 110], [130, 80]],
                left_image_path='left.mp4', right_image_path='right.mp4', frame_time=0)


@pytest.mark.parametrize('changed_frame', [False, True])
@pytest.mark.parametrize('explicit_frame', [False, True])
def test_waits_without_blocking_and_rejects_a_different_video_frame(service, changed_frame, explicit_frame):
    received = threading.Event()
    responses = []

    def receive(response):
        responses.append(response)
        received.set()

    service._send_response = receive
    payload = request()
    if not explicit_frame:
        for name in ('left_image_path', 'right_image_path', 'frame_time'):
            payload.pop(name)
    assert service.handle_request(payload) is None
    assert not responses
    with service._compute_lock:
        service._current_disparity = np.full((200, 200), 10.)
        service._disparity_ready = True
        if changed_frame:
            service._current_frame_time = 1
        service._disparity_event.set()
    assert received.wait(3), 'Deferred measurement did not finish'
    assert len(responses) == 1
    response = responses[0]
    assert response['id'] == 'curve-1'
    if changed_frame:
        assert not response['success']
        assert 'current frame' in response['error']
        assert 'measurement' not in response
    else:
        assert response['success']
        assert response['measurement']['curved_length'] == pytest.approx(6 * np.sqrt(2))


def test_straight_measurement_does_not_wait_for_disparity(service):
    result = service.handle_request(dict(command='measure_line',
        left_line=[[80, 80], [140, 80]], right_line=[[70, 80], [130, 80]]))
    assert result['success']
    assert result['measurement']['length'] == pytest.approx(6)


def test_paired_vertices_measure_at_once_without_disparity(service):
    result = service.handle_request(dict(request(), right_line=[[70, 80], [100, 110], [130, 80]]))
    assert result['success']
    assert result['measurement']['curved_length'] == pytest.approx(6 * np.sqrt(2))
    assert result['measurement']['straight_length'] == pytest.approx(6)


def test_disparity_holes_fall_back_onto_the_right_curve(service):
    disparity = np.full((200, 200), 10.)
    disparity[90:130, :] = 0
    service._current_disparity = disparity
    service._disparity_ready = True
    result = service.handle_request(request())
    assert result['success']
    assert result['measurement']['curved_length'] == pytest.approx(6 * np.sqrt(2))
    matched = np.asarray(result['matched_points'])
    assert np.abs(matched[:, 0] - (np.asarray(result['sampled_points'])[:, 0] - 10)).max() < 1e-6


def test_rejects_stale_identity_even_when_disparity_is_ready(service):
    service._current_disparity = np.full((200, 200), 10.)
    service._disparity_ready = True
    service._current_frame_time = 1
    with pytest.raises(ValueError, match='current frame'):
        service.handle_request(request())
