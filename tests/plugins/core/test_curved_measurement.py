# This file is part of VIAME, and is distributed under an OSI-approved
# BSD 3-Clause License. See the root LICENSE file for details.
import importlib.util
import json
from pathlib import Path
import sys
import types
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('curved_measurement', ROOT / 'plugins/core/curved_measurement.py')
cm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cm)
CAL = dict(rectified=True, fx=100, fy=100, cx_left=100, cx_right=100, cy=100, baseline=1)


def fixture_curve():
    t = np.linspace(-1, 1, 100)
    return np.column_stack([120 + 30 * t, 80 + 25 * t ** 2])


def test_straight_metric_length():
    result = cm.measure_curve([[80, 80], [140, 80]], np.full((200, 200), 10.), CAL)
    assert result['success']
    assert result['curved_length'] == pytest.approx(6)
    assert result['curvature_ratio'] == pytest.approx(1)


def test_parabolic_arc_and_bidirectional_agreement():
    curve = fixture_curve()
    disparity = np.full((200, 200), 10.)
    result = cm.measure_curve(curve, disparity, CAL, right_curve=curve - [10, 0],
                              right_disparity=disparity, mode='bidirectional', smoothing=0, samples=128)
    t = np.linspace(-1, 1, 10000)
    from scipy.integrate import trapezoid
    expected = trapezoid(np.sqrt(30 ** 2 + (50 * t) ** 2), t) / 10
    assert result['success']
    assert result['curved_length'] == pytest.approx(expected, rel=0.001)
    assert result['curved_length'] > result['straight_length']
    assert result['relative_length_disagreement'] < 1e-8


def test_unequal_intrinsics_principal_offset():
    cal = dict(CAL, fy=200, cx_right=98)
    result = cm.measure_curve([[100, 50], [100, 150]], np.full((200, 200), 12.), cal)
    assert result['curved_length'] == pytest.approx(5)


@pytest.mark.parametrize('value', [0., -1., np.nan, np.inf])
def test_invalid_disparity_does_not_bridge(value):
    disparity = np.full((200, 200), 10.)
    disparity[:, 105:125] = value
    result = cm.measure_curve([[80, 80], [140, 80]], disparity, CAL)
    assert not result['success']
    assert 'curved_length' not in result


def test_round_trip_rejection():
    result = cm.measure_curve(fixture_curve(), np.full((200, 200), 10.), CAL,
                              right_disparity=np.full((200, 200), 14.))
    assert not result['success']


def test_opposite_mask_and_curve_rejection():
    curve = fixture_curve()
    for kwargs in [dict(right_mask=np.zeros((200, 200))), dict(right_curve=curve + [0, 30])]:
        assert not cm.measure_curve(curve, np.full((200, 200), 10.), CAL, **kwargs)['success']


def test_rejects_unrectified_or_missing_reverse():
    with pytest.raises(ValueError, match='rectified'):
        cm.measure_curve(fixture_curve(), np.ones((200, 200)), dict(CAL, rectified=False))
    with pytest.raises(ValueError, match='independent'):
        cm.measure_curve(fixture_curve(), np.ones((200, 200)), CAL, mode='bidirectional')


def test_no_out_of_image_clamping():
    result = cm.measure_curve([[-1, 30], [100, 30]], np.full((200, 200), 10.), CAL)
    assert not result['success']


def test_mask_path_ignores_fin_branch():
    pytest.importorskip('skimage')
    mask = np.zeros((100, 150), bool)
    mask[45:56, 20:131] = True
    mask[20:50, 70:76] = True
    curve = cm.mask_centerline(mask, [[20, 50], [130, 50]])
    assert np.max(np.abs(curve[:, 1] - 50)) < 8
    np.testing.assert_equal(curve[[0, -1]], [[20, 50], [130, 50]])
    mask[:, 65:85] = False
    with pytest.raises(ValueError, match='connected'):
        cm.mask_centerline(mask, [[20, 50], [130, 50]])


def test_cli(tmp_path, monkeypatch):
    disp = tmp_path / 'disparity.npy'
    np.save(disp, np.full((200, 200), 2560, np.uint16))
    request = tmp_path / 'request.json'
    request.write_text(json.dumps(dict(left_curve=[[80, 80], [140, 80]],
        rectified_calibration=CAL, left_disparity_path=str(disp), disparity_scale=256)))
    output = tmp_path / 'out.json'
    monkeypatch.setattr(sys, 'argv', ['curved_measurement', str(request), '--output', str(output)])
    cm.main()
    assert json.loads(output.read_text())['curved_length'] == pytest.approx(6)


def test_service_routing_and_stale_frame(monkeypatch):
    # Only the unavailable compiled boundary is stubbed. The real service,
    # disparity geometry and request adapter execute below.
    core = types.ModuleType('viame.core')
    core._measurement = Mock()
    monkeypatch.setitem(sys.modules, 'viame', types.ModuleType('viame'))
    monkeypatch.setitem(sys.modules, 'viame.core', core)
    monkeypatch.setitem(sys.modules, 'viame.core.curved_measurement', cm)
    spec = importlib.util.spec_from_file_location('curve_service_test', ROOT / 'plugins/core/interactive_stereo.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    algo = Mock()
    algo.get_configuration().has_value.return_value = False
    service = module.InteractiveStereoService(algo)
    service._enabled = True
    service._disparity_ready = True
    service._current_disparity = np.full((200, 200), 10.)
    service._current_left_path, service._current_right_path = 'left.png', 'right.png'
    request = dict(command='measure_curve', left_image_path='left.png', right_image_path='right.png',
                   left_curve=[[80, 80], [140, 80]], rectified_calibration=CAL)
    assert service.handle_request(request)['curved_length'] == pytest.approx(6)
    service._dense_measurement = lambda p, d, q, e: dict(
        length=float(np.linalg.norm(np.asarray(p) - q) / 10), stereo_rms=0)
    curve_result = service.handle_request(dict(command='measure_line',
        left_line=[[80, 80], [110, 110], [140, 80]],
        right_line=[[70, 80], [85, 95], [100, 110], [130, 80]]))
    assert curve_result['measurement']['curved_length'] == pytest.approx(6 * np.sqrt(2))
    assert curve_result['measurement']['straight_length'] == pytest.approx(6)
    invalid = service.handle_request(dict(command='measure_line',
        left_line=[[80, 80], [110, 110], [140, 80]], right_line=[[70, 30], [130, 30]]))
    assert not invalid['success']
    assert 'measurement' not in invalid

    with pytest.raises(ValueError, match='current stereo frame'):
        service.handle_request(dict(request, left_image_path='old.png'))
    # Reverse mode must flip/swap input images and unflip the disparity output.
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
    left_image = np.tile(np.arange(200, dtype=np.uint8), (200, 1))
    right_image = left_image + 1
    service._load_image = lambda path, time: Container(Image(left_image if path == 'left.png' else right_image))
    def compute(left, right):
        np.testing.assert_equal(left.image().asarray(), right_image[:, ::-1])
        np.testing.assert_equal(right.image().asarray(), left_image[:, ::-1])
        return Container(Image(np.full((200, 200), 2560, np.uint16)))
    algo.compute.side_effect = compute
    result = service.handle_request(dict(request, right_curve=[[70, 80], [130, 80]],
                                         options=dict(mode='bidirectional')))
    assert result['success']
    assert result['curved_length'] == pytest.approx(6)


def test_bidirectional_length_disagreement_withholds_length():
    disparity = np.full((200, 200), 10.)
    result = cm.measure_curve([[80, 80], [140, 80]], disparity, CAL,
                              right_curve=[[74, 80], [126, 80]], right_disparity=disparity,
                              mode='bidirectional', centerline_tolerance_px=5)
    assert not result['success']
    assert 'curved_length' not in result
    assert 'disagree' in result['error']


def test_excessive_depth_range_withholds_length():
    disparity = np.full((200, 200), 10.)
    disparity[:, 120:] = 2
    result = cm.measure_curve([[80, 80], [140, 80]], disparity, CAL)
    assert not result['success']
    assert 'depth' in result['left']['error'].lower()


def test_subpixel_sample_cannot_interpolate_across_invalid_disparity():
    disparity = np.full((200, 200), 10.)
    disparity[:, 80] = 0
    result = cm.measure_curve([[80.5, 80], [140, 80]], disparity, CAL)
    assert not result['success']


def test_named_centerline_output_and_polyline_corners():
    points = [[80, 80], [110, 110], [140, 80]]
    samples = cm.resample_polyline(points)
    assert any(np.array_equal(p, points[1]) for p in samples)
    assert np.linalg.norm(np.diff(samples, axis=0), axis=1).sum() == pytest.approx(60 * np.sqrt(2))
    result = cm.measure_curve(points, np.full((200, 200), 10.), CAL)
    assert 'spine_001' in result['left_keypoints']
    np.testing.assert_equal(result['left_keypoints']['head'], points[0])
    np.testing.assert_equal(result['right_keypoints']['tail'], [130, 80])
