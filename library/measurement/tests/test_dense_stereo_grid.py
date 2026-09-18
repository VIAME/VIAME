"""The C++ rectified grid the interactive service shares with the pipelines."""
import json
import numpy as np
import pytest

measurement = pytest.importorskip('viame.core._measurement')
if not hasattr(measurement, 'DenseStereoGrid'):
    pytest.skip('built without OpenCV', allow_module_level=True)

K = dict(fx=1000., fy=1000., cx=640., cy=400.)


@pytest.fixture
def grid(tmp_path):
    # Right camera 300mm to the right, 100mm lower, yawed 2 degrees.
    c, s = np.cos(np.radians(2)), np.sin(np.radians(2))
    cal = dict(image_width=1280, image_height=800,
               fx_left=K['fx'], fy_left=K['fy'], cx_left=K['cx'], cy_left=K['cy'],
               k1_left=-0.1, k2_left=0.05, p1_left=0., p2_left=0., k3_left=0.,
               fx_right=K['fx'], fy_right=K['fy'], cx_right=K['cx'], cy_right=K['cy'],
               k1_right=0., k2_right=0., p1_right=0., p2_right=0., k3_right=0.,
               R=[c, 0., s, 0., 1., 0., -s, 0., c], T=[-300., -100., 0.])
    path = tmp_path / 'cal.json'
    path.write_text(json.dumps(cal))
    return measurement.DenseStereoGrid(str(path), 1280, 800, {
        'rectification_alpha': -1.0, 'refine_keypoints_disparity_window': 2,
        'refine_keypoints_disparity_percentile': 0.9})


def project(X, dist):
    x = X[:2] / X[2]
    r2 = x @ x
    x = x * (1 + dist[0] * r2 + dist[1] * r2 ** 2)
    return np.array([K['fx'] * x[0] + K['cx'], K['fy'] * x[1] + K['cy']])


def test_round_trip_and_epipolar_geometry(grid):
    assert grid.intrinsics()['baseline'] == pytest.approx(np.hypot(300., 100.))
    pts = np.array([[100., 50.], [640., 400.], [1200., 780.]])
    for right in (False, True):
        assert grid.unrectify_points(grid.rectify_points(pts, right), right) == pytest.approx(pts, abs=1e-3)
    c, s = np.cos(np.radians(2)), np.sin(np.radians(2))
    R = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    X = np.array([200., -50., 5000.])
    left = grid.rectify_points([project(X, [-0.1, 0.05])], False)[0]
    right = grid.rectify_points([project(R @ X + [-300., -100., 0.], [0., 0.])], True)[0]
    assert left[1] == pytest.approx(right[1], abs=1e-3)
    assert left[0] > right[0]


def test_silhouette_sampling_and_segment_fit(grid):
    disparity = np.full((800, 1280), 20., dtype=np.float32)
    left = np.array([[300., 400.], [500., 420.]])
    lg = grid.rectify_points(left, False)
    # The fish spans the segment at disparity 48; its edge pixel is blended.
    x0, x1 = int(lg[:, 0].min()) - 5, int(lg[:, 0].max()) + 5
    y0, y1 = int(lg[:, 1].min()) - 5, int(lg[:, 1].max()) + 5
    disparity[y0:y1, x0:x1] = 48.
    head = np.rint(lg[0]).astype(int)
    disparity[head[1], head[0]] = 28.
    matched = grid.match_grid_points(disparity, lg)
    assert lg[:, 0] - matched[:, 0] == pytest.approx([48., 48.])
    fitted = np.array(grid.fit_segment(disparity, left[0].tolist(), left[1].tolist()))
    rg = grid.rectify_points(fitted, True)
    assert lg[:, 0] - rg[:, 0] == pytest.approx([48., 48.], abs=0.05)
    assert grid.match_grid_points(np.zeros((800, 1280), np.float32), lg)[0].tolist() == pytest.approx([np.nan, np.nan], nan_ok=True)
