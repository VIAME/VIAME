# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Tests for viame.core.alignment_core multi-pair fitting.

Covers:
1. fit_pooled_homography over synthetic correspondences from a known
   homography across several "frames", including a deliberately corrupted
   frame -- the pooled fit must recover the true homography and the bad
   frame must be flagged by its per-observation RMS / inlier count.
2. loop_closure_residual: ~0 for a mutually consistent synthetic triplet,
   rising when one pair is perturbed.

Pure geometry -- no model weights, no kwiver runtime.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Allow running from a source checkout, where the plugins directory is not
# installed as the viame.core package.
_PLUGIN_DIR = Path(__file__).resolve().parents[3] / "plugins" / "core"
try:
    from viame.core import alignment_core
except ImportError:
    sys.path.insert(0, str(_PLUGIN_DIR))
    import alignment_core


SIZE_A = (1920, 1080)
SIZE_B = (640, 512)

# A mild but non-trivial ground-truth homography A -> B: scale + shift with
# a touch of rotation and perspective, keeping A's frame within B's scale.
TRUE_H = np.array([
    [0.32, 0.02, 12.0],
    [-0.015, 0.31, 8.0],
    [1.5e-6, -1.0e-6, 1.0],
])


def project(H, pts):
    return cv2.perspectiveTransform(
        np.asarray(pts, np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)


def make_observation(rng, n_points=20, noise_px=0.3, offset_px=0.0):
    """Synthetic correspondences following TRUE_H, with optional b-side
    noise and a constant corruption offset."""
    a = np.column_stack([
        rng.uniform(50, SIZE_A[0] - 50, n_points),
        rng.uniform(50, SIZE_A[1] - 50, n_points),
    ])
    b = project(TRUE_H, a)
    b += rng.normal(0.0, noise_px, b.shape)
    if offset_px:
        b += rng.uniform(offset_px * 0.5, offset_px, b.shape) \
            * rng.choice([-1.0, 1.0], b.shape)
    return {
        "points": np.column_stack([a, b]).tolist(),
        "size_a": SIZE_A,
        "size_b": SIZE_B,
    }


class TestFitPooledHomography:
    def test_recovers_known_homography(self):
        rng = np.random.default_rng(7)
        observations = [make_observation(rng) for _ in range(3)]
        result = alignment_core.fit_pooled_homography(observations)
        assert result["success"], result
        H = np.array(result["homography"])
        # Compare via corner projection rather than matrix entries.
        corners = [[0, 0], [SIZE_A[0], 0], list(SIZE_A), [0, SIZE_A[1]]]
        err = np.linalg.norm(
            project(H, corners) - project(TRUE_H, corners), axis=1)
        assert err.max() < 2.0
        assert result["num_points"] == 60
        assert result["rms_px"] < 2.0
        assert len(result["observations"]) == 3

    def test_corrupted_frame_flagged_and_fit_survives(self):
        rng = np.random.default_rng(11)
        observations = [make_observation(rng) for _ in range(3)]
        observations.append(make_observation(rng, offset_px=80.0))
        result = alignment_core.fit_pooled_homography(observations)
        assert result["success"], result

        # The pooled fit must still recover TRUE_H despite the bad frame.
        H = np.array(result["homography"])
        corners = [[0, 0], [SIZE_A[0], 0], list(SIZE_A), [0, SIZE_A[1]]]
        err = np.linalg.norm(
            project(H, corners) - project(TRUE_H, corners), axis=1)
        assert err.max() < 3.0

        good_stats = result["observations"][:3]
        bad_stats = result["observations"][3]
        # The corrupted frame is visibly worse on both diagnostics.
        for good in good_stats:
            assert bad_stats["rms_px"] > 10 * good["rms_px"]
            assert bad_stats["inlier_ratio"] < 0.5 < good["inlier_ratio"]

    def test_empty_observation_reported_not_fatal(self):
        rng = np.random.default_rng(3)
        observations = [
            make_observation(rng),
            {"points": [], "size_a": SIZE_A, "size_b": SIZE_B},
            make_observation(rng),
        ]
        result = alignment_core.fit_pooled_homography(observations)
        assert result["success"], result
        assert result["observations"][1] == {
            "num_points": 0,
            "num_inliers": 0,
            "inlier_ratio": 0.0,
            "rms_px": None,
        }

    def test_too_few_points_fails_cleanly(self):
        result = alignment_core.fit_pooled_homography([
            {"points": [[0, 0, 0, 0]], "size_a": SIZE_A, "size_b": SIZE_B},
        ])
        assert not result["success"]
        assert result["code"] == "insufficient_points"


class TestLoopClosureResidual:
    H_12 = np.array([
        [0.9, 0.05, 30.0],
        [-0.04, 0.92, -12.0],
        [1.0e-6, 2.0e-6, 1.0],
    ])
    H_23 = np.array([
        [1.1, -0.03, -25.0],
        [0.02, 1.05, 18.0],
        [-1.5e-6, 1.0e-6, 1.0],
    ])

    def test_consistent_triplet_is_near_zero(self):
        H_13 = self.H_23 @ self.H_12
        residual = alignment_core.loop_closure_residual(
            self.H_12, self.H_23, H_13, SIZE_A)
        assert residual["mean_px"] < 1e-6
        assert residual["max_px"] < 1e-6

    def test_perturbed_pair_raises_residual(self):
        H_13 = self.H_23 @ self.H_12
        perturbed = self.H_23.copy()
        perturbed[0, 2] += 5.0  # 5 px translation error in one pair
        residual = alignment_core.loop_closure_residual(
            self.H_12, perturbed, H_13, SIZE_A)
        assert residual["mean_px"] > 1.0
        assert residual["max_px"] >= residual["mean_px"]
