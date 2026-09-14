# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""The Kalman filter the ByteTrack-family trackers share.

ByteTrack, OC-SORT, DeepSORT and BoT-SORT each defined this class. P2-T06
found the four agreeing byte for byte -- means, covariances and projections,
over a sixteen-measurement sequence and two sets of process-noise weights --
and made them one. `library/object_trackers/tests/test_kalman.py` holds this
copy to a recording of those four. DeepSORT's `gating_distance`, which none of
the others had, is kept; BoT-SORT called its projection `_project`, only from
inside the class.
"""

import numpy as np
import scipy.linalg


class KalmanFilter:
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space:
        x, y, a, h, vx, vy, va, vh

    where (x, y) is the center of the bounding box, a is the aspect ratio,
    h is the height, and v* are the respective velocities.

    Object motion follows a constant velocity model.
    """

    def __init__(self, std_weight_position=1.0 / 20, std_weight_velocity=1.0 / 160):
        ndim = 4
        dt = 1.0

        # State transition matrix
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Observation matrix
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty weights. Defaults are ByteTrack's
        # originals; the bytetrack trainer fits both, so they are injectable
        # here the same way OCSORTTracker's KalmanFilter takes them.
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity

    def initiate(self, measurement):
        """Create track from unassociated measurement."""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step."""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot([
            self._motion_mat, covariance, self._motion_mat.T
        ]) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space."""
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot([
            self._update_mat, covariance, self._update_mat.T
        ])
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step."""
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot([
            kalman_gain, projected_cov, kalman_gain.T
        ])
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """Compute gating distance (Mahalanobis) between state and measurements."""
        projected_mean, projected_cov = self.project(mean, covariance)

        if only_position:
            projected_mean = projected_mean[:2]
            projected_cov = projected_cov[:2, :2]
            measurements = measurements[:, :2]

        chol_factor = np.linalg.cholesky(projected_cov)
        d = measurements - projected_mean
        z = scipy.linalg.solve_triangular(
            chol_factor, d.T, lower=True, check_finite=False
        )
        return np.sum(z * z, axis=0)
