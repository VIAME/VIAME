# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""The shared tracker Kalman filter, held to what the four it replaced did.

ByteTrack, OC-SORT, DeepSORT and BoT-SORT each carried their own copy of the
same constant-velocity box filter. P2-T06 made them one,
`viame.object_trackers.common.kalman`, and this is the evidence that the one
is the four: `kalman_reference.json` was recorded from all four copies before
the change, over two sets of process-noise weights, and they agreed byte for
byte. None of these trackers is exercised by a CRITICAL example -- those run
SRNN -- so without this nothing would notice the filter drifting.
"""

import json
import os

import numpy as np
import pytest

from viame.object_trackers.common.kalman import KalmanFilter

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "kalman_reference.json")) as handle:
    REFERENCE = json.load(handle)


def _close(actual, expected):
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=0, atol=1e-12)


@pytest.mark.parametrize("run", REFERENCE["runs"],
                         ids=lambda r: "weights_%g_%g" % (r["std_weight_position"], r["std_weight_velocity"]))
def test_initiate_predict_project_update(run):
    measurements = REFERENCE["measurements"]
    kf = KalmanFilter(std_weight_position=run["std_weight_position"],
                      std_weight_velocity=run["std_weight_velocity"])

    mean, covariance = kf.initiate(np.array(measurements[0]))
    _close(mean, run["steps"][0]["mean"])
    _close(covariance, run["steps"][0]["covariance"])

    for measurement, step in zip(measurements[1:], run["steps"][1:]):
        mean, covariance = kf.predict(mean, covariance)
        projected_mean, projected_covariance = kf.project(mean, covariance)
        _close(projected_mean, step["projected_mean"])
        _close(projected_covariance, step["projected_covariance"])
        mean, covariance = kf.update(mean, covariance, np.array(measurement))
        _close(mean, step["mean"])
        _close(covariance, step["covariance"])


def test_gating_distance():
    gating = REFERENCE["gating_distance"]
    kf = KalmanFilter()
    mean, covariance = kf.initiate(np.array(REFERENCE["measurements"][0]))
    measurements = np.array(gating["measurements"])
    _close(kf.gating_distance(mean, covariance, measurements), gating["full"])
    _close(kf.gating_distance(mean, covariance, measurements, only_position=True),
           gating["position_only"])
