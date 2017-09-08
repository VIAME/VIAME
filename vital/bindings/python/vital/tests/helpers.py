"""
ckwg +31
Copyright 2016 by Kitware, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither name of Kitware, Inc. nor the names of any contributors may be used
   to endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

==============================================================================

Helper functions for testing various Vital components

"""
import logging
import math
from six.moves import range

import numpy

from vital.types import (
    Camera,
    CameraIntrinsics,
    CameraMap,
    EigenArray,
    Feature,
    Landmark,
    LandmarkMap,
    Rotation,
    Track,
    TrackSet,
    TrackState,
)


def random_point_3d(stddev):
    pt = [[numpy.random.normal(0., stddev)],
          [numpy.random.normal(0., stddev)],
          [numpy.random.normal(0., stddev)]]
    return EigenArray.from_iterable(pt, target_shape=(3, 1))


def cube_corners(s, c=None):
    """
    Construct map of landmarks at the corners of a cube centered on ``c`` with a
    side length of ``s``.
    :rtype: LandmarkMap
    """
    if c is None:
        c = EigenArray.from_iterable([0, 0, 0])
    s /= 2.
    # Can add lists to numpy.ndarray types
    d = {
        0: Landmark(c + EigenArray.from_iterable([-s, -s, -s])),
        1: Landmark(c + EigenArray.from_iterable([-s, -s,  s])),
        2: Landmark(c + EigenArray.from_iterable([-s,  s, -s])),
        3: Landmark(c + EigenArray.from_iterable([-s,  s,  s])),
        4: Landmark(c + EigenArray.from_iterable([ s, -s, -s])),
        5: Landmark(c + EigenArray.from_iterable([ s, -s,  s])),
        6: Landmark(c + EigenArray.from_iterable([ s,  s, -s])),
        7: Landmark(c + EigenArray.from_iterable([ s,  s,  s])),
    }
    return LandmarkMap.from_dict(d)


def init_landmarks(num_lm, c=None):
    """
    construct map of landmarks will all locations at ``c`` with IDs in range
    ``[0, num_lm]``.
    """
    if c is None:
        c = EigenArray.from_iterable([0, 0, 0])
    d = {}
    for i in range(num_lm):
        d[i] = Landmark(loc=c)
    return LandmarkMap.from_dict(d)


def noisy_landmarks(landmark_map, stdev=1.0):
    """
    Add gausian noise to the landmark positions, returning a new landmark map.
    :type landmark_map: LandmarkMap
    :type stdev: float
    """
    d = landmark_map.as_dict()
    d_new = {}
    for i, l in d.iteritems():
        l2 = l.clone()
        l2.loc += random_point_3d(stdev)
        d_new[i] = l2
    return LandmarkMap.from_dict(d_new)


def camera_seq(num_cams=20, k=None):
    """
    Create a camera sequence (elliptical path)
    :param num_cams: Number of cameras. Default is 20
    :param k: Camera intrinsics to use for all created cameras. Default has
        focal length = 1000 and principle point of (640, 480).
    :return:
    """
    if k is None:
        k = CameraIntrinsics(1000, [640, 480])
    d = {}
    r = Rotation()  # identity
    for i in range(num_cams):
        frac = float(i) / num_cams
        x = 4 * math.cos(2*frac)
        y = 3 * math.sin(2*frac)
        d[i] = Camera([x, y, 2+frac], r, k).clone_look_at([0, 0, 0])

    return CameraMap(d)


def init_cameras(num_cams=20, intrinsics=None):
    """
    Initialize camera sequence with all cameras at the same location (0, 0, 1)
    and looking at origin.

    :param num_cams: Number of cameras to create, default 20.
    :param intrinsics: Intrinsics to use for all cameras.
    :return: Camera map of initialize cameras

    """
    if intrinsics is None:
        intrinsics = CameraIntrinsics(1000, (640, 480))
    r = Rotation()
    c = EigenArray.from_iterable((0, 0, 1))
    d = {}
    for i in range(num_cams):
        cam = Camera(c, r, intrinsics).clone_look_at([0, 0, 0],
                                                     [0, 1, 0])
        d[i] = cam
    return CameraMap(d)


def noisy_cameras(cam_map, pos_stddev=1., rot_stddev=1.):
    """
    Add positional and rotational gaussian noise to cameras
    :type cam_map: CameraMap
    :type pos_stddev: float
    :type rot_stddev: float
    :return: Camera map of new, noidy cameras'
    """
    cmap = {}
    for f, c in cam_map.as_dict().iteritems():
        c2 = Camera(
            c.center + random_point_3d(pos_stddev),
            c.rotation * Rotation.from_rodrigues(random_point_3d(rot_stddev)),
            c.intrinsics
        )
        cmap[f] = c2
    return CameraMap(cmap)


def projected_tracks(lmap, cmap):
    """
    Use the cameras to project the landmarks back into their images.
    :type lmap: LandmarkMap
    :type cmap: CameraMap
    """
    tracks = []

    cam_d = cmap.as_dict()
    landmark_d = lmap.as_dict()

    for lid, l in landmark_d.iteritems():
        t = Track(lid)
        tracks.append(t)

        # Sort camera iteration to make sure that we go in order of frame IDs
        for fid in sorted(cam_d):
            cam = cam_d[fid]
            f = Feature(cam.project(l.loc))
            t.append(TrackState(fid, f))

        assert t.size == len(cam_d)

    return TrackSet(tracks)


def subset_tracks(trackset, keep_fraction=0.75):
    """
    randomly drop a fraction of the track states per track in the given set,
    creating and returning new tracks in a new track-set.

    :type trackset: TrackSet
    :type keep_fraction: float
    """
    log = logging.getLogger(__name__)

    new_tracks = []
    for t in trackset.tracks():
        nt = Track(t.id)

        msg = 'track %d:' % t.id,
        for ts in t:
            if numpy.random.rand() < keep_fraction:
                nt.append(ts)
                msg += '.',
            else:
                msg += 'X',
        log.info(' '.join(msg))
        new_tracks.append(nt)
    return TrackSet(new_tracks)


def reprojection_error_vec(cam, lm, feat):
    """
    Compute the reprojection error vector of lm projected by cam compared to f
    :type cam: Camera
    :type lm: Landmark
    :type feat: Feature
    :rtype: EigenArray
    """
    pt = cam.project(lm.loc)
    return pt - feat.location


def reprojection_error_sqr(cam, lm, feat):
    """
    Compute the square reprojection error of lm projected by cam compared to f
    :type cam: Camera
    :type lm: Landmark
    :type feat: Feature
    :return: double error value
    :rtype: float
    """
    # Faster than squaring numpy.linalg.norm(..., 2) result
    return (reprojection_error_vec(cam, lm, feat) ** 2).sum()


def reprojection_rmse(cmap, lmap, tset):
    """
    Compute the Root-Mean-Square-Error (RMSE) of the re-projection.

    Implemented in Map-TK metrics.

    :type cmap: CameraMap
    :type lmap: LandmarkMap
    :type tset: TrackSet

    :return: Double error value
    :rtype float:

    """
    error_sum = 0.
    num_obs = 0

    for t in tset.tracks():
        lm = lmap.as_dict().get(t.id, None)
        if lm is None:
            # No landmark corresponding to this track, skip
            continue

        for ts in t:
            feat = ts.feature
            if feat is None:
                # No feature for this state
                continue

            cam = cmap.as_dict().get(ts.frame_id, None)
            if cam is None:
                # No camera corresponding to this track state
                continue

            error_sum += reprojection_error_sqr(cam, lm, feat)
            num_obs += 1

    return numpy.sqrt(error_sum / num_obs)
