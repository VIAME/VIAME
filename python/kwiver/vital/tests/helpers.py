"""
ckwg +31
Copyright 2016-2019 by Kitware, Inc.
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

from kwiver.vital.types import (
    Camera,
    CameraIntrinsics,
    CameraMap,
    EigenArray,
    Feature,
    Landmark,
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


def create_numpy_image(dtype_name, nchannels, order='c'):
    if nchannels is None:
        shape = (5, 4)
    else:
        shape = (5, 4, nchannels)
    size = numpy.prod(shape)

    dtype = numpy.dtype(dtype_name)

    if dtype_name == 'bool':
        np_img = numpy.zeros(size, dtype=dtype).reshape(shape)
        np_img[0::2] = 1
    else:
        np_img = numpy.arange(size, dtype=dtype).reshape(shape)

    if order.startswith('c'):
        np_img = numpy.ascontiguousarray(np_img)
    elif order.startswith('fortran'):
        np_img = numpy.asfortranarray(np_img)
    else:
        raise KeyError(order)
    if order.endswith('-reverse'):
        np_img = np_img[::-1, ::-1]

    return np_img

def map_dtype_name_to_pixel_type(dtype_name):
    if dtype_name == 'float16':
        want = 'float16'
    if dtype_name == 'float32':
        want = 'float'
    elif dtype_name == 'float64':
        want = 'double'
    else:
        want = dtype_name
    return want
