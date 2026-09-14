# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
NumPy port of the vital camera intrinsics map/unmap and perspective projection,
for host-side use by the ONNX runner (normalizing keypoints before
triangulation, and computing reprojection RMS exactly as
compute_stereo_measurement does). The math mirrors
packages/kwiver/vital/types/camera_intrinsics.cxx and the Torch version in
epipolar_matcher.py.
"""

import numpy as np

_UNDISTORT_ITERS = 5


def _radial_scale(r2, d):
    r4 = r2 * r2
    r6 = r2 * r4
    num = 1.0 + r2 * d[0] + r4 * d[1] + r6 * d[4]
    den = 1.0 + r2 * d[5] + r4 * d[6] + r6 * d[7]
    return num / den


def _radial_deriv(r2, d):
    r4 = r2 * r2
    r6 = r4 * r2
    base = d[0] + 2.0 * d[1] * r2 + 3.0 * d[4] * r4
    a1 = 1.0 / (d[5] * r2 + d[6] * r4 + d[7] * r6 + 1.0)
    a2 = d[5] + 2.0 * d[6] * r2 + 3.0 * d[7] * r4
    return (base - a2 * a1 * (d[0] * r2 + d[1] * r4 + d[4] * r6 + 1.0)) * a1


def _distort_scale_offset(x, y, d):
    x2, y2 = x * x, y * y
    r2 = x2 + y2
    scale = _radial_scale(r2, d)
    two_xy = 2.0 * x * y
    off_x = d[2] * two_xy + d[3] * (r2 + 2.0 * x2)
    off_y = d[2] * (r2 + 2.0 * y2) + d[3] * two_xy
    return scale, off_x, off_y


def distort(x, y, d):
    scale, off_x, off_y = _distort_scale_offset(x, y, d)
    return scale * x + off_x, scale * y + off_y


def undistort(dx, dy, d):
    x, y = dx, dy
    for _ in range(_UNDISTORT_ITERS):
        x2, y2, xy = x * x, y * y, x * y
        r2 = x2 + y2
        scale = _radial_scale(r2, d)
        d_scale = 2.0 * _radial_deriv(r2, d)
        j00 = d_scale * x2 + scale
        j01 = d_scale * xy
        j10 = d_scale * xy
        j11 = d_scale * y2 + scale
        axy = 2.0 * (d[2] * x + d[3] * y)
        ay = 2.0 * d[2] * y
        ax = 2.0 * d[3] * x
        j00 = j00 + (ay + 3.0 * ax) + (3.0 * ay + ax)
        j01 = j01 + axy
        j10 = j10 + axy
        scale_r, off_x, off_y = _distort_scale_offset(x, y, d)
        res_x = x * scale_r + off_x - dx
        res_y = y * scale_r + off_y - dy
        det = j00 * j11 - j01 * j10
        inv = 1.0 / det
        x = x - inv * (j11 * res_x - j01 * res_y)
        y = y - inv * (-j10 * res_x + j00 * res_y)
    return x, y


def unmap(px, py, K, d):
    f, skew, ppx, fy, ppy = K[0, 0], K[0, 1], K[0, 2], K[1, 1], K[1, 2]
    p0x, p0y = px - ppx, py - ppy
    y = p0y / fy
    x = (p0x - y * skew) / f
    return undistort(x, y, d)


def map_point(nx, ny, K, d):
    f, skew, ppx, fy, ppy = K[0, 0], K[0, 1], K[0, 2], K[1, 1], K[1, 2]
    dxn, dyn = distort(nx, ny, d)
    return dxn * f + dyn * skew + ppx, dyn * fy + ppy


def project(point_3d, K, d, R, t):
    """Project a world 3D point to a pixel (simple_camera_perspective::project)."""
    pc = R @ np.asarray(point_3d, dtype=np.float64).reshape(3) + t
    return np.array(map_point(pc[0] / pc[2], pc[1] / pc[2], K, d))
