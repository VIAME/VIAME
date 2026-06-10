# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Epipolar template-matching stereo correspondence as a single ONNX graph.

This module reimplements, in pure PyTorch (and therefore exportable to a single
self-contained ``.onnx`` file), the "regular computer vision" stereo matching
method from ``plugins/core/measurement_utilities.cxx`` -- the one selected by
``epipolar_template_matching`` in ``measurement_from_annotations_template.pipe``.

The C++ pipeline does, per source keypoint:

  1. ``compute_epipolar_points``  -- unproject the source pixel into a ray, walk
     a range of depths, project each sampled 3D point into the target camera,
     producing a list of candidate pixels along the epipolar curve (works on
     *unrectified* images; handles full lens distortion).
  2. ``find_corresponding_point_epipolar_template_matching`` -- extract a square
     template around the source pixel and score every candidate with
     normalized cross correlation (OpenCV ``TM_CCOEFF_NORMED``), then take the
     best candidate subject to a score threshold and a uniqueness-ratio test.

Both steps are pure tensor math, so the whole thing fits in one ONNX graph.
Calibration is supplied as runtime tensor inputs (K / dist / R / t for each
camera), so a single exported model works for any rig and any calibration
format; see ``calibration_io.py`` for the host-side loaders.

Two graphs are exported (see ``export_stereo_mapping.py``):

  * ``EpipolarMatcher``  -- outputs matched right-image points + NCC scores.
    Triangulation is left to the host (``triangulate.py`` provides a bit-exact
    NumPy port of ``triangulate_fast_two_view``).
  * ``EpipolarMeasurer`` -- additionally triangulates each matched pair to a 3D
    point *inside the graph*, reproducing the C++ path as closely as ONNX
    allows (see ``triangulate.py`` for the one approximation: the final 4x4
    homogeneous null-space solve is replaced by an inhomogeneous normal-equations
    solve, since ONNX has no stable SVD operator).

Conventions (matching ``camera_rig_io.cxx``): the world frame is the left
camera, so ``P_cam = R @ P_world + t`` with the left camera at ``R = I, t = 0``
and the right camera at ``R, t`` (the rig's relative rotation / translation).
Distortion uses the vital radial-tangential model
``d = [k1, k2, p1, p2, k3, k4, k5, k6]`` (8 coeffs, zero-padded), unmapped with
the same 5-iteration Gauss-Newton scheme as ``simple_camera_intrinsics``.
"""

import torch
import torch.nn as nn

# Number of Gauss-Newton iterations for undistortion. Matches the fixed loop
# count in simple_camera_intrinsics::undistort (which also early-exits on
# convergence; we just always run the full count, which is harmless).
_UNDISTORT_ITERS = 5

# A score assigned to candidates that are invalid (behind the target camera or
# whose template window falls outside the image). Mirrors the -1.0 returned by
# score_template_at_point for out-of-bounds candidates.
_INVALID_SCORE = -1.0


# ---------------------------------------------------------------------------
# Intrinsics: distortion / undistortion, map / unmap
#
# These mirror simple_camera_intrinsics in
# packages/kwiver/vital/types/camera_intrinsics.cxx exactly.
# ---------------------------------------------------------------------------

def _radial_scale(r2, d):
    """Radial distortion scale (camera_intrinsics radial_distortion_scale)."""
    r4 = r2 * r2
    r6 = r2 * r4
    num = 1.0 + r2 * d[0] + r4 * d[1] + r6 * d[4]
    den = 1.0 + r2 * d[5] + r4 * d[6] + r6 * d[7]
    return num / den


def _radial_deriv(r2, d):
    """d(scale)/d(r2) (camera_intrinsics radial_distortion_deriv)."""
    r4 = r2 * r2
    r6 = r4 * r2
    base = d[0] + 2.0 * d[1] * r2 + 3.0 * d[4] * r4
    a1 = 1.0 / (d[5] * r2 + d[6] * r4 + d[7] * r6 + 1.0)
    a2 = d[5] + 2.0 * d[6] * r2 + 3.0 * d[7] * r4
    deriv = base - a2 * a1 * (d[0] * r2 + d[1] * r4 + d[4] * r6 + 1.0)
    return deriv * a1


def _distort_scale_offset(x, y, d):
    """Return (scale, off_x, off_y) per distortion_scale_offset()."""
    x2 = x * x
    y2 = y * y
    r2 = x2 + y2
    scale = _radial_scale(r2, d)
    two_xy = 2.0 * x * y
    # tangential: dx = p1*2xy + p2*(r2 + 2x2); dy = p1*(r2 + 2y2) + p2*2xy
    off_x = d[2] * two_xy + d[3] * (r2 + 2.0 * x2)
    off_y = d[2] * (r2 + 2.0 * y2) + d[3] * two_xy
    return scale, off_x, off_y


def distort(x, y, d):
    """Map normalized coords to distorted normalized coords (vital ::distort)."""
    scale, off_x, off_y = _distort_scale_offset(x, y, d)
    return scale * x + off_x, scale * y + off_y


def undistort(dx, dy, d):
    """
    Invert distortion via 5 Gauss-Newton iterations
    (simple_camera_intrinsics::undistort). The 2x2 Jacobian solve is done with
    an analytic inverse so it is ONNX-exportable.

    Note: the Jacobian below is a faithful transcription of vital's
    distortion_jacobian, including its quirk of only updating the (0,0) entry
    for the tangential terms. Because Gauss-Newton converges to the residual
    root regardless of small Jacobian inaccuracies, the 5-iteration fixed point
    matches vital's output.
    """
    x = dx
    y = dy
    for _ in range(_UNDISTORT_ITERS):
        x2 = x * x
        y2 = y * y
        xy = x * y
        r2 = x2 + y2
        scale = _radial_scale(r2, d)
        d_scale = 2.0 * _radial_deriv(r2, d)

        j00 = d_scale * x2 + scale
        j01 = d_scale * xy
        j10 = d_scale * xy
        j11 = d_scale * y2 + scale

        # tangential contribution (transcribed verbatim from vital)
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
        inv_det = 1.0 / det
        # J^{-1} @ residual
        step_x = inv_det * (j11 * res_x - j01 * res_y)
        step_y = inv_det * (-j10 * res_x + j00 * res_y)
        x = x - step_x
        y = y - step_y
    return x, y


def _intrinsic_params(K):
    """Extract (f, skew, ppx, ppy, fy) from a 3x3 intrinsics matrix."""
    f = K[0, 0]
    skew = K[0, 1]
    ppx = K[0, 2]
    fy = K[1, 1]
    ppy = K[1, 2]
    return f, skew, ppx, ppy, fy


def unmap(px, py, K, d):
    """
    Image pixel -> normalized (undistorted) coords (camera_intrinsics::unmap).
    px, py are tensors of matching shape; K is 3x3, d is the 8-vector.
    """
    f, skew, ppx, ppy, fy = _intrinsic_params(K)
    p0x = px - ppx
    p0y = py - ppy
    # aspect_ratio = f / fy  =>  y = p0y * aspect / f = p0y / fy
    y = p0y / fy
    x = (p0x - y * skew) / f
    return undistort(x, y, d)


def map_point(nx, ny, K, d):
    """
    Normalized coords -> image pixel (camera_intrinsics::map).
    """
    f, skew, ppx, ppy, fy = _intrinsic_params(K)
    dxn, dyn = distort(nx, ny, d)
    out_x = dxn * f + dyn * skew + ppx
    # pt.y * f / aspect_ratio + ppy  with aspect = f / fy  =>  pt.y * fy + ppy
    out_y = dyn * fy + ppy
    return out_x, out_y


# ---------------------------------------------------------------------------
# Patch gather + normalized cross correlation (TM_CCOEFF_NORMED)
# ---------------------------------------------------------------------------

def _round_half_up(v):
    """Match C++ static_cast<int>(v + 0.5) for the rounding of pixel centers."""
    return torch.floor(v + 0.5)


def _gather_patches(image, cx, cy, half):
    """
    Gather (2*half+1)x(2*half+1) patches from a [H, W] image centered at integer
    pixels (cx, cy). cx, cy are int64 tensors of arbitrary shape S; returns a
    tensor of shape S + (T, T). Coordinates are clamped into bounds; callers
    mask invalid centers separately.
    """
    H = image.shape[0]
    W = image.shape[1]
    offs = torch.arange(-half, half + 1, device=image.device, dtype=torch.int64)
    # Build full index grids of shape S + (T, T). Offset views carry only the
    # trailing (T, T) so any leading shape S (e.g. [P] or [P, N]) broadcasts.
    gy = cy.unsqueeze(-1).unsqueeze(-1) + offs.view(-1, 1)   # S + (T, 1)
    gx = cx.unsqueeze(-1).unsqueeze(-1) + offs.view(1, -1)   # S + (1, T)
    gy = gy.clamp(0, H - 1)
    gx = gx.clamp(0, W - 1)
    flat = image.reshape(-1)
    idx = gy * W + gx
    return flat[idx.reshape(-1)].reshape(idx.shape)


def _ncc(template, patches):
    """
    Zero-mean normalized cross correlation, matching OpenCV TM_CCOEFF_NORMED.

    template : [P, T, T]
    patches  : [P, N, T, T]
    returns  : [P, N] correlation scores in [-1, 1]
    """
    t = template.unsqueeze(1)  # [P, 1, T, T]
    t_mean = t.mean(dim=(-1, -2), keepdim=True)
    w_mean = patches.mean(dim=(-1, -2), keepdim=True)
    tc = t - t_mean
    wc = patches - w_mean
    num = (tc * wc).sum(dim=(-1, -2))
    t_energy = (tc * tc).sum(dim=(-1, -2))
    w_energy = (wc * wc).sum(dim=(-1, -2))
    den = torch.sqrt(t_energy * w_energy)
    return num / den.clamp_min(1e-12)


# ---------------------------------------------------------------------------
# Epipolar candidate generation (compute_epipolar_points)
# ---------------------------------------------------------------------------

def compute_epipolar_points(px, py, K_src, d_src, R_src, t_src,
                            K_tgt, d_tgt, R_tgt, t_tgt,
                            min_depth, max_depth, num_samples):
    """
    For each source pixel (px, py) (shape [P]) produce num_samples candidate
    pixels in the target image by sampling depths along the back-projected ray.

    Returns:
      proj   : [P, N, 2] candidate pixel coordinates in the target image
      valid  : [P, N]    bool, False where the sample is behind the target cam
      nx, ny : [P]       normalized (undistorted) source coords (reused for
                         triangulation so we do not recompute the unmap)
    """
    nx, ny = unmap(px, py, K_src, d_src)              # [P]
    # ray = normalize([nx, ny, 1])
    ray = torch.stack([nx, ny, torch.ones_like(nx)], dim=-1)  # [P, 3]
    ray = ray / ray.norm(dim=-1, keepdim=True)

    steps = torch.arange(num_samples, device=px.device, dtype=px.dtype)
    frac = steps / max(num_samples - 1, 1)
    depths = min_depth + frac * (max_depth - min_depth)        # [N]

    p_cam = ray.unsqueeze(1) * depths.view(1, num_samples, 1)  # [P, N, 3]
    # world = R_src^T @ (p_cam - t_src)
    p_world = torch.matmul(p_cam - t_src.view(1, 1, 3), R_src)  # (R^T) via right-mul
    # target cam = R_tgt @ world + t_tgt
    p_tgt = torch.matmul(p_world, R_tgt.t()) + t_tgt.view(1, 1, 3)

    z = p_tgt[..., 2]
    valid = z > 0.0
    z_safe = torch.where(valid, z, torch.ones_like(z))
    ntx = p_tgt[..., 0] / z_safe
    nty = p_tgt[..., 1] / z_safe
    ox, oy = map_point(ntx, nty, K_tgt, d_tgt)
    proj = torch.stack([ox, oy], dim=-1)
    return proj, valid, nx, ny


# ---------------------------------------------------------------------------
# The matcher module
# ---------------------------------------------------------------------------

class EpipolarMatcher(nn.Module):
    """
    Single-graph epipolar NCC matcher.

    forward inputs (all float32 unless noted):
      left_gray   : [H, W]   grayscale source (left) image
      right_gray  : [Hr, Wr] grayscale target (right) image
      points_left : [P, 2]   source keypoints (x, y) in the left image
      K_left, K_right        : [3, 3] intrinsics
      dist_left, dist_right  : [8]    radial-tangential coeffs (zero padded)
      R_left, R_right        : [3, 3] world->camera rotation
      t_left, t_right        : [3]    world->camera translation
      min_depth, max_depth   : scalar depth-sampling bounds (calibration units)

    forward outputs:
      right_points : [P, 2] best-match pixel in the right image (the float
                            epipolar-curve coordinate, as the C++ returns)
      best_score   : [P]    NCC score of the best candidate
      second_score : [P]    best NCC score outside a template_size neighborhood
                            (for the host-side uniqueness-ratio test)

    template_size and num_samples are fixed at export time (graph constants),
    mirroring the C++ config keys :template_size and :epipolar_num_samples.
    Score threshold and uniqueness ratio are intentionally *not* baked in: the
    host applies them to best_score/second_score so they can be tuned without
    re-exporting.
    """

    def __init__(self, template_size=25, num_samples=5000):
        super().__init__()
        assert template_size % 2 == 1, "template_size must be odd"
        self.template_size = int(template_size)
        self.half = self.template_size // 2
        self.num_samples = int(num_samples)

    def _match(self, left_gray, right_gray, points_left,
               K_left, dist_left, R_left, t_left,
               K_right, dist_right, R_right, t_right,
               min_depth, max_depth):
        """Shared matching core; returns (right_points, best, second, idx, nx, ny)."""
        half = self.half
        Hr = right_gray.shape[0]
        Wr = right_gray.shape[1]

        px = points_left[:, 0]
        py = points_left[:, 1]

        proj, valid, nx, ny = compute_epipolar_points(
            px, py, K_left, dist_left, R_left, t_left,
            K_right, dist_right, R_right, t_right,
            min_depth, max_depth, self.num_samples)              # proj [P, N, 2]

        # --- source template ---
        sx = _round_half_up(px).to(torch.int64)
        sy = _round_half_up(py).to(torch.int64)
        template = _gather_patches(left_gray, sx, sy, half)       # [P, T, T]

        # --- candidate patches ---
        cx = _round_half_up(proj[..., 0]).to(torch.int64)         # [P, N]
        cy = _round_half_up(proj[..., 1]).to(torch.int64)
        patches = _gather_patches(right_gray, cx, cy, half)       # [P, N, T, T]

        scores = _ncc(template, patches)                          # [P, N]

        # Mask candidates whose template window leaves the image, or that are
        # behind the camera. score_template_at_point requires the center to be
        # in [half, dim-1-half].
        in_x = (cx >= half) & (cx <= Wr - 1 - half)
        in_y = (cy >= half) & (cy <= Hr - 1 - half)
        ok = valid & in_x & in_y
        scores = torch.where(ok, scores, torch.full_like(scores, _INVALID_SCORE))

        best, idx = scores.max(dim=1)                             # [P]
        right_points = torch.gather(
            proj, 1, idx.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)  # [P, 2]

        # Second-best outside a template_size pixel neighborhood of the best
        # match. Mirrors the suppression radius used by the strip-NCC path and
        # the "min distinct distance" of the point-by-point path.
        dx = proj[..., 0] - right_points[:, 0:1]
        dy = proj[..., 1] - right_points[:, 1:2]
        far = (dx * dx + dy * dy) >= float(self.template_size * self.template_size)
        second_scores = torch.where(far, scores,
                                    torch.full_like(scores, _INVALID_SCORE))
        second, _ = second_scores.max(dim=1)
        return right_points, best, second, idx, nx, ny

    def forward(self, left_gray, right_gray, points_left,
                K_left, dist_left, R_left, t_left,
                K_right, dist_right, R_right, t_right,
                min_depth, max_depth):
        right_points, best, second, _, _, _ = self._match(
            left_gray, right_gray, points_left,
            K_left, dist_left, R_left, t_left,
            K_right, dist_right, R_right, t_right,
            min_depth, max_depth)
        return right_points, best, second


class EpipolarMeasurer(EpipolarMatcher):
    """
    EpipolarMatcher plus in-graph triangulation of each matched pair to a 3D
    point, reproducing triangulate_fast_two_view as closely as ONNX permits.

    Extra forward output:
      points_3d : [P, 3] triangulated 3D point per keypoint, in world (= left
                         camera) coordinates.

    See triangulate.py :: triangulate_fast_torch for the (single, documented)
    deviation from the C++ path.
    """

    def forward(self, left_gray, right_gray, points_left,
                K_left, dist_left, R_left, t_left,
                K_right, dist_right, R_right, t_right,
                min_depth, max_depth):
        from triangulate import triangulate_fast_torch

        right_points, best, second, _, nx_left, ny_left = self._match(
            left_gray, right_gray, points_left,
            K_left, dist_left, R_left, t_left,
            K_right, dist_right, R_right, t_right,
            min_depth, max_depth)

        # Normalized (undistorted) coords for the matched right points.
        nx_r, ny_r = unmap(right_points[:, 0], right_points[:, 1],
                           K_right, dist_right)
        points_3d = triangulate_fast_torch(
            nx_left, ny_left, nx_r, ny_r,
            R_left, t_left, R_right, t_right)
        return right_points, best, second, points_3d
