# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Two-view triangulation, in two flavors:

  * triangulate_fast_numpy -- a bit-exact NumPy port of
    kwiver::arrows::mvg::triangulate_fast_two_view (essential-matrix optimal
    correction + homogeneous DLT via SVD). Used host-side for the
    matching-only ONNX model so results are identical to the C++ measurer.

  * triangulate_fast_torch -- an ONNX-exportable Torch version used inside the
    EpipolarMeasurer graph. It reproduces the C++ path EXCEPT for the final
    homogeneous null-space solve: ONNX has no stable SVD operator. The SVD null
    vector of the 4x4 design matrix A is instead obtained as a column of the
    adjugate of M = A^T A: when M is (near-)rank-3 its adjugate is (near-)rank-1
    and every column is proportional to the smallest-eigenvector of M, i.e. the
    SVD null vector. This is closed-form (16 cofactor determinants) and matches
    the homogeneous DLT to floating-point precision for finite points. The
    closed-form Lindstrom optimal correction IS reproduced exactly (it needs no
    SVD). Everything runs in float32: because the result is recovered by
    hnormalizing a column of the adjugate (a ratio), the dynamic range of A^T A
    cancels and float32 matches float64 to ~1e-3 units (and avoids onnxruntime's
    missing float64 FusedMatMul kernel).

Both assume the world frame is the left camera (P_cam = R @ P_world + t), matching
camera_rig_io.cxx, and take normalized (undistorted) image coordinates as input.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Shared math: essential matrix from the relative pose of two cameras.
# Mirrors essential_matrix_from_cameras + essential_matrix_d (which normalizes
# the translation before forming E = [t]_x R).
# ---------------------------------------------------------------------------

def _essential_rt_numpy(R_left, t_left, R_right, t_right):
    R_e = R_right @ R_left.T
    t_e = t_right - R_e @ t_left
    t_e = t_e / np.linalg.norm(t_e)
    skew = np.array([[0.0, -t_e[2], t_e[1]],
                     [t_e[2], 0.0, -t_e[0]],
                     [-t_e[1], t_e[0], 0.0]])
    return skew @ R_e


# ===========================================================================
# NumPy (host) -- bit-exact port
# ===========================================================================

def _find_optimal_image_points_numpy(E, p1, p2):
    """Lindstrom 1-step correction; p1, p2 are normalized 2-vectors."""
    p1h = np.array([p1[0], p1[1], 1.0])
    p2h = np.array([p2[0], p2[1], 1.0])
    S = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    e_sub = E[:2, :2]

    l1 = S @ E @ p2h
    l2 = S @ E.T @ p1h

    a = float(l1 @ e_sub @ l2)
    b = (l1 @ l1 + l2 @ l2) / 2.0
    c = float(p1h @ E @ p2h)
    d = np.sqrt(b * b - a * c)

    lam = c / (b + d)
    l1 = l1 - e_sub @ (lam * l1)
    l2 = l2 - e_sub.T @ (lam * l2)
    lam = lam * (2.0 * d) / (l1 @ l1 + l2 @ l2)

    cp1 = p1h - S.T @ (lam * l1)
    cp2 = p2h - S.T @ (lam * l2)
    return cp1[:2] / cp1[2], cp2[:2] / cp2[2]


def _triangulate_dlt_numpy(pose0, pose1, p1, p2):
    A = np.empty((4, 4))
    A[0] = p1[0] * pose0[2] - pose0[0]
    A[1] = p1[1] * pose0[2] - pose0[1]
    A[2] = p2[0] * pose1[2] - pose1[0]
    A[3] = p2[1] * pose1[2] - pose1[1]
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def triangulate_fast_numpy(left_norm, right_norm,
                           R_left, t_left, R_right, t_right):
    """
    left_norm, right_norm : normalized (undistorted) 2D coords (length-2 arrays).
    Returns the triangulated 3D point in world (= left camera) coordinates.
    """
    E = _essential_rt_numpy(R_left, t_left, R_right, t_right)
    cp0, cp1 = _find_optimal_image_points_numpy(E, left_norm, right_norm)
    pose0 = np.hstack([R_left, t_left.reshape(3, 1)])
    pose1 = np.hstack([R_right, t_right.reshape(3, 1)])
    return _triangulate_dlt_numpy(pose0, pose1, cp0, cp1)


# ===========================================================================
# Torch (graph) -- ONNX exportable, vectorized over P points
# ===========================================================================

def _det3(m):
    """Determinant of a batch of 3x3 matrices m : [P, 3, 3] -> [P]."""
    return (m[:, 0, 0] * (m[:, 1, 1] * m[:, 2, 2] - m[:, 1, 2] * m[:, 2, 1])
            - m[:, 0, 1] * (m[:, 1, 0] * m[:, 2, 2] - m[:, 1, 2] * m[:, 2, 0])
            + m[:, 0, 2] * (m[:, 1, 0] * m[:, 2, 1] - m[:, 1, 1] * m[:, 2, 0]))


def _nullvector_4x4(M):
    """
    Smallest-eigenvector (SVD null vector) of a batch of symmetric 4x4 matrices
    M : [P, 4, 4], returned hnormalized as [P, 3].

    Uses the adjugate: adj(M)_{ji} = (-1)^(i+j) * minor(i, j). When M is
    (near-)rank-3, adj(M) is (near-)rank-1 and each column is proportional to
    the null vector. We take the column of largest norm for stability, then
    divide by its 4th component.
    """
    import torch
    P = M.shape[0]
    rows = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]  # rows kept for minor i
    adj = []
    for j in range(4):
        keep_cols = [c for c in range(4) if c != j]
        col_entries = []
        for i in range(4):
            sub = M[:, rows[i], :][:, :, keep_cols]      # [P, 3, 3] minor(i,j)
            sign = -1.0 if ((i + j) % 2) else 1.0
            col_entries.append(sign * _det3(sub))        # adj[j, i] = cofactor(i,j)
        adj.append(torch.stack(col_entries, dim=-1))     # column j -> [P, 4]
    adj = torch.stack(adj, dim=-1)                       # [P, 4, 4], columns

    col_norm = adj.norm(dim=1)                           # [P, 4]
    best = col_norm.argmax(dim=1)                        # [P]
    vec = torch.gather(adj, 2, best.view(-1, 1, 1).expand(-1, 4, 1)).squeeze(-1)
    return vec[:, :3] / vec[:, 3:4]


def triangulate_fast_torch(nx_l, ny_l, nx_r, ny_r,
                           R_left, t_left, R_right, t_right):
    """
    nx_*, ny_* : [P] normalized (undistorted) coords for left / right matches.
    R_*, t_*   : [3,3] / [3] world->camera pose.
    Returns points_3d : [P, 3] in world (= left camera) coordinates.
    """
    import torch
    one = torch.ones_like(nx_l)

    # Essential matrix (same for all points). [t]_x R, t normalized.
    R_e = torch.matmul(R_right, R_left.t())
    t_e = t_right - torch.matmul(R_e, t_left)
    t_e = t_e / t_e.norm()
    skew = torch.stack([
        torch.stack([torch.zeros_like(t_e[0]), -t_e[2], t_e[1]]),
        torch.stack([t_e[2], torch.zeros_like(t_e[0]), -t_e[0]]),
        torch.stack([-t_e[1], t_e[0], torch.zeros_like(t_e[0])]),
    ])
    E = torch.matmul(skew, R_e)
    e_sub = E[:2, :2]

    p1h = torch.stack([nx_l, ny_l, one], dim=-1)   # [P, 3]
    p2h = torch.stack([nx_r, ny_r, one], dim=-1)

    # l1 = (E p2h)[:2] ; l2 = (E^T p1h)[:2]
    l1 = torch.matmul(p2h, E.t())[:, :2]
    l2 = torch.matmul(p1h, E)[:, :2]

    a = (l1 * torch.matmul(l2, e_sub.t())).sum(-1)
    b = (l1.pow(2).sum(-1) + l2.pow(2).sum(-1)) / 2.0
    c = (p1h * torch.matmul(p2h, E.t())).sum(-1)
    d = torch.sqrt(torch.clamp(b * b - a * c, min=0.0))

    lam = c / (b + d)
    l1 = l1 - torch.matmul(lam.unsqueeze(-1) * l1, e_sub.t())
    l2 = l2 - torch.matmul(lam.unsqueeze(-1) * l2, e_sub)
    lam = lam * (2.0 * d) / (l1.pow(2).sum(-1) + l2.pow(2).sum(-1))

    cp0 = p1h[:, :2] - lam.unsqueeze(-1) * l1   # [P, 2]
    cp1 = p2h[:, :2] - lam.unsqueeze(-1) * l2

    # Inhomogeneous DLT. pose0 = [R_left|t_left], pose1 = [R_right|t_right].
    pose0 = torch.cat([R_left, t_left.unsqueeze(-1)], dim=1)   # [3, 4]
    pose1 = torch.cat([R_right, t_right.unsqueeze(-1)], dim=1)

    def design_rows(pt, pose):
        # rows: pt.x * pose[2] - pose[0] ; pt.y * pose[2] - pose[1]
        r0 = pt[:, 0:1] * pose[2].unsqueeze(0) - pose[0].unsqueeze(0)  # [P, 4]
        r1 = pt[:, 1:2] * pose[2].unsqueeze(0) - pose[1].unsqueeze(0)
        return r0, r1

    r0, r1 = design_rows(cp0, pose0)
    r2, r3 = design_rows(cp1, pose1)
    A = torch.stack([r0, r1, r2, r3], dim=1)      # [P, 4, 4]

    M = torch.matmul(A.transpose(1, 2), A)        # [P, 4, 4], symmetric
    return _nullvector_4x4(M)
