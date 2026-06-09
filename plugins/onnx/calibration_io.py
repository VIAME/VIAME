# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Host-side calibration loaders for the epipolar ONNX matcher.

ONNX graphs cannot parse calibration files, so this module mirrors
``viame::read_stereo_rig`` (``plugins/core/camera_rig_io.cxx``) on the Python
side and produces the tensor inputs the graph expects. Every loader returns a
dict with these NumPy arrays (float64):

  K_left, K_right     : (3, 3) intrinsics
  dist_left, dist_right : (8,) radial-tangential coeffs [k1,k2,p1,p2,k3,k4,k5,k6]
  R_left, R_right     : (3, 3) world->camera rotation
  t_left, t_right     : (3,)   world->camera translation

Following camera_rig_io.cxx, the world frame is the left camera: the left
camera is at (R=I, t=0) and the right camera carries the rig's relative
rotation R and translation T (set_translation(T)).

Supported formats (dispatched by extension, matching read_stereo_rig):
  .npz   NumPy archive: R, T, cameraMatrixL/R, distCoeffsL/R
  .json  fx_left/.. cx_left/.. k1_left/.. plus R (9 or 3x3) and T
  .yml/.yaml  OpenCV FileStorage: M1,D1,M2,D2,R,T
  .mat   MATLAB Bouguet toolbox: om, T, fc_*, cc_*, kc_* (optionally in 'Cal')
  <dir>  directory with intrinsics.yml + extrinsics.yml
"""

import json
import os

import numpy as np


def _pad_dist(d):
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    out = np.zeros(8, dtype=np.float64)
    out[: min(8, d.size)] = d[: min(8, d.size)]
    return out


def _assemble(K_left, dist_left, K_right, dist_right, R, T):
    """Build the standard dict from intrinsics and the rig's relative R, T."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T = np.asarray(T, dtype=np.float64).reshape(3)
    return {
        "K_left": np.asarray(K_left, dtype=np.float64).reshape(3, 3),
        "dist_left": _pad_dist(dist_left),
        "R_left": np.eye(3, dtype=np.float64),
        "t_left": np.zeros(3, dtype=np.float64),
        "K_right": np.asarray(K_right, dtype=np.float64).reshape(3, 3),
        "dist_right": _pad_dist(dist_right),
        "R_right": R,
        "t_right": T,
    }


# ---------------------------------------------------------------------------
# .npz
# ---------------------------------------------------------------------------

def load_npz(path):
    z = np.load(path)

    def get(*names):
        for n in names:
            if n in z:
                return np.asarray(z[n], dtype=np.float64)
        return None

    K1 = get("cameraMatrixL", "cameraMatrix1", "M1")
    K2 = get("cameraMatrixR", "cameraMatrix2", "M2")
    R = get("R")
    T = get("T")
    if K1 is None or K2 is None or R is None or T is None:
        raise ValueError(
            "NPZ missing required arrays (R, T, cameraMatrixL, cameraMatrixR)")
    d1 = get("distCoeffsL", "distCoeffs1", "D1")
    d2 = get("distCoeffsR", "distCoeffs2", "D2")
    return _assemble(K1.reshape(3, 3), d1 if d1 is not None else [],
                     K2.reshape(3, 3), d2 if d2 is not None else [], R, T)


# ---------------------------------------------------------------------------
# .json
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, "r") as f:
        j = json.load(f)

    def K_for(side):
        return np.array([
            [j["fx_" + side], 0.0, j["cx_" + side]],
            [0.0, j["fy_" + side], j["cy_" + side]],
            [0.0, 0.0, 1.0],
        ])

    def dist_for(side):
        keys = ["k1_", "k2_", "p1_", "p2_", "k3_"]
        return [float(j[k + side]) for k in keys if (k + side) in j]

    return _assemble(K_for("left"), dist_for("left"),
                     K_for("right"), dist_for("right"),
                     j["R"], j["T"])


# ---------------------------------------------------------------------------
# OpenCV YAML (.yml/.yaml) and calibration directories
# ---------------------------------------------------------------------------

def _read_ocv_yaml(path, names):
    import cv2
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise ValueError("Could not open OpenCV YAML: " + path)
    out = {}
    for n in names:
        node = fs.getNode(n)
        out[n] = node.mat() if not node.empty() else None
    fs.release()
    return out


def _find(d, *names):
    for n in names:
        if d.get(n) is not None:
            return d[n]
    return None


def load_yaml(path):
    d = _read_ocv_yaml(path, ["M1", "cameraMatrixL", "cameraMatrix1",
                              "M2", "cameraMatrixR", "cameraMatrix2",
                              "D1", "distCoeffsL", "distCoeffs1",
                              "D2", "distCoeffsR", "distCoeffs2", "R", "T"])
    K1 = _find(d, "M1", "cameraMatrixL", "cameraMatrix1")
    K2 = _find(d, "M2", "cameraMatrixR", "cameraMatrix2")
    R = _find(d, "R")
    T = _find(d, "T")
    if K1 is None or K2 is None or R is None or T is None:
        raise ValueError("YAML missing required matrices (M1/M2/R/T): " + path)
    d1 = _find(d, "D1", "distCoeffsL", "distCoeffs1")
    d2 = _find(d, "D2", "distCoeffsR", "distCoeffs2")
    return _assemble(K1, [] if d1 is None else d1,
                     K2, [] if d2 is None else d2, R, T)


def load_ocv_dir(dir_path):
    intr = os.path.join(dir_path, "intrinsics.yml")
    extr = os.path.join(dir_path, "extrinsics.yml")
    di = _read_ocv_yaml(intr, ["M1", "M2", "D1", "D2"])
    de = _read_ocv_yaml(extr, ["R", "T"])
    if di.get("M1") is None or di.get("M2") is None:
        raise ValueError("intrinsics.yml missing M1/M2 in " + dir_path)
    return _assemble(di["M1"], di.get("D1") if di.get("D1") is not None else [],
                     di["M2"], di.get("D2") if di.get("D2") is not None else [],
                     de["R"], de["T"])


# ---------------------------------------------------------------------------
# MATLAB Bouguet (.mat)
# ---------------------------------------------------------------------------

def load_mat(path):
    import cv2
    from scipy.io import loadmat
    m = loadmat(path, squeeze_me=True, struct_as_record=False)
    if "Cal" in m:                # fields may be nested in a "Cal" struct
        cal = m["Cal"]
        get = lambda k: getattr(cal, k)
        has = lambda k: hasattr(cal, k)
    else:
        get = lambda k: m[k]
        has = lambda k: k in m

    def K_for(side):
        fc = np.asarray(get("fc_" + side)).reshape(-1)
        cc = np.asarray(get("cc_" + side)).reshape(-1)
        return np.array([[fc[0], 0.0, cc[0]],
                         [0.0, fc[1], cc[1]],
                         [0.0, 0.0, 1.0]])

    dist_l = get("kc_left") if has("kc_left") else []
    dist_r = get("kc_right") if has("kc_right") else []
    om = np.asarray(get("om"), dtype=np.float64).reshape(3)
    R = cv2.Rodrigues(om)[0]
    T = np.asarray(get("T"), dtype=np.float64).reshape(3)
    return _assemble(K_for("left"), dist_l, K_for("right"), dist_r, R, T)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def load_calibration(path):
    """Load any supported calibration format into the standard tensor dict."""
    if os.path.isdir(path):
        return load_ocv_dir(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        return load_npz(path)
    if ext == ".json":
        return load_json(path)
    if ext in (".yml", ".yaml"):
        return load_yaml(path)
    if ext == ".mat":
        return load_mat(path)
    raise ValueError("Unsupported calibration format: " + path)
