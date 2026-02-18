#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #


"""
Unified stereo camera calibration format conversion tool.

Supports conversion between:
  - npz: NumPy archive format
  - json: KWIVER camera_rig_io compatible JSON format
  - opencv: OpenCV FileStorage YML format (intrinsics.yml + extrinsics.yml)
  - zed: ZED camera INI-style configuration format
  - cal: CamCAL/PtsCAL binary calibration format
"""

import argparse
import configparser
import json
import math
import os
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


# =============================================================================
# Internal calibration data representation
# =============================================================================

class StereoCalibration:
    """Internal representation of stereo calibration data."""

    def __init__(self):
        # Intrinsics
        self.camera_matrix_left = None   # 3x3 camera matrix
        self.camera_matrix_right = None  # 3x3 camera matrix
        self.dist_coeffs_left = None     # distortion coefficients
        self.dist_coeffs_right = None    # distortion coefficients

        # Extrinsics
        self.R = None  # 3x3 rotation matrix (right camera relative to left)
        self.T = None  # 3x1 translation vector

        # Rectification (optional, can be computed from above)
        self.R1 = None  # 3x3 rectification transform for left camera
        self.R2 = None  # 3x3 rectification transform for right camera
        self.P1 = None  # 3x4 projection matrix for left camera
        self.P2 = None  # 3x4 projection matrix for right camera
        self.Q = None   # 4x4 disparity-to-depth mapping matrix

        # Metadata (optional)
        self.image_width = None
        self.image_height = None
        self.grid_width = None
        self.grid_height = None
        self.square_size_mm = None
        self.rms_error_left = None
        self.rms_error_right = None
        self.rms_error_stereo = None

    def has_intrinsics(self):
        """Check if intrinsic parameters are available."""
        return (self.camera_matrix_left is not None and
                self.camera_matrix_right is not None and
                self.dist_coeffs_left is not None and
                self.dist_coeffs_right is not None)

    def has_extrinsics(self):
        """Check if extrinsic parameters (R, T) are available."""
        return self.R is not None and self.T is not None

    def has_rectification(self):
        """Check if rectification parameters are available."""
        return all(x is not None for x in [self.R1, self.R2, self.P1, self.P2, self.Q])

    def compute_rectification(self, image_width=None, image_height=None):
        """Compute rectification parameters from intrinsics and extrinsics."""
        if not self.has_intrinsics() or not self.has_extrinsics():
            raise ValueError("Cannot compute rectification without intrinsics and extrinsics")

        width = image_width or self.image_width
        height = image_height or self.image_height

        if width is None or height is None:
            raise ValueError("Image dimensions required to compute rectification")

        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            cameraMatrix1=self.camera_matrix_left,
            distCoeffs1=self.dist_coeffs_left,
            cameraMatrix2=self.camera_matrix_right,
            distCoeffs2=self.dist_coeffs_right,
            imageSize=(width, height),
            R=self.R,
            T=self.T,
            flags=cv2.CALIB_ZERO_DISPARITY
        )

    def validate(self, require_extrinsics=True):
        """Validate that minimum required data is present."""
        if not self.has_intrinsics():
            raise ValueError("Missing intrinsic parameters")
        if require_extrinsics and not self.has_extrinsics():
            raise ValueError("Missing extrinsic parameters (R, T)")


# =============================================================================
# Format readers
# =============================================================================

def read_npz(input_path):
    """Read calibration from NumPy NPZ format."""
    calib = StereoCalibration()
    data = np.load(input_path)

    # Intrinsics
    calib.camera_matrix_left = data['cameraMatrixL']
    calib.dist_coeffs_left = data['distCoeffsL']
    calib.camera_matrix_right = data['cameraMatrixR']
    calib.dist_coeffs_right = data['distCoeffsR']

    # Extrinsics
    calib.R = data['R']
    calib.T = data['T']

    # Rectification (optional)
    if all(key in data for key in ['R1', 'R2', 'P1', 'P2', 'Q']):
        calib.R1 = data['R1']
        calib.R2 = data['R2']
        calib.P1 = data['P1']
        calib.P2 = data['P2']
        calib.Q = data['Q']

    return calib


def read_json(input_path):
    """Read calibration from KWIVER-compatible JSON format."""
    calib = StereoCalibration()

    with open(input_path, 'r') as f:
        data = json.load(f)

    # Metadata
    calib.image_width = data.get('image_width')
    calib.image_height = data.get('image_height')
    calib.grid_width = data.get('grid_width')
    calib.grid_height = data.get('grid_height')
    calib.square_size_mm = data.get('square_size_mm')
    calib.rms_error_left = data.get('rms_error_left')
    calib.rms_error_right = data.get('rms_error_right')
    calib.rms_error_stereo = data.get('rms_error_stereo')

    # Intrinsics - left
    calib.camera_matrix_left = np.array([
        [data['fx_left'], data.get('skew_left', 0), data['cx_left']],
        [0, data['fy_left'], data['cy_left']],
        [0, 0, 1]
    ], dtype=np.float64)

    calib.dist_coeffs_left = np.array([[
        data['k1_left'],
        data['k2_left'],
        data['p1_left'],
        data['p2_left'],
        data.get('k3_left', 0.0)
    ]], dtype=np.float64)

    # Intrinsics - right
    calib.camera_matrix_right = np.array([
        [data['fx_right'], data.get('skew_right', 0), data['cx_right']],
        [0, data['fy_right'], data['cy_right']],
        [0, 0, 1]
    ], dtype=np.float64)

    calib.dist_coeffs_right = np.array([[
        data['k1_right'],
        data['k2_right'],
        data['p1_right'],
        data['p2_right'],
        data.get('k3_right', 0.0)
    ]], dtype=np.float64)

    # Extrinsics (optional for intrinsics-only files)
    if 'R' in data and 'T' in data:
        calib.R = np.array(data['R']).reshape(3, 3)
        calib.T = np.array(data['T']).reshape(3, 1)

    return calib


def _parse_cv_yml(path, *keys):
    """Parse keys from an OpenCV FileStorage YML file."""
    outputs = []
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"Cannot open file: {path}")

    for key in keys:
        outputs.append(fs.getNode(key).mat())

    fs.release()
    return tuple(outputs)


def read_opencv(input_path):
    """Read calibration from OpenCV YML format (intrinsics.yml + extrinsics.yml)."""
    calib = StereoCalibration()
    input_path = Path(input_path)

    intrinsics_file = input_path / "intrinsics.yml"
    extrinsics_file = input_path / "extrinsics.yml"

    if not intrinsics_file.exists():
        raise IOError(f"Intrinsics file not found: {intrinsics_file}")
    if not extrinsics_file.exists():
        raise IOError(f"Extrinsics file not found: {extrinsics_file}")

    # Read intrinsics
    M1, D1, M2, D2 = _parse_cv_yml(intrinsics_file, "M1", "D1", "M2", "D2")
    calib.camera_matrix_left = M1
    calib.dist_coeffs_left = D1
    calib.camera_matrix_right = M2
    calib.dist_coeffs_right = D2

    # Read extrinsics
    R, T, R1, R2, P1, P2, Q = _parse_cv_yml(extrinsics_file, "R", "T", "R1", "R2", "P1", "P2", "Q")
    calib.R = R
    calib.T = T
    calib.R1 = R1
    calib.R2 = R2
    calib.P1 = P1
    calib.P2 = P2
    calib.Q = Q

    return calib


def _zed_section_to_intrinsics(config, section):
    """Parse ZED camera section into camera matrix and distortion coefficients."""
    items = dict(config.items(section))

    fx = float(items['fx'])
    fy = float(items['fy'])
    cx = float(items['cx'])
    cy = float(items['cy'])
    k1 = float(items['k1'])
    k2 = float(items['k2'])
    k3 = float(items['k3'])
    p1 = float(items['p1'])
    p2 = float(items['p2'])

    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

    return camera_matrix, dist_coeffs


def _Rx(theta):
    """Rotation matrix around X axis."""
    return np.array([
        [1, 0, 0],
        [0, math.cos(theta), -math.sin(theta)],
        [0, math.sin(theta), math.cos(theta)]
    ], dtype=np.float64)


def _Ry(theta):
    """Rotation matrix around Y axis."""
    return np.array([
        [math.cos(theta), 0, math.sin(theta)],
        [0, 1, 0],
        [-math.sin(theta), 0, math.cos(theta)]
    ], dtype=np.float64)


def _Rz(theta):
    """Rotation matrix around Z axis."""
    return np.array([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta), math.cos(theta), 0],
        [0, 0, 1]
    ], dtype=np.float64)


def read_zed(input_path, camera_mode='HD'):
    """Read calibration from ZED camera configuration format."""
    calib = StereoCalibration()

    config = configparser.ConfigParser()
    config.optionxform = str  # Preserve case
    config.read(input_path)

    # Read stereo extrinsics
    if config.has_section('STEREO'):
        stereo = dict(config.items('STEREO'))
        TX = float(stereo['Baseline'])
        TY = float(stereo['TY'])
        TZ = float(stereo['TZ'])
        RX = float(stereo[f'RX_{camera_mode}'])
        RY = float(stereo[f'CV_{camera_mode}'])
        RZ = float(stereo[f'RZ_{camera_mode}'])

        calib.R = _Rx(RX) @ _Ry(RY) @ _Rz(RZ)
        calib.T = np.array([[TX], [TY], [TZ]], dtype=np.float64)

    # Read left camera intrinsics
    left_section = f'LEFT_CAM_{camera_mode}'
    if config.has_section(left_section):
        calib.camera_matrix_left, calib.dist_coeffs_left = _zed_section_to_intrinsics(config, left_section)

    # Read right camera intrinsics
    right_section = f'RIGHT_CAM_{camera_mode}'
    if config.has_section(right_section):
        calib.camera_matrix_right, calib.dist_coeffs_right = _zed_section_to_intrinsics(config, right_section)

    return calib


# =============================================================================
# CamCAL / PtsCAL support
# =============================================================================

def parse_camcal(filepath):
    """Parse a CamCAL binary file (camera interior orientation).

    Format:
      - Header: 'CBL\\x00' magic + int32 (negative = UTF-16LE string length)
      - Description string (UTF-16LE encoded)
      - Pixel sizes: 2 x float64 (pixel_size_x, pixel_size_y in mm)
      - Image dimensions: 2 x int32 (width, height)
      - MAT[0]: 10 x float64 parameters [xp, yp, c, k1, k2, k3, p1, p2, b1, b2]
      - MAT[1]: 10 x uint8 flags (1=fixed, 0=estimated)
      - MAT[2]: 20 x float64 (10 std devs + 10 min thresholds) + 1 trailing byte

    Returns:
        dict with keys: description, pixel_size, image_size, params (dict of
        named values), flags (dict), uncertainties (dict of std_dev values)
    """
    with open(filepath, 'rb') as f:
        # Magic header
        magic = f.read(4)
        if magic != b'CBL\x00':
            raise ValueError(f"Not a CamCAL file (expected CBL header): {filepath}")

        # Description string
        str_field = struct.unpack('<i', f.read(4))[0]
        if str_field < 0:
            str_len = -str_field
            desc_bytes = f.read(str_len * 2)  # UTF-16LE: 2 bytes per char
            description = desc_bytes.decode('utf-16-le').rstrip('\x00')
        else:
            description = ""

        # Pixel sizes (mm per pixel)
        pixel_size_x, pixel_size_y = struct.unpack('<2d', f.read(16))

        # Image dimensions
        width, height = struct.unpack('<2i', f.read(8))

        # Each MAT section has a 12-byte header: 'MAT\x00' + uint32 rows + uint32 cols
        def read_mat_header():
            mat_magic = f.read(4)
            if mat_magic != b'MAT\x00':
                raise ValueError(f"Expected MAT header, got {mat_magic!r}")
            rows, cols = struct.unpack('<2I', f.read(8))
            return rows, cols

        # MAT[0]: 10 interior orientation parameters
        param_names = ['xp', 'yp', 'c', 'k1', 'k2', 'k3', 'p1', 'p2', 'b1', 'b2']
        read_mat_header()
        raw_params = struct.unpack('<10d', f.read(80))
        params = dict(zip(param_names, raw_params))

        # MAT[1]: 10 flag bytes (1=fixed, 0=estimated)
        read_mat_header()
        raw_flags = struct.unpack('<10B', f.read(10))
        flags = dict(zip(param_names, raw_flags))

        # MAT[2]: 20 doubles (10 std devs + 10 min thresholds) + 1 trailing byte
        read_mat_header()
        raw_uncert = struct.unpack('<20d', f.read(160))
        std_devs = dict(zip(param_names, raw_uncert[:10]))
        # raw_uncert[10:20] are minimum thresholds (rarely used)
        f.read(1)  # trailing byte

    return {
        'description': description,
        'pixel_size': (pixel_size_x, pixel_size_y),
        'image_size': (width, height),
        'params': params,
        'flags': flags,
        'uncertainties': std_devs,
    }


def parse_ptscal(filepath):
    """Parse a PtsCAL binary file (3D reference points).

    Format:
      - Header: 'MBL\\x00' magic + uint32 point count
      - Point entries: 146 bytes each (PBL format)
        - 'PBL\\x01' magic (4 bytes) + uint32 point ID (4 bytes)
        - 3 x 41-byte coordinate sub-records (X, Y, Z), each containing:
          - float64 coordinate value (mm)
          - float64 standard deviation
          - 16 bytes reserved (zeros)
          - 1 flag byte
          - float64 weight (typically 1.0)
        - 15-byte trailer (float64 + 7 footer bytes)
      - Optional BDI section: known inter-point distances
        - 'BDI\\x00' magic + uint32 count
        - Each entry: 2 x uint32 point IDs + float64 distance (mm)

    Returns:
        dict with keys:
          points: OrderedDict of {label: (x, y, z)} in mm
          distances: list of (label1, label2, distance_mm) tuples
    """
    with open(filepath, 'rb') as f:
        # Magic header
        magic = f.read(4)
        if magic != b'MBL\x00':
            raise ValueError(f"Not a PtsCAL file (expected MBL header): {filepath}")

        count = struct.unpack('<I', f.read(4))[0]

        points = {}
        point_ids = {}  # id -> label mapping for BDI lookups
        for _ in range(count):
            entry = f.read(146)
            if len(entry) < 146:
                break

            # Header: PBL\x01 + uint32 point ID
            pbl_magic = entry[:4]
            if pbl_magic != b'PBL\x01':
                raise ValueError(f"Expected PBL entry header, got {pbl_magic!r}")
            pt_id = struct.unpack('<I', entry[4:8])[0]
            label = str(pt_id)

            # 3 coordinate sub-records at offsets 8, 49, 90 (41 bytes each)
            coords = []
            for dim in range(3):
                base = 8 + dim * 41
                coord = struct.unpack('<d', entry[base:base+8])[0]
                coords.append(coord)

            points[label] = tuple(coords)
            point_ids[pt_id] = label

        # BDI section: known inter-point distances
        # Format: uint32 count, then count x 45-byte entries
        # Each entry: BDI\x00(4) + id1(4) + id2(4) + distance(8) + uncertainty(8) +
        #             8 reserved + 8 reserved + 1 trailing byte
        distances = []
        bdi_count_raw = f.read(4)
        if len(bdi_count_raw) == 4:
            bdi_count = struct.unpack('<I', bdi_count_raw)[0]
            for _ in range(bdi_count):
                bdi_entry = f.read(45)
                if len(bdi_entry) < 45:
                    break
                # Skip BDI\x00 magic (4 bytes)
                id1 = struct.unpack('<I', bdi_entry[4:8])[0]
                id2 = struct.unpack('<I', bdi_entry[8:12])[0]
                dist_val = struct.unpack('<d', bdi_entry[12:20])[0]
                label1 = point_ids.get(id1, str(id1))
                label2 = point_ids.get(id2, str(id2))
                distances.append((label1, label2, dist_val))

    return {
        'points': points,
        'distances': distances,
    }


def camcal_to_opencv(camcal_data):
    """Convert CamCAL parameters to OpenCV camera matrix and distortion.

    CamCAL interior orientation model:
      xp, yp  - principal point offset from image center (mm)
      c       - focal length (mm)
      k1..k3  - radial distortion (CamCAL convention)
      p1, p2  - tangential distortion (CamCAL convention)
      b1, b2  - affinity/non-orthogonality terms

    Conversion to OpenCV:
      fx = c*(1+b1) / pixel_size_x     fy = c / pixel_size_y
      cx = W/2 + xp/pixel_size_x       cy = H/2 + yp/pixel_size_y
      skew = c*b2 / pixel_size_x
      k1_cv = k1*c^2   k2_cv = k2*c^4   k3_cv = k3*c^6
      p1_cv = p1*c      p2_cv = p2*c

    Returns:
        (camera_matrix, dist_coeffs) as numpy arrays
        camera_matrix is 3x3 with skew term at [0,1]
        dist_coeffs is (1,5) array [k1, k2, p1, p2, k3]
    """
    p = camcal_data['params']
    px, py = camcal_data['pixel_size']
    W, H = camcal_data['image_size']

    c = p['c']
    fx = c * (1.0 + p['b1']) / px
    fy = c / py
    cx = W / 2.0 + p['xp'] / px
    cy = H / 2.0 + p['yp'] / py
    skew = c * p['b2'] / px

    camera_matrix = np.array([
        [fx, skew, cx],
        [0,  fy,   cy],
        [0,  0,    1 ]
    ], dtype=np.float64)

    k1_cv = p['k1'] * c**2
    k2_cv = p['k2'] * c**4
    k3_cv = p['k3'] * c**6
    p1_cv = p['p1'] * c
    p2_cv = p['p2'] * c

    dist_coeffs = np.array([[k1_cv, k2_cv, p1_cv, p2_cv, k3_cv]], dtype=np.float64)

    return camera_matrix, dist_coeffs


def read_cal(left_camcal=None, right_camcal=None, ptscal_path=None,
             extrinsics_mode='skip'):
    """Read stereo calibration from CamCAL/PtsCAL files.

    Args:
        left_camcal: Path to left camera .CamCAL file
        right_camcal: Path to right camera .CamCAL file
        ptscal_path: Path to .PtsCAL file (3D reference points)
        extrinsics_mode: How to handle extrinsics:
            'skip'    - Output intrinsics only, no R/T
            'derive'  - Use PtsCAL + both cameras to derive R,T via
                        cv2.stereoCalibrate with fixed intrinsics
            'extract' - Placeholder for future CamCAL formats with extrinsics

    Returns:
        StereoCalibration object (may lack extrinsics if mode='skip')
    """
    if left_camcal is None and right_camcal is None:
        raise ValueError("At least one .CamCAL file is required")

    calib = StereoCalibration()

    # Parse and convert left camera
    left_data = None
    if left_camcal is not None:
        left_data = parse_camcal(left_camcal)
        calib.camera_matrix_left, calib.dist_coeffs_left = camcal_to_opencv(left_data)
        calib.image_width, calib.image_height = left_data['image_size']

    # Parse and convert right camera
    right_data = None
    if right_camcal is not None:
        right_data = parse_camcal(right_camcal)
        calib.camera_matrix_right, calib.dist_coeffs_right = camcal_to_opencv(right_data)
        if calib.image_width is None:
            calib.image_width, calib.image_height = right_data['image_size']

    # Parse PtsCAL if provided
    ptscal_data = None
    if ptscal_path is not None:
        ptscal_data = parse_ptscal(ptscal_path)

    # Handle extrinsics
    if extrinsics_mode == 'derive':
        if left_data is None or right_data is None:
            raise ValueError("Both --left-cal and --right-cal files required to derive extrinsics")
        if ptscal_data is None:
            raise ValueError("--pts file required to derive extrinsics")

        # Use 3D reference points projected through each camera model to
        # establish correspondences, then solve for R, T
        pts_3d = np.array(list(ptscal_data['points'].values()), dtype=np.float64)
        n_pts = len(pts_3d)
        if n_pts < 4:
            raise ValueError(f"Need at least 4 reference points, got {n_pts}")

        # Project 3D points through each camera to get synthetic 2D points
        # Use identity pose for left camera (world frame = left camera frame)
        rvec_zero = np.zeros((3, 1), dtype=np.float64)
        tvec_zero = np.zeros((3, 1), dtype=np.float64)

        pts_2d_left, _ = cv2.projectPoints(
            pts_3d, rvec_zero, tvec_zero,
            calib.camera_matrix_left, calib.dist_coeffs_left)
        pts_2d_right, _ = cv2.projectPoints(
            pts_3d, rvec_zero, tvec_zero,
            calib.camera_matrix_right, calib.dist_coeffs_right)

        # Reshape for stereoCalibrate
        obj_pts = [pts_3d.reshape(-1, 1, 3).astype(np.float32)]
        img_left = [pts_2d_left.reshape(-1, 1, 2).astype(np.float32)]
        img_right = [pts_2d_right.reshape(-1, 1, 2).astype(np.float32)]

        img_size = (calib.image_width, calib.image_height)

        _, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
            obj_pts, img_left, img_right,
            calib.camera_matrix_left, calib.dist_coeffs_left,
            calib.camera_matrix_right, calib.dist_coeffs_right,
            img_size, flags=cv2.CALIB_FIX_INTRINSIC)

        calib.R = R
        calib.T = T

    elif extrinsics_mode == 'extract':
        raise NotImplementedError(
            "Extracting extrinsics from .CamCAL is not yet supported. "
            "Current .CamCAL files contain interior orientation only.")

    # Store CamCAL/PtsCAL metadata
    calib._cal_metadata = {}
    if left_data is not None:
        calib._cal_metadata['left_description'] = left_data['description']
        calib._cal_metadata['left_params'] = left_data['params']
    if right_data is not None:
        calib._cal_metadata['right_description'] = right_data['description']
        calib._cal_metadata['right_params'] = right_data['params']
    if ptscal_data is not None:
        calib._cal_metadata['reference_points'] = {
            k: list(v) for k, v in ptscal_data['points'].items()
        }
        calib._cal_metadata['known_distances'] = [
            {'from': d[0], 'to': d[1], 'distance_mm': d[2]}
            for d in ptscal_data['distances']
        ]

    return calib


# =============================================================================
# Format writers
# =============================================================================

def write_npz(calib, output_path):
    """Write calibration to NumPy NPZ format."""
    calib.validate()

    data = {
        'cameraMatrixL': calib.camera_matrix_left,
        'distCoeffsL': calib.dist_coeffs_left,
        'cameraMatrixR': calib.camera_matrix_right,
        'distCoeffsR': calib.dist_coeffs_right,
        'R': calib.R,
        'T': calib.T,
    }

    if calib.has_rectification():
        data.update({
            'R1': calib.R1,
            'R2': calib.R2,
            'P1': calib.P1,
            'P2': calib.P2,
            'Q': calib.Q,
        })

    np.savez(output_path, **data)


def write_json(calib, output_path):
    """Write calibration to KWIVER-compatible JSON format.

    Supports intrinsics-only output when extrinsics are not available.
    """
    calib.validate(require_extrinsics=calib.has_extrinsics())

    data = {}

    # Metadata
    if calib.image_width is not None:
        data['image_width'] = calib.image_width
    if calib.image_height is not None:
        data['image_height'] = calib.image_height
    if calib.grid_width is not None:
        data['grid_width'] = calib.grid_width
    if calib.grid_height is not None:
        data['grid_height'] = calib.grid_height
    if calib.square_size_mm is not None:
        data['square_size_mm'] = calib.square_size_mm
    if calib.rms_error_left is not None:
        data['rms_error_left'] = float(calib.rms_error_left)
    if calib.rms_error_right is not None:
        data['rms_error_right'] = float(calib.rms_error_right)
    if calib.rms_error_stereo is not None:
        data['rms_error_stereo'] = float(calib.rms_error_stereo)

    # Extrinsics (if available)
    if calib.has_extrinsics():
        data['T'] = calib.T.flatten().tolist()
        data['R'] = calib.R.flatten().tolist()

    # Left camera intrinsics
    if calib.camera_matrix_left is not None:
        M_left = calib.camera_matrix_left
        D_left = calib.dist_coeffs_left.flatten()
        data['fx_left'] = float(M_left[0, 0])
        data['fy_left'] = float(M_left[1, 1])
        data['cx_left'] = float(M_left[0, 2])
        data['cy_left'] = float(M_left[1, 2])
        if M_left[0, 1] != 0:
            data['skew_left'] = float(M_left[0, 1])
        data['k1_left'] = float(D_left[0])
        data['k2_left'] = float(D_left[1])
        data['p1_left'] = float(D_left[2])
        data['p2_left'] = float(D_left[3])
        data['k3_left'] = float(D_left[4]) if len(D_left) > 4 else 0.0

    # Right camera intrinsics
    if calib.camera_matrix_right is not None:
        M_right = calib.camera_matrix_right
        D_right = calib.dist_coeffs_right.flatten()
        data['fx_right'] = float(M_right[0, 0])
        data['fy_right'] = float(M_right[1, 1])
        data['cx_right'] = float(M_right[0, 2])
        data['cy_right'] = float(M_right[1, 2])
        if M_right[0, 1] != 0:
            data['skew_right'] = float(M_right[0, 1])
        data['k1_right'] = float(D_right[0])
        data['k2_right'] = float(D_right[1])
        data['p1_right'] = float(D_right[2])
        data['p2_right'] = float(D_right[3])
        data['k3_right'] = float(D_right[4]) if len(D_right) > 4 else 0.0

    # CamCAL/PtsCAL metadata (reference points, known distances)
    if hasattr(calib, '_cal_metadata') and calib._cal_metadata:
        data['cal_metadata'] = calib._cal_metadata

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def _write_cv_yml(path, **kwargs):
    """Write data to an OpenCV FileStorage YML file."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise IOError(f"Cannot open file: {path}")

    for key, value in kwargs.items():
        fs.write(key, value)

    fs.release()


def write_opencv(calib, output_path, image_width=None, image_height=None):
    """Write calibration to OpenCV YML format (intrinsics.yml + extrinsics.yml)."""
    calib.validate()

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Compute rectification if not available
    if not calib.has_rectification():
        calib.compute_rectification(image_width, image_height)

    # Write intrinsics
    _write_cv_yml(
        output_path / "intrinsics.yml",
        M1=calib.camera_matrix_left,
        D1=calib.dist_coeffs_left,
        M2=calib.camera_matrix_right,
        D2=calib.dist_coeffs_right
    )

    # Write extrinsics
    _write_cv_yml(
        output_path / "extrinsics.yml",
        R=calib.R,
        T=calib.T,
        R1=calib.R1,
        R2=calib.R2,
        P1=calib.P1,
        P2=calib.P2,
        Q=calib.Q
    )


def _intrinsics_to_zed_dict(camera_matrix, dist_coeffs):
    """Convert camera matrix and distortion coefficients to ZED format dict."""
    dist = dist_coeffs.flatten()
    return {
        'fx': camera_matrix[0, 0],
        'fy': camera_matrix[1, 1],
        'cx': camera_matrix[0, 2],
        'cy': camera_matrix[1, 2],
        'k1': dist[0],
        'k2': dist[1],
        'k3': dist[4] if len(dist) > 4 else 0.0,
        'p1': dist[2],
        'p2': dist[3],
    }


def _extrinsics_to_zed_dict(R, T, camera_mode):
    """Convert rotation matrix and translation to ZED stereo format dict."""
    T = T.flatten()
    rotation_angles = Rotation.from_matrix(R).as_euler("xyz", degrees=False)

    return {
        'Baseline': T[0],
        'TY': T[1],
        'TZ': T[2],
        f'RX_{camera_mode}': rotation_angles[0],
        f'CV_{camera_mode}': rotation_angles[1],
        f'RZ_{camera_mode}': rotation_angles[2],
    }


def write_zed(calib, output_path, camera_mode='HD'):
    """Write calibration to ZED camera configuration format."""
    calib.validate()

    config = configparser.ConfigParser()
    config.optionxform = str  # Preserve case

    config['STEREO'] = _extrinsics_to_zed_dict(calib.R, calib.T, camera_mode)
    config[f'LEFT_CAM_{camera_mode}'] = _intrinsics_to_zed_dict(
        calib.camera_matrix_left, calib.dist_coeffs_left
    )
    config[f'RIGHT_CAM_{camera_mode}'] = _intrinsics_to_zed_dict(
        calib.camera_matrix_right, calib.dist_coeffs_right
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        config.write(f)


# =============================================================================
# Format detection
# =============================================================================

SUPPORTED_FORMATS = ['npz', 'json', 'opencv', 'zed', 'cal']


def detect_format(path):
    """Auto-detect calibration format from path."""
    path = Path(path)

    if path.is_dir():
        # Check for OpenCV YML files
        if (path / "intrinsics.yml").exists() and (path / "extrinsics.yml").exists():
            return 'opencv'
        # Could also be a directory containing calibration.npz
        if (path / "calibration.npz").exists():
            return 'npz'
        raise ValueError(f"Cannot detect format for directory: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    suffix = path.suffix.lower()

    if suffix == '.npz':
        return 'npz'
    elif suffix == '.json':
        return 'json'
    elif suffix in ['.yml', '.yaml']:
        # Could be OpenCV format - check if parent has both files
        parent = path.parent
        if (parent / "intrinsics.yml").exists() and (parent / "extrinsics.yml").exists():
            return 'opencv'
        raise ValueError(f"Single YML file found, but OpenCV format requires both intrinsics.yml and extrinsics.yml")
    elif suffix.lower() in ['.camcal', '.ptscal']:
        return 'cal'
    elif suffix in ['.conf', '.ini', '.cfg', '']:
        # Try to detect ZED format by reading first few lines
        try:
            config = configparser.ConfigParser()
            config.read(str(path))
            if config.has_section('STEREO') or any('CAM_' in s for s in config.sections()):
                return 'zed'
        except Exception:
            pass

    raise ValueError(f"Cannot detect format for: {path}")


def infer_output_format(output_path):
    """Infer output format from path."""
    path = Path(output_path)
    suffix = path.suffix.lower()

    if suffix == '.npz':
        return 'npz'
    elif suffix == '.json':
        return 'json'
    elif suffix in ['.conf', '.ini', '.cfg']:
        return 'zed'
    elif suffix in ['.yml', '.yaml']:
        # Single YML implies OpenCV directory format
        return 'opencv'
    elif suffix == '' or path.is_dir():
        # Directory implies OpenCV format
        return 'opencv'

    return None


# =============================================================================
# Main conversion function
# =============================================================================

def convert(input_path, output_path, input_format=None, output_format=None,
            camera_mode='HD', image_width=None, image_height=None,
            left_camcal=None, right_camcal=None, ptscal=None,
            extrinsics_mode='skip'):
    """
    Convert calibration between formats.

    Args:
        input_path: Path to input calibration file or directory (unused for cal format)
        output_path: Path to output calibration file or directory
        input_format: Input format (auto-detected if None)
        output_format: Output format (inferred from output_path if None)
        camera_mode: ZED camera mode (2K, FHD, HD, VGA)
        image_width: Image width for rectification computation
        image_height: Image height for rectification computation
        left_camcal: Path to left .CamCAL file
        right_camcal: Path to right .CamCAL file
        ptscal: Path to .PtsCAL file
        extrinsics_mode: Extrinsics handling for cal format (skip, derive, extract)
    """
    # Auto-detect input format from CamCAL args or input_path
    if input_format is None:
        if left_camcal is not None or right_camcal is not None:
            input_format = 'cal'
        else:
            input_format = detect_format(input_path)
        print(f"Detected input format: {input_format}")

    # Infer output format
    if output_format is None:
        output_format = infer_output_format(output_path)
        if output_format is None:
            raise ValueError("Cannot infer output format from path. Please specify --output-format")
        print(f"Inferred output format: {output_format}")

    # Read calibration
    if input_format == 'cal':
        if left_camcal is None and right_camcal is None:
            raise ValueError("cal format requires --left-cal and/or --right-cal")
        calib = read_cal(left_camcal, right_camcal, ptscal, extrinsics_mode)
    else:
        readers = {
            'npz': read_npz,
            'json': read_json,
            'opencv': read_opencv,
            'zed': lambda p: read_zed(p, camera_mode),
        }

        if input_format not in readers:
            raise ValueError(f"Unsupported input format: {input_format}")

        calib = readers[input_format](input_path)

    # Store image dimensions if provided
    if image_width is not None:
        calib.image_width = image_width
    if image_height is not None:
        calib.image_height = image_height

    # Compute rectification if needed and possible
    if not calib.has_rectification() and calib.has_extrinsics() and output_format == 'opencv':
        if calib.image_width is not None and calib.image_height is not None:
            print("Computing rectification parameters...")
            calib.compute_rectification()
        else:
            raise ValueError(
                "OpenCV format requires rectification parameters. "
                "Please provide --image-width and --image-height to compute them."
            )

    # Write calibration
    writers = {
        'npz': write_npz,
        'json': write_json,
        'opencv': lambda c, p: write_opencv(c, p, image_width, image_height),
        'zed': lambda c, p: write_zed(c, p, camera_mode),
    }

    if output_format not in writers:
        raise ValueError(f"Unsupported output format: {output_format}")

    writers[output_format](calib, output_path)
    print(f"Conversion complete: -> {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    description = """
Convert stereo camera calibration files between different formats.

Supported formats:
  npz     - NumPy archive format (calibration.npz)
  json    - KWIVER camera_rig_io compatible JSON format
  opencv  - OpenCV FileStorage format (intrinsics.yml + extrinsics.yml in a directory)
  zed     - ZED camera INI-style configuration format
  cal     - CamCAL/PtsCAL binary calibration format (input only)

Examples:
  # Convert OpenCV YML to NPZ
  %(prog)s ./calib_folder/ output.npz

  # Convert NPZ to JSON
  %(prog)s calibration.npz calibration.json

  # Convert ZED to OpenCV (requires image dimensions)
  %(prog)s camera.conf ./opencv_calib/ --image-width 1280 --image-height 720

  # Convert with explicit formats
  %(prog)s input.conf output.json --input-format zed --output-format json

  # Convert CamCAL intrinsics to JSON (no extrinsics)
  %(prog)s --left-cal left.CamCAL --right-cal right.CamCAL -o json output.json

  # Convert CamCAL with derived extrinsics from PtsCAL
  %(prog)s --left-cal left.CamCAL --right-cal right.CamCAL \\
    --pts points.PtsCAL --extrinsics-mode derive -o json output.json
"""

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("input_path", nargs='?', default=None,
                        help="Path to input calibration file or directory "
                             "(not required for cal format)")
    parser.add_argument("output_path",
                        help="Path to output calibration file or directory")

    parser.add_argument("-i", "--input-format", choices=SUPPORTED_FORMATS, default=None,
                        help="Input format (auto-detected if not specified)")
    parser.add_argument("-o", "--output-format", choices=SUPPORTED_FORMATS, default=None,
                        help="Output format (inferred from output path if not specified)")

    parser.add_argument("--camera-mode", default="HD", choices=["2K", "FHD", "HD", "VGA"],
                        help="ZED camera mode for reading/writing ZED format (default: HD)")
    parser.add_argument("--image-width", type=int, default=None,
                        help="Image width in pixels (required for computing rectification)")
    parser.add_argument("--image-height", type=int, default=None,
                        help="Image height in pixels (required for computing rectification)")

    # CamCAL/PtsCAL options
    cal_group = parser.add_argument_group('CamCAL/PtsCAL options')
    cal_group.add_argument("--left-cal", default=None, metavar="PATH",
                           help="Path to left camera .CamCAL file")
    cal_group.add_argument("--right-cal", default=None, metavar="PATH",
                           help="Path to right camera .CamCAL file")
    cal_group.add_argument("--pts", default=None, metavar="PATH",
                           help="Path to .PtsCAL file (3D reference points)")
    cal_group.add_argument("--extrinsics-mode", default="skip",
                           choices=["skip", "derive", "extract"],
                           help="How to handle extrinsics for cal format: "
                                "skip (intrinsics only), derive (fit R,T from PtsCAL), "
                                "extract (future, not yet supported) (default: skip)")

    args = parser.parse_args()

    # Validate: need either input_path or cal args
    is_cal = args.left_cal is not None or args.right_cal is not None
    if args.input_path is None and not is_cal:
        parser.error("input_path is required unless using --left-cal/--right-cal")

    try:
        convert(
            input_path=args.input_path,
            output_path=args.output_path,
            input_format=args.input_format,
            output_format=args.output_format,
            camera_mode=args.camera_mode,
            image_width=args.image_width,
            image_height=args.image_height,
            left_camcal=args.left_cal,
            right_camcal=args.right_cal,
            ptscal=args.pts,
            extrinsics_mode=args.extrinsics_mode,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
