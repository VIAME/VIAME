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
"""

import argparse
import configparser
import json
import math
import os
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

    def validate(self):
        """Validate that minimum required data is present."""
        if not self.has_intrinsics():
            raise ValueError("Missing intrinsic parameters")
        if not self.has_extrinsics():
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
        [data['fx_left'], 0, data['cx_left']],
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
        [data['fx_right'], 0, data['cx_right']],
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

    # Extrinsics
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
    """Write calibration to KWIVER-compatible JSON format."""
    calib.validate()

    # Extract intrinsics
    M_left = calib.camera_matrix_left
    D_left = calib.dist_coeffs_left.flatten()
    M_right = calib.camera_matrix_right
    D_right = calib.dist_coeffs_right.flatten()

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

    # Extrinsics
    data['T'] = calib.T.flatten().tolist()
    data['R'] = calib.R.flatten().tolist()

    # Left camera intrinsics
    data['fx_left'] = float(M_left[0, 0])
    data['fy_left'] = float(M_left[1, 1])
    data['cx_left'] = float(M_left[0, 2])
    data['cy_left'] = float(M_left[1, 2])
    data['k1_left'] = float(D_left[0])
    data['k2_left'] = float(D_left[1])
    data['p1_left'] = float(D_left[2])
    data['p2_left'] = float(D_left[3])
    data['k3_left'] = float(D_left[4]) if len(D_left) > 4 else 0.0

    # Right camera intrinsics
    data['fx_right'] = float(M_right[0, 0])
    data['fy_right'] = float(M_right[1, 1])
    data['cx_right'] = float(M_right[0, 2])
    data['cy_right'] = float(M_right[1, 2])
    data['k1_right'] = float(D_right[0])
    data['k2_right'] = float(D_right[1])
    data['p1_right'] = float(D_right[2])
    data['p2_right'] = float(D_right[3])
    data['k3_right'] = float(D_right[4]) if len(D_right) > 4 else 0.0

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

SUPPORTED_FORMATS = ['npz', 'json', 'opencv', 'zed']


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
            camera_mode='HD', image_width=None, image_height=None):
    """
    Convert calibration between formats.

    Args:
        input_path: Path to input calibration file or directory
        output_path: Path to output calibration file or directory
        input_format: Input format (auto-detected if None)
        output_format: Output format (inferred from output_path if None)
        camera_mode: ZED camera mode (2K, FHD, HD, VGA)
        image_width: Image width for rectification computation
        image_height: Image height for rectification computation
    """
    # Auto-detect input format
    if input_format is None:
        input_format = detect_format(input_path)
        print(f"Detected input format: {input_format}")

    # Infer output format
    if output_format is None:
        output_format = infer_output_format(output_path)
        if output_format is None:
            raise ValueError("Cannot infer output format from path. Please specify --output-format")
        print(f"Inferred output format: {output_format}")

    # Read calibration
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
    if not calib.has_rectification() and output_format == 'opencv':
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
    print(f"Conversion complete: {input_path} -> {output_path}")


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

Examples:
  # Convert OpenCV YML to NPZ
  %(prog)s ./calib_folder/ output.npz

  # Convert NPZ to JSON
  %(prog)s calibration.npz calibration.json

  # Convert ZED to OpenCV (requires image dimensions)
  %(prog)s camera.conf ./opencv_calib/ --image-width 1280 --image-height 720

  # Convert with explicit formats
  %(prog)s input.conf output.json --input-format zed --output-format json
"""

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("input_path",
                        help="Path to input calibration file or directory")
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

    args = parser.parse_args()

    try:
        convert(
            input_path=args.input_path,
            output_path=args.output_path,
            input_format=args.input_format,
            output_format=args.output_format,
            camera_mode=args.camera_mode,
            image_width=args.image_width,
            image_height=args.image_height,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
