# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import configparser
import os.path
import sys
import pytest

from pathlib import Path

# Add tools directory to path for imports
_this_dir = os.path.dirname(os.path.abspath(__file__))
_viame_src_dir = os.path.abspath(os.path.join(_this_dir, "..", ".."))
sys.path.insert(0, os.path.join(_viame_src_dir, "tools"))

from convert_cam_format import (
    read_zed, read_opencv, read_npz, read_json,
    write_zed, write_opencv, write_npz, write_json,
    StereoCalibration, convert
)

import numpy as np


@pytest.fixture
def a_test_data_folder():
    file_folder = os.path.dirname(__file__)
    return os.path.join(file_folder, "..", "data")


@pytest.fixture
def a_test_zed_conf():
    file_folder = os.path.dirname(__file__)
    return os.path.join(file_folder, "..", "..", "configs", "add-ons", "ifremer", "models", "SN13238190.conf")


def test_zed_roundtrip(tmpdir, a_test_zed_conf):
    """Test ZED -> NPZ -> ZED roundtrip conversion."""
    # Read ZED, write NPZ
    npz_file = os.path.join(tmpdir, "calib.npz")
    convert(a_test_zed_conf, npz_file, input_format='zed', output_format='npz',
            camera_mode='HD', image_width=1280, image_height=720)

    # Read NPZ, write ZED
    out_zed_file = os.path.join(tmpdir, "zed.conf")
    convert(npz_file, out_zed_file, input_format='npz', output_format='zed',
            camera_mode='HD')

    exp = configparser.ConfigParser()
    exp.read(a_test_zed_conf)

    act = configparser.ConfigParser()
    act.read(out_zed_file)
    assert act.has_section("STEREO")
    assert act.has_section("LEFT_CAM_HD")
    assert act.has_section("RIGHT_CAM_HD")

    for key in ["baseline", "ty", "tz", "rx_hd", "cv_hd", "rz_hd"]:
        assert float(act["STEREO"][key]) == pytest.approx(float(exp["STEREO"][key]), abs=1e-4)

    for cam in ["LEFT", "RIGHT"]:
        for key in ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2"]:
            assert float(act[f"{cam}_CAM_HD"][key]) == pytest.approx(float(exp[f"{cam}_CAM_HD"][key]), abs=1e-9)


def test_opencv_to_npz_to_opencv(a_test_data_folder, tmpdir):
    """Test OpenCV -> NPZ -> OpenCV roundtrip conversion."""
    # Convert OpenCV YML to NPZ
    npz_file = os.path.join(tmpdir, "calibration.npz")
    convert(a_test_data_folder, npz_file, input_format='opencv', output_format='npz')

    # Convert NPZ back to OpenCV YML
    opencv_out = os.path.join(tmpdir, "opencv_out")
    convert(npz_file, opencv_out, input_format='npz', output_format='opencv',
            image_width=1280, image_height=720)

    assert os.path.exists(os.path.join(opencv_out, "intrinsics.yml"))
    assert os.path.exists(os.path.join(opencv_out, "extrinsics.yml"))


def test_rectification_computed_when_missing(a_test_data_folder, tmpdir):
    """Test that rectification parameters are computed when missing from input."""
    # Convert OpenCV to NPZ (includes rectification)
    npz_file = os.path.join(tmpdir, "calibration.npz")
    convert(a_test_data_folder, npz_file, input_format='opencv', output_format='npz')
    data = np.load(npz_file)

    # Create a new NPZ without rectification parameters
    trimmed_npz = os.path.join(tmpdir, "trimmed.npz")
    trimmed_keys = [k for k in data.keys() if k not in ["R1", "R2", "P1", "P2", "Q"]]
    np.savez(trimmed_npz, **{k: data[k] for k in trimmed_keys})

    # Convert to OpenCV - should compute rectification
    opencv_out = os.path.join(tmpdir, "opencv_out")
    convert(trimmed_npz, opencv_out, input_format='npz', output_format='opencv',
            image_width=1280, image_height=720)

    # Read back and verify rectification was computed
    calib = read_opencv(opencv_out)
    assert calib.has_rectification()

    # Expected values
    R1_expected = np.array([[0.9950887778458454, -0.001233347312382034, -0.09897880106198935],
                            [0.001295813956420944, 0.9999989997862252, 0.000566826904725766],
                            [0.0989780029675896, -0.0006923012036796471, 0.9950893807330029]])

    np.testing.assert_allclose(calib.R1, R1_expected, rtol=1e-5)


def test_opencv_to_zed(a_test_data_folder, tmpdir):
    """Test OpenCV -> ZED conversion."""
    zed_file = Path(tmpdir).joinpath("zed.conf")
    convert(a_test_data_folder, str(zed_file), input_format='opencv', output_format='zed',
            camera_mode='HD')
    assert zed_file.exists()

    # Verify we can read back the ZED file and convert to NPZ
    npz_file = Path(tmpdir).joinpath("calibration.npz")
    convert(str(zed_file), str(npz_file), input_format='zed', output_format='npz',
            camera_mode='HD', image_width=1280, image_height=720)
    assert npz_file.exists()


def test_json_roundtrip(a_test_data_folder, tmpdir):
    """Test OpenCV -> JSON -> NPZ conversion."""
    # Convert OpenCV to JSON
    json_file = os.path.join(tmpdir, "calibration.json")
    convert(a_test_data_folder, json_file, input_format='opencv', output_format='json')
    assert os.path.exists(json_file)

    # Convert JSON to NPZ
    npz_file = os.path.join(tmpdir, "calibration.npz")
    convert(json_file, npz_file, input_format='json', output_format='npz')
    assert os.path.exists(npz_file)

    # Verify data
    calib = read_npz(npz_file)
    assert calib.has_intrinsics()
    assert calib.has_extrinsics()


def test_auto_format_detection(a_test_data_folder, a_test_zed_conf, tmpdir):
    """Test automatic format detection."""
    from ..convert_cam_format import detect_format

    assert detect_format(a_test_data_folder) == 'opencv'
    assert detect_format(a_test_zed_conf) == 'zed'

    # Create test files for other formats
    npz_file = os.path.join(tmpdir, "test.npz")
    convert(a_test_data_folder, npz_file, output_format='npz')
    assert detect_format(npz_file) == 'npz'

    json_file = os.path.join(tmpdir, "test.json")
    convert(a_test_data_folder, json_file, output_format='json')
    assert detect_format(json_file) == 'json'
