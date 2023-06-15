import configparser
import os.path
import pytest

from pathlib import Path

from ..convert_cam_zed_to_npz import convert_npz_to_zed_conf, convert_zed_to_npz, convert_npz_to_cv_yml, \
    convert_cv_yml_to_npz, get_extrinsics_parameters_from_np, convert_cv_yml_to_conf

import numpy as np


@pytest.fixture
def a_test_data_folder():
    file_folder = os.path.dirname(__file__)
    return os.path.join(file_folder, "..", "..", "configs", "camera_calibration")


@pytest.fixture
def a_test_npz_file(a_test_data_folder):
    return os.path.join(a_test_data_folder, "calibration.npz")


@pytest.fixture
def a_test_zed_conf(a_test_data_folder):
    return os.path.join(a_test_data_folder, "SN13238190.conf")


def test_convert_npz_to_zed_conf(tmpdir, a_test_zed_conf):
    npz_file = os.path.join(tmpdir, "calib.npz")
    convert_zed_to_npz(a_test_zed_conf, npz_file, "HD", 1280, 720)

    out_zed_file = os.path.join(tmpdir, "zed.conf")
    convert_npz_to_zed_conf(npz_file, out_zed_file, "HD")

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


def test_npz_to_yml(a_test_data_folder, tmpdir):
    npz_file = os.path.join(tmpdir, "calibration.npz")
    convert_cv_yml_to_npz(a_test_data_folder, npz_file)
    convert_npz_to_cv_yml(npz_file, tmpdir, image_width=1280, image_height=720)
    assert os.path.exists(os.path.join(tmpdir, "intrinsics.yml"))
    assert os.path.exists(os.path.join(tmpdir, "extrinsics.yml"))


def test_extrinsic_parameters_can_be_infered_from_np_data_if_missing_from_input(a_test_data_folder, tmpdir):
    npz_file = os.path.join(tmpdir, "calibration.npz")
    convert_cv_yml_to_npz(a_test_data_folder, npz_file)
    data = np.load(npz_file)

    trimmed_keys = [k for k in data.keys() if k not in ["R1", "R2", "P1", "P2", "Q"]]
    np.savez(npz_file, **{k: data[k] for k in trimmed_keys})
    act_data = np.load(npz_file)
    actual = get_extrinsics_parameters_from_np(act_data, image_width=1280, image_height=720)

    R = np.array([[0.9999998143840519, -0.0005966199917079363, 0.0001235979256623501],
                  [0.0005964671974351601, 0.9999990624732277, 0.001232590584136468],
                  [-0.0001243331979700502, -0.001232516633239672, 0.999999232721708]])
    T = np.array([[-0.06486585457413174], [4.965984169759214e-05], [0.00646006865736615]])
    R1 = np.array([[0.9950887778458454, -0.001233347312382034, -0.09897880106198935],
                   [0.001295813956420944, 0.9999989997862252, 0.000566826904725766],
                   [0.0989780029675896, -0.0006923012036796471, 0.9950893807330029]])

    R2 = np.array(
        [[0.9950770954066653, -0.0007618086797626445, -0.09910092756664587],
         [0.000699264379540492, 0.999999533836415, -0.0006658500433945483],
         [0.09910138861974471, 0.000593274378530668, 0.9950771642436329]]
    )
    P1 = np.array(
        [[898.7523027614053, 0, 760.3563995361328, 0],
         [0, 898.7523027614053, 359.6799621582031, 0],
         [0, 0, 1, 0]]
    )
    P2 = np.array(
        [[898.7523027614053, 0, 760.3563995361328, -58.58675316535361],
         [0, 898.7523027614053, 359.6799621582031, 0],
         [0, 0, 1, 0]]
    )
    Q = np.array(
        [[1, 0, 0, -760.3563995361328],
         [0, 1, 0, -359.6799621582031],
         [0, 0, 0, 898.7523027614053],
         [0, 0, 15.3405378213193, -0]]
    )

    expected = R, T, R1, R2, P1, P2, Q
    for exp, act in zip(expected, actual):
        np.testing.assert_allclose(exp, act)


def test_cv_yml_to_zed(a_test_data_folder, tmpdir):
    zed_file = Path(tmpdir).joinpath("zed.conf")
    convert_cv_yml_to_conf(a_test_data_folder, zed_file, "HD")
    assert zed_file.exists()

    npz_file = Path(tmpdir).joinpath("calibration.npz")
    convert_zed_to_npz(zed_file, npz_file, "HD", 1280, 720)
    assert npz_file.exists()
