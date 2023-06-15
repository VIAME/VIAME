import configparser
import math as m
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def parse_cv_yml(path, *keys):
    outputs = []
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"Cannot open file : {path}")

    for key in keys:
        outputs.append(fs.getNode(key).mat())

    fs.release()
    return tuple(outputs)


def write_cv_yml(path, **kwargs):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise IOError(f"Cannot open file : {path}")

    for key, value in kwargs.items():
        fs.write(key, value)

    fs.release()


def parse_cv_intrinsics(path):
    return parse_cv_yml(path, "M1", "D1", "M2", "D2")


def write_cv_intrinsics(path, M1, D1, M2, D2):
    write_cv_yml(path, M1=M1, D1=D1, M2=M2, D2=D2)


def parse_cv_extrinsics(path):
    return parse_cv_yml(path, "R", "T", "R1", "R2", "P1", "P2", "Q")


def write_cv_extrinsics(path, R, T, R1, R2, P1, P2, Q):
    write_cv_yml(path, R=R, T=T, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)


def convert_cv_yml_to_npz(input_path, output_path):
    input_extrinsics = Path(input_path).joinpath("extrinsics.yml")
    input_intrinsics = Path(input_path).joinpath("intrinsics.yml")
    if not input_extrinsics.exists() or not input_intrinsics.exists():
        raise IOError(f"Input files don't exists : \n{input_extrinsics}\n{input_intrinsics}")

    M1, D1, M2, D2 = parse_cv_intrinsics(input_intrinsics)
    R, T, R1, R2, P1, P2, Q = parse_cv_extrinsics(input_extrinsics)

    np.savez(output_path, cameraMatrixL=M1, distCoeffsL=D1, cameraMatrixR=M2, distCoeffsR=D2, R=R,
             T=T, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)


def get_intrinsics_parameters_from_np(data):
    return data["cameraMatrixL"], data["distCoeffsL"], data["cameraMatrixR"], data["distCoeffsR"]


def compute_stereo_rectification(data, image_width, image_height):
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1=data["cameraMatrixL"],
                                                      distCoeffs1=data["distCoeffsL"],
                                                      cameraMatrix2=data["cameraMatrixR"],
                                                      distCoeffs2=data["distCoeffsR"],
                                                      imageSize=(image_width, image_height),
                                                      R=data["R"],
                                                      T=data["T"],
                                                      flags=cv2.CALIB_ZERO_DISPARITY)
    return R1, R2, P1, P2, Q


def get_extrinsics_parameters_from_np(data, image_width, image_height):
    R, T = data["R"], data["T"]

    if all(key in data for key in ["R1", "R2", "P1", "P2", "Q"]):
        return R, T, data["R1"], data["R2"], data["P1"], data["P2"], data["Q"]

    R1, R2, P1, P2, Q = compute_stereo_rectification(data, image_width, image_height)
    return R, T, R1, R2, P1, P2, Q


def convert_npz_to_cv_yml(input_path, output_path, image_width, image_height):
    data = np.load(input_path)
    M1, D1, M2, D2 = get_intrinsics_parameters_from_np(data)
    R, T, R1, R2, P1, P2, Q = get_extrinsics_parameters_from_np(data, image_width, image_height)

    write_cv_intrinsics(os.path.join(output_path, "intrinsics.yml"), M1, D1, M2, D2)
    write_cv_extrinsics(os.path.join(output_path, "extrinsics.yml"), R, T, R1, R2, P1, P2, Q)


def convert_cv_yml_to_conf(input_path, output_path, camera_mode, **_):
    npz_file = os.path.join(input_path, "calibration.npz")
    convert_cv_yml_to_npz(input_path, npz_file)
    convert_npz_to_zed_conf(npz_file, output_path, camera_mode)


def Rx(theta):
    return np.matrix([[1, 0, 0],
                      [0, m.cos(theta), -m.sin(theta)],
                      [0, m.sin(theta), m.cos(theta)]])


def Ry(theta):
    return np.matrix([[m.cos(theta), 0, m.sin(theta)],
                      [0, 1, 0],
                      [-m.sin(theta), 0, m.cos(theta)]])


def Rz(theta):
    return np.matrix([[m.cos(theta), -m.sin(theta), 0],
                      [m.sin(theta), m.cos(theta), 0],
                      [0, 0, 1]])


def np_stereo_to_dict(R, T, camera_mode):
    T = np.reshape(T, (max(T.shape), 1))
    rotation_angles = Rotation.from_matrix(R).as_euler("xyz", degrees=False)

    return {
        "Baseline": T[0, 0],
        "TY": T[1, 0],
        "TZ": T[2, 0],
        f"RX_{camera_mode}": rotation_angles[0],
        f"CV_{camera_mode}": rotation_angles[1],
        f"RZ_{camera_mode}": rotation_angles[2],
    }


def np_cam_to_zed_dict(camera_matrix, dist_coeffs):
    dist_coeffs = np.reshape(dist_coeffs, (max(dist_coeffs.shape), 1))
    return {
        "fx": camera_matrix[0, 0],
        "fy": camera_matrix[1, 1],
        "cx": camera_matrix[0, 2],
        "cy": camera_matrix[1, 2],
        "k1": dist_coeffs[0, 0],
        "k2": dist_coeffs[1, 0],
        "k3": dist_coeffs[4, 0],
        "p1": dist_coeffs[2, 0],
        "p2": dist_coeffs[3, 0],
    }


def zed_conf_camera_section_to_np(config, cam_section):
    fx = float(dict(config.items(cam_section))['fx'])
    fy = float(dict(config.items(cam_section))['fy'])
    cx = float(dict(config.items(cam_section))['cx'])
    cy = float(dict(config.items(cam_section))['cy'])
    k1 = float(dict(config.items(cam_section))['k1'])
    k2 = float(dict(config.items(cam_section))['k2'])
    k3 = float(dict(config.items(cam_section))['k3'])
    p1 = float(dict(config.items(cam_section))['p1'])
    p2 = float(dict(config.items(cam_section))['p2'])
    R = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    distCoeffs = np.array([k1, k2, p1, p2, k3])
    return R, distCoeffs


def convert_npz_to_zed_conf(input_path, output_path, camera_mode):
    data = np.load(input_path)
    config = configparser.ConfigParser()
    config.optionxform = str
    config["STEREO"] = np_stereo_to_dict(data['R'], data['T'], camera_mode)
    config[f"RIGHT_CAM_{camera_mode}"] = np_cam_to_zed_dict(data['cameraMatrixR'], data['distCoeffsR'])
    config[f"LEFT_CAM_{camera_mode}"] = np_cam_to_zed_dict(data['cameraMatrixL'], data['distCoeffsL'])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        config.write(f)


def convert_zed_to_npz(input_path, output_path, camera_mode, image_width, image_height):
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(input_path)
    data = dict()

    if config.has_section('STEREO'):
        TX = float(dict(config.items('STEREO'))['Baseline'])
        TY = float(dict(config.items('STEREO'))['TY'])
        TZ = float(dict(config.items('STEREO'))['TZ'])
        RX = float(dict(config.items('STEREO'))[f'RX_{camera_mode}'])
        RY = float(dict(config.items('STEREO'))[f'CV_{camera_mode}'])
        RZ = float(dict(config.items('STEREO'))[f'RZ_{camera_mode}'])
        data['R'] = Rx(RX) * Ry(RY) * Rz(RZ)
        data['T'] = np.array([TX, TY, TZ])

    right_cam_section = f'RIGHT_CAM_{camera_mode}'
    if config.has_section(right_cam_section):
        data['cameraMatrixR'], data['distCoeffsR'] = zed_conf_camera_section_to_np(config, right_cam_section)

    left_cam_section = f'LEFT_CAM_{camera_mode}'
    if config.has_section(left_cam_section):
        data['cameraMatrixL'], data['distCoeffsL'] = zed_conf_camera_section_to_np(config, left_cam_section)

    data["R1"], data["R2"], data["P1"], data["P2"], data["Q"] = compute_stereo_rectification(data,
                                                                                             image_width,
                                                                                             image_height)
    print(data)
    np.savez(output_path, **data)


def convert_zed_to_cv_yml(input_path, output_path, camera_mode, image_width, image_height, **_):
    output_path = Path(output_path)
    file_name, file_extension = os.path.splitext(output_path)
    if file_extension:
        raise ValueError(f"For VIAME conversion, selected output path should be a folder and not a file path.\n"
                         f"Output path : {output_path} (file ext = {file_extension})")

    output_path.mkdir(parents=True, exist_ok=True)
    npz_path = output_path.joinpath("calibration.npz")

    convert_zed_to_npz(input_path, npz_path, camera_mode, image_width, image_height)
    convert_npz_to_cv_yml(npz_path, output_path, image_width, image_height)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Convert stereo calibration files from ZED to VIAME")
    parser.add_argument("input_path",
                        help="Path to the input ZED file to convert or folder containing OpenCV YML files.")
    parser.add_argument("output_path", help="Path to the output file or destination folder for OpenCV YML files.")
    parser.add_argument("conversion_type", choices=["TO_VIAME", "TO_ZED"],
                        help="Direction of conversion between VIAME / ZED")

    parser.add_argument("--camera_mode", default="HD", choices=["2K", "FHD", "HD", "VGA"],
                        help="Camera mode in : 2K, FHD, HD, VGA.")
    parser.add_argument("--image_width", default=1280, type=int, help="Camera image width (pix)")
    parser.add_argument("--image_height", default=720, type=int, help="Camera image height (pix)")
    args = parser.parse_args()

    if args.conversion_type == "TO_VIAME":
        convert_zed_to_cv_yml(**vars(args))
    else:
        convert_cv_yml_to_conf(**vars(args))
