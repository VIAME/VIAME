import os.path
from argparse import ArgumentParser
import cv2
import numpy as np
from collections import OrderedDict


def parse_cv_yml(path, *keys):
    outputs = []
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"Cannot open file : {path}")

    for key in keys:
        outputs.append(fs.getNode(key).mat())

    fs.release()
    return tuple(outputs)


def parse_intrinsics(path):
    return parse_cv_yml(path, "M1", "D1", "M2", "D2")


def parse_extrinsics(path):
    return parse_cv_yml(path, "R", "T", "R1", "R2", "P1", "P2", "Q")


def write_numpy_file(output_path, **kwargs):
    with open(os.path.join(output_path, "calibration.npz"), "wb") as f:
        np.savez(f, **kwargs)


def convert_camera_format(input_path, output_path):
    input_extrinsics = os.path.join(input_path, "extrinsics.yml")
    input_intrinsics = os.path.join(input_path, "intrinsics.yml")
    if not os.path.exists(input_extrinsics) or not os.path.exists(input_intrinsics):
        raise IOError(f"Input files don't exists : \n{input_extrinsics}\n{input_intrinsics}")

    M1, D1, M2, D2 = parse_intrinsics(input_intrinsics)
    R, T, R1, R2, P1, P2, Q = parse_extrinsics(input_extrinsics)

    write_numpy_file(output_path=output_path, cameraMatrixL=M1, distCoeffsL=D1, cameraMatrixR=M2, distCoeffsR=D2, R=R,
                     T=T, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)


def main():
    parser = ArgumentParser(
        description="Converter from VIAME calib_camera.py camera calibration format to numpy NPZ format."
                    " Expects one folder as input containing two OpenCV FileStorage files extrinsics.yml"
                    " and intrinsics.yml as well as one output folder which will contain the output NPZ file.")

    parser.add_argument("input_path", default="./",
                        help="Input directory containing an extrinsics.yml and intrinsics.yml file")

    parser.add_argument("output_path", default="./",
                        help="Output directory which will contain the resulting calibration.npz file")

    args = parser.parse_args()

    convert_camera_format(args.input_path, args.output_path)


if __name__ == '__main__':
    main()
