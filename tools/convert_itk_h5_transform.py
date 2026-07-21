#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Convert an ITK HDF5 transform (.h5) into a DIVE camera registration .json.

The legacy EO/IR registration workflow saved 2D affine transforms with
ITK's HDF5 transform writer (e.g. the Kotz flight files used by the
arctic-seal embedded pipelines). This converts those files once so the
non-ITK transform readers (warp_detections / warp_image with
transform_reader type "dive") can consume them.

The h5 transform's forward direction maps the source camera's pixel
coordinates into the destination camera's (for the Kotz files: thermal
onto optical), which becomes the pair's leftToRight matrix; rightToLeft
is its inverse. Composite transforms of multiple affines are composed
following ITK semantics (the last transform added is applied first).

Example:
    python convert_itk_h5_transform.py Kotz-2019-Flight-Center.h5 \\
        --left thermal --right optical -o Kotz-2019-Flight-Center.json
"""

import argparse
import json
import sys

import h5py
import numpy as np


def affine_to_matrix(parameters, fixed_parameters):
    """Homogeneous 3x3 from an ITK AffineTransform_double_2_2 parameter set.

    ITK stores [a11 a12 a21 a22 tx ty] plus a fixed-parameter center c,
    mapping x -> A(x - c) + c + t.
    """
    if len(parameters) != 6:
        raise ValueError(
            f"Expected 6 affine parameters, got {len(parameters)}"
        )
    a = np.array(parameters[:4], dtype=float).reshape(2, 2)
    t = np.array(parameters[4:6], dtype=float)
    c = np.array(fixed_parameters[:2], dtype=float) if len(fixed_parameters) >= 2 \
        else np.zeros(2)
    matrix = np.eye(3)
    matrix[:2, :2] = a
    matrix[:2, 2] = t + c - a @ c
    return matrix


def load_itk_h5(path):
    """Compose the transform(s) in an ITK HDF5 file into one 3x3 matrix."""
    with h5py.File(path, "r") as h5:
        if "TransformGroup" not in h5:
            raise ValueError(f"No TransformGroup in {path}; not an ITK transform file")
        group = h5["TransformGroup"]
        indices = sorted(group.keys(), key=int)
        transforms = []
        for index in indices:
            entry = group[index]
            transform_type = entry["TransformType"][0]
            if isinstance(transform_type, bytes):
                transform_type = transform_type.decode()
            if transform_type.startswith("CompositeTransform"):
                # Wrapper entry for a composite file; components follow.
                continue
            if not transform_type.startswith("AffineTransform_double_2"):
                raise ValueError(
                    f"Unsupported transform type {transform_type} in {path}; "
                    "only 2D affine transforms can be converted"
                )
            transforms.append(
                affine_to_matrix(
                    np.array(entry["TransformParameters"]),
                    np.array(entry.get("TransformFixedParameters", [])),
                )
            )
    if not transforms:
        raise ValueError(f"No affine transforms found in {path}")
    # ITK composite transforms apply back-to-front: the last transform in
    # the file is applied to the point first, so the composed matrix is
    # M0 @ M1 @ ... @ MN.
    matrix = np.eye(3)
    for component in transforms:
        matrix = matrix @ component
    return matrix


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="ITK .h5 transform file")
    parser.add_argument(
        "-o", "--output",
        help="Output .json path (default: input name with .json extension)",
    )
    parser.add_argument(
        "--left", default="left",
        help="Name of the camera the transform maps FROM (default: left)",
    )
    parser.add_argument(
        "--right", default="right",
        help="Name of the camera the transform maps TO (default: right)",
    )
    args = parser.parse_args()

    matrix = load_itk_h5(args.input)

    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        sys.exit(f"Transform in {args.input} is not invertible")

    output = args.output or args.input.rsplit(".", 1)[0] + ".json"

    registration = {
        "type": "dive-camera-registration",
        "version": 1,
        "pairs": [
            {
                "left": args.left,
                "right": args.right,
                "points": [],
                "leftToRight": matrix.tolist(),
                "rightToLeft": inverse.tolist(),
                "transformType": "homography",
            }
        ],
    }

    with open(output, "w", encoding="utf-8") as out_file:
        json.dump(registration, out_file, indent=2)

    print(f"Wrote {output}")
    print(f"  {args.left} -> {args.right}:")
    for row in matrix:
        print("    [ " + "  ".join(f"{value:12.6f}" for value in row) + " ]")


if __name__ == "__main__":
    main()
