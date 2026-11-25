#!/usr/bin/env python
# Copyright (c) Megvii, Inc. and its affiliates. All Rights Reserved

# VIAME patch: upstream's setup.py compiles the FastCOCOEvalOp C++ extension,
# maps tools/ and exps/ into the package and installs requirements.txt (thop,
# lap, motmetrics, onnx==1.8.1, ...). The VIAME plugins only import the model,
# data-augmentation and utility modules, so this installs the python package
# alone; its runtime dependencies are listed in cmake/add_project_python_deps.cmake.

import re
import setuptools


def get_yolox_version():
    with open("yolox/__init__.py", "r") as f:
        version = re.search(
            r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
            f.read(), re.MULTILINE
        ).group(1)
    return version


setuptools.setup(
    name="yolox",
    version=get_yolox_version(),
    author="megvii basedet team",
    url="https://github.com/YuHengsss/YOLOV",
    packages=setuptools.find_packages(include=("yolox", "yolox.*")),
    python_requires=">=3.6",
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3", "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
    ],
)
