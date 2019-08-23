#!/usr/bin/env python

from __future__ import print_function

import os
import re
import setuptools

################################################################################

setuptools.setup(
    name='viame-python-deps',
    version="1.0.0",
    description='Ensure any required python dependencies are in the install',
    author='Kitware, Inc.',
    author_email='viame.developers@kitware.com',
    url='https://github.com/VIAME/VIAME',
    license='BSD 3-Clause',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition'
    ],
    platforms=[
        'Linux',
        'Max OS-X',
        'Unix',
        'Windows',
    ],
    setup_requires=[
        'setuptools',
    ],
    install_requires=[
        'matplotlib',
        'numpy',
        'opencv-python',
    ],
    extras_require={
        'ubelt': [
            'ubelt',
        ],
        'caffe': [
            'protobuf',
            'scikit-image',
        ],
    },
    package_dir='',
    packages='',
    package_data={},
    entry_points={},
)
