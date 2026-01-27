# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import os
import torch

from setuptools import setup, find_packages

from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

modules = []

if torch.cuda.is_available():
    nvcc_flags = ['-O2']
    # Support NVCC_FLAGS environment variable to pass additional flags to nvcc
    # (e.g., -allow-unsupported-compiler for newer VS versions)
    nvcc_flags_env = os.getenv("NVCC_FLAGS", "")
    if nvcc_flags_env:
        nvcc_flags.extend(nvcc_flags_env.split())

    modules.append(
        CUDAExtension(
            'roi_align.roi_align_cuda',
            ['roi_align/src/roi_align_cuda.cpp',
             'roi_align/src/roi_align_kernel.cu'],
            extra_compile_args={'cxx': ['-g'],
                                'nvcc': nvcc_flags}
        )
    )

setup(
    name='roi_align',
    version='1.0.0',
    description='PyTorch version of RoIAlign specific to MDNet',
    author='VIAME',
    author_email='viame.developers@gmail.com',
    url='https://github.com/VIAME/VIAME/tree/master/plugins/pytorch/mdnet',
    packages=find_packages(exclude=('tests',)),
    ext_modules=modules,
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch>=1.0.0']
)
