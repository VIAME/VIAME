from skbuild import setup
import os.path as osp
import os
from setuptools import find_packages

with open('VERSION', 'r') as f:
    version = f.read().strip()

kwiver_install_dir = 'kwiver'
kwiver_source_dir = '../'
setup(
        name='kwiver',
        version=version,
        description='Python and C++ toolkit that pulls together computer vision algorithms '
                     ' into highly modular run time configurable systems',
        author='Kitware, Inc.',
        author_email='http://public.kitware.com/mailman/listinfo/kwiver-users',
        url='https://github.com/Kitware/kwiver',
        cmake_install_dir=kwiver_install_dir,
        cmake_source_dir=kwiver_source_dir,
        license='BSD 3-Clause',
        classifiers=[
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Operating System :: Unix',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            ],
        platforms=[
                   'linux',
                   'Unix'
                   ],
        cmake_minimum_required_version='3.3',
        packages = find_packages(exclude=['test-*', 'kwiver.sprokit.util']),
        python_requires='>=3.5',
        setup_requires=[
                        'setuptools',
                        'cmake',
                        'scikit-build'
                       ],
        install_requires=[
                          'numpy',
                          'pillow',
                          'six',
                         ],
        tests_require=[
                        'nose',
                        'mock',
                        'coverage',
                        'external_arrow',
                      ],
        cmake_args=[
                    '-DCMAKE_BUILD_TYPE=Release',
                    '-DKWIVER_BUILD_SHARED=OFF',
                    '-DKWIVER_ENABLE_C_BINDINGS=ON',
                    '-DKWIVER_ENABLE_PYTHON=ON',
                    '-DKWIVER_PYTHON_MAJOR_VERSION=3',
                    '-DPYBIND11_PYTHON_VERSION=3',
                    '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON',
                    '-DKWIVER_ENABLE_SPROKIT=ON',
                    '-DKWIVER_ENABLE_ARROWS=ON',
                    '-DKWIVER_ENABLE_PROCESSES=ON',
                    '-DKWIVER_ENABLE_TOOLS=ON',
                    '-DKWIVER_ENABLE_LOG4CPLUS=ON',
                    '-DKWIVER_INSTALL_SET_UP_SCRIPT=OFF',
                    '-DKWIVER_ENABLE_OPENCV=ON',
                    '-DKWIVER_ENABLE_FFMPEG=ON',
                    '-DKWIVER_ENABLE_ZeroMQ=ON',
                    '-DKWIVER_ENABLE_SERIALIZE_JSON=ON',
                    '-DKWIVER_ENABLE_SERIALIZE_PROTOBUF=ON'
                   ],
        entry_points={
            'kwiver.python_plugin_registration': [
                'pythread_process=kwiver.sprokit.schedulers.pythread_per_process',
                'pythread_process_scheduler=kwiver.sprokit.schedulers.pythread_per_process_scheduler',
                'apply_descriptor=kwiver.sprokit.processes.ApplyDescriptor',
                'process_image=kwiver.sprokit.processes.ProcessImage',
                'print_number_process=kwiver.sprokit.processes.kw_print_number_process',
                'homography_writer=kwiver.sprokit.processes.homography_writer',

                ],
            'kwiver.cpp_search_paths': [
                'sprokit_process=kwiver.vital.util.entrypoint:sprokit_process_path',
                'applets=kwiver.vital.util.entrypoint:applets_path',
                'plugin_explorer=kwiver.vital.util.entrypoint:plugin_explorer_path'
                ],
            'kwiver.env.ld_library_path': [
                'kwiver_ld_library_path=kwiver.vital.util.entrypoint:get_library_path',
                ],
            'kwiver.env.logger_factory': [
                'vital_log4cplus_logger_factory=kwiver.vital.util.entrypoint:get_vital_logger_factory',
                ],
            'console_scripts': [
                'plugin_explorer=kwiver.kwiver_tools:plugin_explorer',
                'kwiver=kwiver.kwiver_tools:kwiver',
                ]
        },
    )
