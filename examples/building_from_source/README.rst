
==========================
Building VIAME From Source
==========================

See the platform-specific guides below, though the process is similar for each.
This document corresponds to the example `located online here`_ and also to the
building_from_source example folder in a VIAME installation.

.. _located online here: https://github.com/VIAME/VIAME/tree/master/examples/building_from_source


*****************
Building on Linux
*****************

These instructions build VIAME on a fresh machine. The release builds run on
Rocky Linux 8 and Ubuntu; other distributions differ mainly in how the
dependencies below are installed.

Install Dependencies
====================

The list is short, and it got much shorter: VIAME used to build its own
dependencies -- OpenCV, FFmpeg, VXL, Eigen, Boost, Qt, zlib and about fifty
more -- and the packages below were what that build itself needed. It does
not build them any more. Every VIAME library on a finished Linux install
links the C and C++ runtimes, libpython, libgomp for OpenMP, and, if CUDA is
enabled, CUDA's own libraries. Nothing else.

On Ubuntu:

.. code-block:: bash

   sudo apt-get install g++ cmake git zip wget curl \
     python3 python3-dev python3-numpy python3-pip

On Fedora or RHEL:

.. code-block:: bash

   sudo dnf -y groupinstall 'Development Tools'
   sudo dnf install -y gcc-c++ cmake git zip wget curl \
     python3 python3-devel python3-numpy python3-pip

On macOS, with Homebrew:

.. code-block:: bash

   brew install cmake git python@3.10 libomp

A C++17 compiler is required: GCC 8 or newer, Clang 7 or newer, or MSVC
2019 or newer. Python 3.10 or above is recommended, with its development
headers, pip and numpy -- the headers are what the extension modules compile
against, so ``python3-dev`` (``python3-devel`` on Fedora) is not optional if
``VIAME_ENABLE_PYTHON`` is on. Anaconda3 works too, as does any other
distribution that ships headers.

The remaining python packages -- torch, opencv-python, and the rest -- are
wheels, installed by the build from the lock files in ``python/requirements``.
They are not system packages and should not be installed with apt.

If using VIAME_ENABLE_CUDA for GPU support, you should install CUDA (version 12.6 is
preferred; version 11.0 or above is required). Other versions may work depending
on your build settings but are not officially supported yet. Link to NVIDIA's site:

.. code-block:: bash

   https://developer.nvidia.com/cuda-toolkit-archive

Install CMake
=============

Configuring from a preset needs CMake 3.25 or newer; the build servers use
4.2. Try the package manager's first and check ``cmake --version``. If it is
older, download a binary release from ``https://cmake.org/download`` -- on
Linux the ``cmake-<version>-linux-x86_64.sh`` installer needs nothing else --
or build it from source:

.. code-block:: bash

   tar zxfv cmake-4.2.0.tar.gz
   cd cmake-4.2.0
   ./bootstrap --system-curl
   make -j8
   sudo make install

Clone the Source Code
=====================

VIAME's remaining third-party sources -- DIVE and the patched python
packages -- are git submodules, so clone recursively:

.. code-block:: bash

   git clone https://github.com/VIAME/VIAME.git src
   cd src
   git submodule update --init --recursive

Build VIAME
===========

A build is configured from a preset in ``CMakePresets.json`` at the top of the
source tree. From the source directory:

.. code-block:: bash

   cmake --preset linux-gpu
   cmake --build --preset linux-gpu

The first command configures ``build/linux-gpu``; the second builds it and
installs into ``build/linux-gpu/install``. To use the install:

.. code-block:: bash

   source build/linux-gpu/install/setup_viame.sh
   viame help

``ctest --preset linux-gpu`` runs the CRITICAL tests against it: the example
pipelines a release has to pass.

+------------------+----------------------------------------------------------------------------+
| Preset           | What it builds                                                             |
+==================+============================================================================+
| ``linux-gpu``    | The Linux desktop release: CUDA, its own Python, DIVE, default model packs |
+------------------+----------------------------------------------------------------------------+
| ``linux-cpu``    | The same without CUDA or the PyTorch components that need it               |
+------------------+----------------------------------------------------------------------------+
| ``windows-gpu``  | The Windows desktop release                                                |
+------------------+----------------------------------------------------------------------------+
| ``windows-cpu``  | The Windows desktop release without CUDA                                   |
+------------------+----------------------------------------------------------------------------+
| ``macos``        | macOS, CPU only                                                            |
+------------------+----------------------------------------------------------------------------+
| ``docker``       | Inside a Docker image: system Python, no DIVE                              |
+------------------+----------------------------------------------------------------------------+
| ``docker-web``   | The VIAME-Web image                                                        |
+------------------+----------------------------------------------------------------------------+

``cmake --list-presets`` shows them. Each is composed from smaller hidden
presets -- ``base``, ``gpu``, ``cpu``, ``desktop``, ``linux``, ``container``,
``web`` -- and a later one in a preset's ``inherits`` list is overridden by an
earlier one, so the list reads most specific first.

Any setting can be changed on the command line, after the preset:

.. code-block:: bash

   cmake --preset linux-gpu -DVIAME_ENABLE_PYTORCH-MMDET=OFF

For a configuration of your own, write a ``CMakeUserPresets.json`` beside
``CMakePresets.json``, inheriting from one of these. It is yours: git ignores
it.

Three settings in the release presets need something from the machine:

- ``linux-gpu`` and ``linux-cpu`` build the DIVE desktop client from source,
  which needs Node.js 22 or newer. ``-DVIAME_BUILD_DIVE_FROM_SOURCE=OFF``
  downloads a released DIVE instead.
- The desktop presets build their own CPython
  (``VIAME_BUILD_PYTHON_FROM_SOURCE``) so the install carries it.
  ``-DVIAME_BUILD_PYTHON_FROM_SOURCE=OFF`` uses the Python CMake finds.
- Python packages -- torch and the rest -- are installed from the lock files in
  ``python/requirements`` during the build, which needs the network.
  ``-DVIAME_INSTALL_PYTHON_DEPS=OFF`` skips that for a Python environment you
  manage yourself.

The options most often changed:

+----------------------------------+------------------------------------------------------------------+
| Option                           | What it does                                                     |
+==================================+==================================================================+
| ``VIAME_ENABLE_CUDA``            | GPU support, with ``VIAME_ENABLE_CUDNN``                         |
+----------------------------------+------------------------------------------------------------------+
| ``VIAME_ENABLE_PYTHON``          | Python algorithms and processes                                  |
+----------------------------------+------------------------------------------------------------------+
| ``VIAME_ENABLE_PYTORCH``         | PyTorch detectors, trackers and classifiers; each family also    |
|                                  | has its own ``VIAME_ENABLE_PYTORCH-<NAME>``                      |
+----------------------------------+------------------------------------------------------------------+
| ``VIAME_ENABLE_ONNX``            | ONNX Runtime inference                                           |
+----------------------------------+------------------------------------------------------------------+
| ``VIAME_ENABLE_DARKNET``         | The Darknet YOLO detector                                        |
+----------------------------------+------------------------------------------------------------------+
| ``VIAME_ENABLE_SVM``             | libsvm classifiers                                               |
+----------------------------------+------------------------------------------------------------------+
| ``VIAME_ENABLE_DIVE``            | Install the DIVE annotation tool                                 |
+----------------------------------+------------------------------------------------------------------+
| ``VIAME_ENABLE_TESTS``           | Build the test suite                                             |
+----------------------------------+------------------------------------------------------------------+
| ``VIAME_ENABLE_DOCS``            | Build the documentation                                          |
+----------------------------------+------------------------------------------------------------------+
| ``VIAME_DOWNLOAD_MODELS``        | Download model packs; each has ``VIAME_DOWNLOAD_MODELS-<NAME>``  |
+----------------------------------+------------------------------------------------------------------+

The build compiles several hundred pybind11 binding sources, each needing
around a gigabyte of memory. On a machine with little memory, or one doing
other work, limit the parallelism: ``cmake --build --preset linux-gpu -j2``.

.. _mac-label:

*******************
Building on Mac OSX
*******************

Install Xcode's command line tools, Homebrew, and the packages listed above,
then follow the Linux instructions with the ``macos`` preset:

.. code-block:: bash

   cmake --preset macos
   cmake --build --preset macos

.. _windows-label:

*******************
Building on Windows
*******************

Install Visual Studio 2022 or newer with the C++ workload, CMake 3.25 or
newer, git, and, for GPU support, CUDA 12.6 with cuDNN. Clone recursively as
above, then from a *Developer Command Prompt*, in the source directory:

.. code-block:: bat

   cmake --preset windows-gpu
   cmake --build --preset windows-gpu

``windows-cpu`` builds without CUDA. Visual Studio can also open the source
folder directly and offers the presets in its configuration list.

Windows limits path length, and a deep build folder is the usual cause of
errors in the python bindings. Keep the source tree high in the folder tree,
e.g. ``C:\VIAME``.

.. _tips-label:

**************
Updating VIAME
**************

After pulling or switching branches, update the submodules too:

.. code-block:: bash

   git submodule update --init --recursive

If that fails with "cannot fetch hash", the address of a submodule has
changed; run ``git submodule sync`` first.

.. _issues-label:

******************
Known Build Issues
******************

**Issue:** the compiler is killed, e.g. ``c++: internal compiler error: Killed
(program cc1plus)``.

**Solution:** the machine ran out of memory. Build with less parallelism
(``cmake --build --preset <preset> -j2``), or add memory.

**Issue:** ``Unable to locate CUDNN library``.

**Solution:** cuDNN is installed separately from CUDA. Install it, or point
CMake at it with ``-DCUDNN_ROOT_DIR=<path>``, or configure without it:
``-DVIAME_ENABLE_CUDNN=OFF``.

**Issue:** ``nvcc fatal : Visual Studio configuration file 'vcvars64.bat' could
not be found``.

**Solution:** configure from a Developer Command Prompt, where Visual Studio's
environment is already set up.
