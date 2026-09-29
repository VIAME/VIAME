
==========================
Building VIAME From Source
==========================

See the platform-specific guides below, though the process is similar for each.
This document corresponds to the example `located online here`_ and also to the
building_from_source example folder in a VIAME installation.

.. _located online here: https://github.com/VIAME/VIAME/tree/main/examples/building_from_source


**************
VIAME Versions
**************

There are 2 versions of VIAME, each kept on its own branch of the repository:

+-----------+--------------------------------------------------------------------------------+
| Branch    | Description                                                                    |
+===========+================================================================================+
| main      | The full version of VIAME, which the desktop installers and docker images are  |
|           | built from                                                                     |
+-----------+--------------------------------------------------------------------------------+
| main-lite | A reduced version of VIAME with fewer dependencies, used for the PyPI packages |
|           | and for embedded systems                                                       |
+-----------+--------------------------------------------------------------------------------+

The instructions below were written for ``main``. To get the reduced version instead, add
``-b main-lite`` to the clone command.


*****************
Building on Linux
*****************

These instructions are designed to help build VIAME on a fresh machine. They were written for
and tested on Ubuntu 24.04. Other Linux machines will have similar directions, but some steps
(particularly the dependency install) may not be exactly identical. VIAME is also built on
Ubuntu 22.04 and Rocky Linux 9.

Install Dependencies
====================

Different Linux distributions may have different packages already installed, or may
use a different package manager than apt, but on Ubuntu this should help to provide
a starting point:

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install -y git zip wget tar curl bzip2 gcc g++ gfortran libgl1-mesa-dev \
     libexpat1-dev libgtk2.0-dev libxt-dev libxml2-dev liblapack-dev openssl libssl-dev \
     libcurl4-openssl-dev zlib1g-dev libbz2-dev liblzma-dev

And on Rocky Linux 9:

.. code-block:: bash

   sudo dnf -y groupinstall 'Development Tools'
   sudo dnf install -y zip git wget zlib zlib-devel zstd freeglut-devel freetype-devel \
     mesa-libGLU-devel libffi-devel libXt-devel libXmu-devel libXi-devel expat-devel \
     readline-devel curl-devel atlas-devel file which bzip2 bzip2-devel xz-devel perl perl-IPC-Cmd

If using VIAME_ENABLE_PYTHON, Python 3.10 or above is required, along with its development
packages, pip, and numpy. Ubuntu 24.04 provides Python 3.12, which can be installed with:

.. code-block:: bash

   sudo apt-get install -y python3 python3-dev python3-pip python3-numpy python-is-python3

Ubuntu 24.04 marks its system Python as externally managed, so pip refuses to install packages
into it. The VIAME build installs the Python packages it needs into its own install tree and
accounts for this by itself. Only packages you add to the system Python by hand need either a
virtual environment or the ``--break-system-packages`` option of pip. Other Python
distributions, such as `Anaconda3 <https://repo.anaconda.com/archive/>`__, can be used in
place of the system one.

If using VIAME_ENABLE_CUDA for GPU support, you should install CUDA and cuDNN (CUDA 12.6 with
cuDNN 9 is preferred; CUDA 11.0 or above is required). Other versions may work depending
on your build settings but are not officially supported yet. Link to NVIDIA's site:

.. code-block:: bash

   https://developer.nvidia.com/cuda-toolkit-archive

Install CMake
=============

Building VIAME requires CMake 3.16 or above. Ubuntu 24.04 provides CMake 3.28, so installing
it with the package manager is enough:

.. code-block:: bash

   sudo apt-get install -y cmake
   cmake --version

On a distribution whose CMake is too old, or to use the latest release (4.4.3 at the time of
writing), go to the cmake website, ``https://cmake.org/download``, and download the appropriate
binary distribution (for Linux, cmake-4.4.3-linux-x86_64.sh), or for windows the .msi or .zip
installer. Lastly the source version could be built using the below instructions, though this
is usually not necessary if a binary version is available for your platform.

.. code-block:: bash

   wget https://cmake.org/files/v4.4/cmake-4.4.3.tar.gz
   tar zxfv cmake-4.4.3.tar.gz
   cd cmake-4.4.3
   ./bootstrap --system-curl
   make -j8
   sudo make install

These instructions build the source code into a working executable and install it into
/usr/local/bin, which comes before the package manager's version in the default PATH on
Ubuntu.

Clone the Source Code
=====================

With all our dependencies installed, we need to build the environment for VIAME
itself. VIAME uses git submodules rather than requiring the user to grab each
repository totally separately. To prepare the environment and obtain all the
necessary source code, use the following commands. Note that you can change ``src``
to whatever you want to name your VIAME source directory.

.. code-block:: bash

   git clone https://github.com/VIAME/VIAME.git src
   cd src
   git submodule update --init --recursive

Build VIAME
===========

VIAME may be built with a number of optional plugins--VXL, PyTorch, OpenCV,
Scallop-TK, and Matlab--with a corresponding option called VIAME_ENABLE_[option],
in all caps. For each plugin to install, you need a cmake build flag setting the
option. The flag looks like ``-DVIAME_ENABLE_OPENCV:BOOL=ON``, of course changing
OPENCV to match the plugin. Multiple plugins may be used, or none. If uncertain what
to turn on, it's best to just leave the default enable and disable flags which will
build most (though not all) functionalities. At a minimum, these are core components
we recommend leaving turned on:


+------------------------------+---------------------------------------------------------------------------------------+
| Flag                         | Description                                                                           |
+==============================+=======================================================================================+
| VIAME_ENABLE_OPENCV          | Builds OpenCV and basic OpenCV processes (video readers, simple GUIs)                 |
+------------------------------+---------------------------------------------------------------------------------------+
| VIAME_ENABLE_VXL             | Builds VXL and basic VXL processes (video readers, image filters)                     |
+------------------------------+---------------------------------------------------------------------------------------+
| VIAME_ENABLE_PYTHON          | Turns on support for using python processes (multiple algorithms)                     |
+------------------------------+---------------------------------------------------------------------------------------+
| VIAME_ENABLE_PYTORCH         | Installs all pytorch processes (detectors, trackers, classifiers)                     |
+------------------------------+---------------------------------------------------------------------------------------+

And a number of flags which control which system utilities and optimizations are built, e.g.:

+------------------------------+---------------------------------------------------------------------------------------------+
| Flag                         | Description                                                                                 |
+==============================+=============================================================================================+
| VIAME_ENABLE_CUDA            | Enables CUDA (GPU) optimizations across all processes (OpenCV, Torch, etc...)               |
+------------------------------+---------------------------------------------------------------------------------------------+
| VIAME_ENABLE_CUDNN           | Enables CUDNN (GPU) optimizations across all processes                                      |
+------------------------------+---------------------------------------------------------------------------------------------+
| VIAME_ENABLE_VIVIA           | Builds VIVIA GUIs (tools for making annotations and viewing detections)                     |
+------------------------------+---------------------------------------------------------------------------------------------+
| VIAME_ENABLE_DOCS            | Builds Doxygen class-level documentation for projects (puts in install share tree)          |
+------------------------------+---------------------------------------------------------------------------------------------+
| VIAME_BUILD_DEPENDENCIES     | Build VIAME as a super-build, building all dependencies (default behavior)                  |
+------------------------------+---------------------------------------------------------------------------------------------+
| VIAME_INSTALL_EXAMPLES       | Installs examples for the above modules into install/examples tree                          |
+------------------------------+---------------------------------------------------------------------------------------------+
| VIAME_DOWNLOAD_MODELS        | Downloads pre-trained models for use with the examples and training new models              |
+------------------------------+---------------------------------------------------------------------------------------------+

And lastly, a number of flags which build algorithms with more specialized functionality:

+------------------------------+---------------------------------------------------------------------------------------------+
| Flag                         | Description                                                                                 |
+==============================+=============================================================================================+
| VIAME_ENABLE_TENSORFLOW      | Builds TensorFlow object detector plugin                                                    |
+------------------------------+---------------------------------------------------------------------------------------------+
| VIAME_ENABLE_DARKNET         | Builds Darknet (YOLO) object detector plugin                                                |
+------------------------------+---------------------------------------------------------------------------------------------+
| VIAME_ENABLE_SEAL            | Builds Seal Multi-Modality GUI                                                              |
+------------------------------+---------------------------------------------------------------------------------------------+
| VIAME_ENABLE_UW_CLASSIFIER   | Builds UW fish classifier plugin                                                            |
+------------------------------+---------------------------------------------------------------------------------------------+
| VIAME_ENABLE_MATLAB          | Turns on support for and installs all matlab processes                                      |
+------------------------------+---------------------------------------------------------------------------------------------+

VIAME can be built either in the source directory tree or in a separate build
directory (recommended). Replace "[build-directory]" with your location of choice,
and run the following commands:

.. code-block:: bash

   mkdir [build-directory]
   cd [build-directory]
   cmake [build_flags] [path_to_source_tree]
   make -j8 # or just make for a unthreaded build

Depending on which enable flags you have set and your system configuration, you may
need to set additional cmake variables to point to dependency locations. An example
is below for a system with CUDA, Python, and Matlab enabled, though the versions are
old. Please do not use CUDA below 11.0 or Python below 3.10 anymore.

.. image:: https://raw.githubusercontent.com/VIAME/VIAME/main/docs/manual/_static/images/cmake-options.jpg
   :width: 40%
   :align: center

.. _mac-label:

*******************
Building on Mac OSX
*******************

Building on Mac is very similar to Linux, minus the dependency install stage.
Mac builds are CPU-only, as CUDA is not available there, and the release builds use a
Miniconda Python environment. Make sure you have a C/C++ development environment set up,
install git, install cmake either from the source or a using a binary installer, and
lastly, follow the same Linux build instructions above.

.. _windows-label:

*******************
Building on Windows
*******************

Building on windows can be very similar to Linux if using a shell like cygwin
(``https://www.cygwin.com/``), though if not you may want to go grab the GUI
ersions of CMake (``https://cmake.org/``) and TortoiseGit (``https://tortoisegit.org/``).
Currently Visual Studio 2026 is used for the release builds and is the most tested version.

First do a Git clone of the source code for VIAME. If you have TortoiseGit this
involves right clicking in your folder of choice, selecting Git Clone, and then
entering the URL to VIAME (``https://github.com/VIAME/VIAME.git``) and the location
of where you want to put the downloaded source code.

Next, do a git submodule update to pull down all required packages. In TortoiseGit
right click on the folder you checked out the source into, move to the TortoiseGit
menu section, and select ``Submodule Update``.

Next, install any required dependencies for items you want to build. If using CUDA,
version 12.6 is preferred (version 11.0 or above is required), along with Python 3.10+.
Other versions have yet to be tested extensively, though may work. On Windows it can
also be beneficial to use Anaconda to get multiple python packages. Boost Python
(turned on by default when Python is enabled) requires Numpy and a few other dependencies.

Finally, create a build folder and run the CMake GUI (``https://cmake.org/runningcmake/``).
Point it to your source and build directories, select your compiler of choice, and
setup and build flags you want.

The biggest build issues on Windows arise from building VIAME in super-build and
exceeded the windows maximum folder path length. This will typically manifest as build
errors in the kwiver python libraries. To bypass these errors you have 2 options:

1. Build VIAME in as high level as possible (e.g. C:/VIAME) or, alternatively
2. Set the VIAME_BUILD_KWIVER_DIR path to be something small outside of your
   superbuild location, e.g. C:/tmp/kwiver to bypass path length limits. This
   is performed, for example, in the nightly build server cmake script as an
   example https://github.com/VIAME/VIAME/blob/main/cmake/build_server_windows.cmake


.. _tips-label:

**************
Updating VIAME
**************

If you already have a checkout of VIAME and want to switch branches or
update your code, it is important to re-run:

``git submodule update --init --recursive``

After switching branches to ensure that you have on the correct hashes
of sub-packages within the build (e.g. fletch or KWIVER). Very rarely
you may also need to run:

``git submodule sync``

Just in case the address of submodules has changed. You only need to
run this command if you get a "cannot fetch hash #hashid" error.

********************
Build Tips 'n Tricks
********************

**Super-Build Optimizations:**

When VIAME is built as a super-build, multiple solutions or makefiles are generated
for each individual project in the super-build. These can be opened up if you want
to experiment with changes in one and not rebuild the entire superbuild. VIAME
places these projects in [build-directory]/build/src/\* and fletch in
[build-directory]/build/src/fletch-build/build/src/\*. You can also run ccmake or
the cmake GUI in these locations, which can let you manually change the build settings
for sub-projects (say, for example, if one doesn't build).


**Python:**

The system Python is used by default, which is 3.12 on Ubuntu 24.04, and versions 3.10 and
above are supported. Alternatively, ``VIAME_BUILD_PYTHON_FROM_SOURCE`` builds Python inside
of VIAME, version 3.12 by default. Which versions work depends on your build settings,
operating system, and which dependency projects are turned on.


.. _issues-label:

******************
Known Build Issues
******************

**Issue:**

When compiling with CUDA turned on:

.. code-block:: console

   nvcc fatal   : Visual Studio configuration file 'vcvars64.bat' could not be found for
   installation at 'Microsoft Visual Studio XX.0/VC/bin/x86_amd64/../../..'

or similar.

**Solution:**

Express/Community versions of visual studio don't ship with a file called vcvars64.bat
You can add one manually be placing a bat file called 'vcvars64.bat' in folder
'Microsoft Visual Studio XX.0\VC\bin\amd64' for your version of visual studio. This
file should contain just a single line:

``CALL setenv /x64``


**Issue:**

Boost fails to build early with error in \*_out.txt:

.. code-block:: console

   c++: internal compiler error: Killed (program cc1plus)

**Solution:**

You are likely running out of memory and your C++ compiler is crashing (common on VMs
with a small amount of memory). Increase the amount of memory availability to your VM or
buy a better computer if not running a VM with at least 1 Gb of RAM.


**Issue:**

On Windows with Python enabled: ``error LNK1104: cannot open file 'python312_d.lib'``

**Solution:**

If you want to link against python in debug mode, you'll have to build Python itself
to enable debug libraries, as the default python distributions do not contain them.
Alternatively switch to Release or RelWDebug modes.


**Issue:**

.. code-block:: console

   ImportError: No module named numpy.distutils

**Solution:**

You have python installed, but not numpy. Install numpy.


**Issue:**

``cannot find cublas_v2.h`` or linking issues against CUDA

**Solution:**

VIAME contains a ``VIAME_DISABLE_GPU_SUPPORT`` flag due to numerous issues relating to
GPU code building. Alternatively you can debug the issue (incorrect CUDA drivers for
OpenCV, Torch, etc...), or alternatively not having your CUDA headers set to be in your include path.


**Issue:**

.. code-block:: console

   CMake Error at CMakeLists.txt:200 (message):
     Unable to locate CUDNN library

**Solution:**

You have enabled CUDNN but the system is unable to locate CUDNN, as the message says.

Note CUDNN is installed separately from CUDA, they are different things.

You need to set the VIAME flag CUDNN_LIBRARY to something like /usr/local/cuda/lib64/libcudnn.so.
Alternatively you can set CUDNN_ROOT to /usr/local/cuda/lib64 manually if that's where you installed it.


**Issue:**

When ``VIAME_ENABLE_DOC`` is turned on and doing a multi-threaded build, sometimes the build fails.

**Solution:**

Run ``make -jX`` multiple times, or don't run ``make -jX`` when ``VIAME_ENABLE_DOCS`` is enabled.


**Issue:**

CMake says it cannot find MATLAB

**Solution:**

Make sure your matlab CMake paths are set to something like the following

.. code-block:: console

   Matlab_ENG_LIBRARY:FILEPATH=[matlab_install_loc]/bin/glnxa64/libeng.so
   Matlab_INCLUDE_DIRS:PATH=[matlab_install_loc]/extern/include
   Matlab_MEX_EXTENSION:STRING=mexa64
   Matlab_MEX_LIBRARY:FILEPATH=[matlab_install_loc]/bin/glnxa64/libmex.so
   Matlab_MX_LIBRARY:FILEPATH=[matlab_install_loc]/bin/glnxa64/libmx.so
   Matlab_ROOT_DIR:PATH=[matlab_install_loc]

