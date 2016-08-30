
VIAME
=====

VIAME is a computer vision library designed to integrate several image and
video processing algorithms together in a common distributed processing framework,
majorly targeting marine species analytics. The core infrastructure connecting
different system components is currently the KWIVER library, which can connect
C/C++, python, and matlab nodes together in a graph-like pipeline architecture.
For more information about KWIVER's capabilities, please
see [here](https://github.com/Kitware/kwiver/).


Quick Build Instructions
------------------------

More in-depth build instructions can be found [here](doc/install_guide.rst).

VIAME itself can be built either as a super-build, which builds most of its
dependencies alongside itself, or standalone. To build viame as a super-build
requires [Git](https://git-scm.com/) and [CMake](https://cmake.org/). Run the
following commands:

	git clone https://github.com/Kitware/VIAME.git /path/to/viame/source/directory

	cd /path/to/viame/source/directory && git submodule update --init

Then, create a build directory and run the following `cmake` command (or alternatively
use the cmake GUI):

	cmake -DCMAKE_BUILD_TYPE:STRING=Release /path/to/viame/source/directory

Once your `cmake` command has completed, you can build with the following
command if on linux:

	make

Or alternatively by building it in visual studio or your compiler of choice on windows.

There are currently several optional arguments to viame, such as "VIAME_ENABLE_MATLAB",
"VIAME_ENABLE_OPENCV", "VIAME_ENABLE_VXL",  and "VIAME_ENABLE_CAFFE" to turn on
support for different components.
