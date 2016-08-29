# VIAME

VIAME is a processing framework designed to integrate several image and
video processing algorithms in a distributed processing framework. The
core infrastructure connecting different system components is currently
the KWIVER library. For more information please see [here](https://github.com/Kitware/kwiver/)
until this notice can be further flushed out.

VIAME itself can be built either as a super-build, which builds all of its
dependencies alongside itself, or standalone. To build viame:

	git clone https://github.com/Kitware/VIAME.git

	git submodule update --init

Then, create a build directory and run the following `cmake` command:

	cmake -DCMAKE_BUILD_TYPE:STRING=Release /path/to/viame/source/directory

Once your `cmake` command has completed, you can build with the following
command if on linux:

	make

Or alternatively by building in visual studio on windows.

There are currently several optional arguments to viame, such as "VIAME_ENABLE_MATLAB",
"VIAME_ENABLE_OPENCV", "VIAME_ENABLE_VXL",  and "VIAME_ENABLE_CAFFE" to turn on
support for different cpomponents.
