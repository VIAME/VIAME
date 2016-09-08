
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

	cd /path/to/viame/source/directory && git submodule update --init --recursive

Then, create a build directory and run the following `cmake` command (or alternatively
use the cmake GUI):

	cmake -DCMAKE_BUILD_TYPE:STRING=Release /path/to/viame/source/directory

Once your `cmake` command has completed, you can build with the following
command if on linux:

	make -j8

Or alternatively by building it in visual studio or your compiler of choice on windows.

There are currently several optional arguments to viame, such as:

| Flag                         | Description                                       |
|------------------------------|---------------------------------------------------|
| VIAME_ENABLE_MATLAB          | Turns on support for using matlab processes       |
| VIAME_ENABLE_OPENCV          | Builds OpenCV and basic OpenCV processe           |
| VIAME_ENABLE_VXL             | Builds VXL and basic VXL processes                |
| VIAME_ENABLE_CAFFE           | Builds Caffe and basic Caffe processes            |
| VIAME_ENABLE_PYTHON          | Turns on support for using python processes       |
| VIAME_ENABLE_VIVIA           | Builds VIVIA GUIs                                 |
| VIAME_ENABLE_SCALLOP_TK      | Builds all ScallopTK-based plugins                |
| VIAME_DISABLE_GPU_SUPPORT    | Builds all VIAME processes without GPU support    |
| VIAME_DISABLE_FFMPEG_SUPPORT | Builds all VIAME processes without FFMPEG support |

Quick Run Instructions
----------------------

One way to test the system is to see if you can run a pipelined application.
There are some environment variables that need to be set up before you can run,
which are all in the install/setup_viame.sh script.

A good initial test is to run the install/bin/plugin_explorer program. It
will generate a prodigious number of log messages and then list all the loadable
algorithms. The output should look as follows:

---- Algorithm search path

/disk2/projects/NOAA/VIAME/build/install/lib/modules:

---- Registered module names:

*  kwiver_algo_matlab_plugin
*  kwiver_algo_ocv_plugin
*  kwiver_algo_plugin
*  kwiver_algo_vxl_plugin
*  viame_scallop_tk_plugin
            etc...


---- registered algorithms (type_name:impl_name)

*  analyze_tracks:ocv
*  bundle_adjust:hierarchical
*  bundle_adjust:vxl
*  close_loops:bad_frames_only
*  close_loops:exhaustive
*  close_loops:keyframe
*  close_loops:multi_method
*  close_loops:vxl_homography_guided
*  compute_ref_homography:core
*  convert_image:bypass
*  detect_features:ocv_BRISK
*  detect_features:ocv_FAST
            etc...


The modules loaded list represents the shared objects that have been detected
and loaded. Each shared object can contain multiple algorithms. The algorithm
list shows each concrete algorithm that could be loaded. Check the log messages
to see if there are any libraries that could not be located.

Each algorithm listed consists of two names. The first name is the type of
algorithm and the second is the actual implementation type. For example the
entry image_object_detector:hough_circle_detector indicates that it implements
the image_object_detector interface and it is a hough_circle_detector.

Algorithms can be instantiated in any program and use a configuration based
approach to select which concrete implementation to instantiate.

The next thing to check is to verify the process loading environment by running
VIAME/install/bin/processopedia. This program will search for and load sprokit
processes.

The output should appear as follows (omitting the log messages):

* collate: Collates data from multiple worker processes
* compute_homography: Compute a frame to frame homography based on tracks
* detect_features: Detect features in an image that will be used for stabilization
* distribute: Distributes data to multiple worker processes
* draw_detected_object_boxes: Draw detected object boxes on images.
* draw_tracks: Draw feature tracks on image
* extract_descriptors: Extract descriptors from detected features
* feature_matcher: Match extracted descriptors and detected features
              etc...

We will be using the image_object_detector process type in a pipeline to apply a
detector to a stream of images. This process wraps the image_object_detector 
algorithm interface in a process. The process can be configured to instantiate
any available detector implementation.

For a simple pipeline test, go to -

$ cd VIAME/source/packages/kwiver/sprokit/pipelines/examples/hough_detector

In that directory, run the following command (or look at run_pipe.sh)

$ pipeline_runner -p hough_detector.pipe

The results should be an image displayed with a box around each can end.

This is a good check of the underlying components.
