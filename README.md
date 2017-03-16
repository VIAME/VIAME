
<img src="http://www.viametoolkit.org/wp-content/uploads/2016/08/viami_logo.png" alt="VIAME Logo" width="200" height="78">
<br>
VIAME is a computer vision library designed to integrate several image and
video processing algorithms together in a common distributed processing framework,
majorly targeting marine species analytics. The core infrastructure connecting
different system components is currently the KWIVER library, which can connect
C/C++, python, and matlab nodes together in a graph-like pipeline architecture.
For more information about KWIVER's capabilities, please
see [here](https://github.com/Kitware/kwiver/). Alongside the pipelined image
processing system are a number of standalone utilties for model training,
output detection visualization, and detector/tracker evaluation (a.k.a. scoring).
<p align="center">
<br>
<nobr>
<img src="http://www.viametoolkit.org/wp-content/uploads/2017/03/video_player.png" alt="vsPlay Example" width="430" height="284">
<img src="http://www.viametoolkit.org/wp-content/uploads/2017/03/image_player.png" alt="vpView Example" width="430" height="284">
</nobr>
</p>
<br>

Quick Build Instructions
------------------------

More in-depth build instructions can be found [here](doc/build_and_install_guide.rst)
and with additional tips [here](doc/build_tips_n_tricks.md).

VIAME itself can be built either as a super-build, which builds most of its
dependencies alongside itself, or standalone. To build viame as a super-build
requires [Git](https://git-scm.com/) and [CMake](https://cmake.org/). First,
run the following commands:

	git clone https://github.com/Kitware/VIAME.git [source-directory]

	cd [source-directory] && git submodule update --init --recursive

Next, create a build directory and run the following `cmake` command (or alternatively
use the cmake GUI):

	mkdir [build-directory] && cd [build-directory]

	cmake -DCMAKE_BUILD_TYPE:STRING=Release [source-directory]

Once your `cmake` command has completed, you can configure any build flags you want
using 'ccmake' or the cmake GUI, and then build with the following command:

	make -j8

Or alternatively by building it in Visual Studio or your compiler of choice on windows.
Currently VS2013 thru VS2017 is supported. If using CUDA, version 8.0 is desired,
along with Python 2.7. Other versions have yet to be tested extensively. The '-j8' tells
the build to run multi-threaded using 8 threads, this is useful for a faster build though
if you get an error it can be difficult to know where it was, in which case running just
'make' might be more helpful.

There are several optional arguments to viame which control which plugins get built, such as:

<center>

| Flag                         | Description                                                                           |
|------------------------------|---------------------------------------------------------------------------------------|
| VIAME_ENABLE_OPENCV          | Builds OpenCV and basic OpenCV processes (video readers, filters, simple GUIs)        |
| VIAME_ENABLE_VXL             | Builds VXL and basic VXL processes (video readers, image filters)                     |
| VIAME_ENABLE_CAFFE           | Builds Caffe and basic Caffe processes (pixel classifiers, required FRCNN dependency) |
| VIAME_ENABLE_PYTHON          | Turns on support for using python processes                                           |
| VIAME_ENABLE_MATLAB          | Turns on support for and installs all matlab processes                                |
| VIAME_ENABLE_SCALLOP_TK      | Builds Scallop-TK based object detector plugin                                        |
| VIAME_ENABLE_FASTER_RCNN     | Builds Faster-RCNN based object detector plugin                                       |
| VIAME_ENABLE_YOLO            | Builds YOLO (Darknet) object detector plugin                                          |
| VIAME_ENABLE_UW_CLASSIFIER   | Builds UW fish classifier plugin                                                      |

</center>

And a number of flags which control which system utilities and optimizations are built, e.g.:

<center>

| Flag                         | Description                                                                                 |
|------------------------------|---------------------------------------------------------------------------------------------|
| VIAME_ENABLE_CUDA            | Enables CUDA (GPU) optimizations across all processes (enables it in OpenCV, etc...)        |
| VIAME_ENABLE_CUDNN           | Enables CUDNN (GPU) optimizations across all processes                                      |
| VIAME_ENABLE_VIVIA           | Builds VIVIA Graphical User Interfaces (tools for annotation and detection viewing)         |
| VIAME_ENABLE_KWANT           | Builds KWANT detection and track evaluation tools (can evaluate detections or tracks)       |
| VIAME_ENABLE_DOCS            | Builds Doxygen class-level documentation for projects (puts in install share tree)          |
| VIAME_BUILD_DEPENDENCIES     | Build VIAME as a super-build, building all dependencies (default behavior)                  |
| VIAME_INSTALL_EXAMPLES       | Installs examples for the above modules into install/examples tree                          |
| VIAME_DOWNLOAD_MODELS        | Downloads pre-trained models for use with the examples and training new models              |

</center>

Quick Run Instructions
----------------------

All final compiled binaries are placed in the [build-directory]/install directory.
One way to test the system is to see if you can run the examples in the [build-directory]/install/examples
folder, for example, the pipelined object detectors. There are some environment variables
that need to be set up before you can run on Linux or Mac, which are all in the
install/setup_viame.sh script. This script is sourced in all of the example run
scripts, and similar paths are added in the generated windows .bat example scripts.

Another good initial test is to run the install/bin/plugin_explorer program. It
will generate a prodigious number of log messages and then list all the loadable
algorithms. The output should look as follows:

---- Algorithm search path

[build-directory]/install/lib/modules:

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

For a simple pipeline test, go to -

$ cd [build-directory]/install/examples/hello_world_pipeline/

or

$ cd [build-directory]/install/examples/detector_pipelines/

In those directories, run one of the detector pipelines (which ENABLE_FLAGS you
enabled will control which detector pipelines you can run). They can be
run via one of the scripts placed in the directory, or via:

'pipeline_runner -p [pipeline-file].pipe'

Output detections can then be viewed in the GUI, e.g., see:

[build-directory]/install/examples/visualizing_detections_in_gui/

Additional Documentation
------------------------

| Topic                                                                    |  Extras                                                   |
|--------------------------------------------------------------------------|------------------------------------------------------------|
| [Build and Install Guide](doc/build_and_install_guide.rst)               |  [Tips and Tricks](doc/build_tips_n_tricks.md)             |
| [Running Detectors](doc/detector_introduction.rst)                       |  [Examples](examples/detector_pipelines)                   |
| [How to Integrate Your Own Plugin](doc/cxx_plugin_creation.md)           |  [Examples](plugins)                                       |
| [Graphical User Interfaces for Visualization](doc/vpview_gui_introduction.md) |  [Examples](examples/visualizing_detections_in_gui)   |
| [Scoring and Evaluation of Detectors](doc/vpview_gui_introduction.md)     |  [Examples](examples/visualizing_detections_in_gui)       |





 










