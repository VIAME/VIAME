
<img src="http://www.viametoolkit.org/wp-content/uploads/2016/08/viami_logo.png" alt="VIAME Logo" width="200" height="78">
<br>
VIAME is a computer vision library designed to integrate several image and
video processing algorithms together in a common distributed processing framework,
majorly targeting marine species analytics. As it contains many common algorithms
and compiles several other popular repositories together as a part of its build process,
VIAME is also useful as a general computer vision toolkit. The core infrastructure connecting
different system components is currently the KWIVER library, which can connect C/C++, python,
and matlab nodes together in a graph-like pipeline architecture. Alongside the pipelined
image processing system are a number of standalone utilties for model training, output detection
visualization, groundtruth annotation, detector/tracker evaluation (a.k.a. scoring),
image/video search, and rapid model generation.

Example Capabilities
--------------------
<p align="left">
<br>
<nobr>
<img src="http://www.viametoolkit.org/wp-content/uploads/2018/06/viame-gh-splash-04.png" alt="Search Example" width="303" height="183">
<img src="http://www.viametoolkit.org/wp-content/uploads/2018/06/tracks.png" alt="Tracking Example" width="235" height="183">
<img src="http://www.viametoolkit.org/wp-content/uploads/2018/06/viame-gh-splash-02.png" alt="Detection Example" width="303" height="183">
</nobr>
</p>
<p align="left">
<nobr>
<img src="http://www.viametoolkit.org/wp-content/uploads/2018/06/viame-gh-splash-03.png" alt="Measurement Example" width="415" height="148">
<img src="http://www.viametoolkit.org/wp-content/uploads/2018/06/viame-gh-splash-01.png" alt="Query Example" width="429" height="148">
</nobr>
</p>

Documentation
-------------

The [VIAME manual](http://viame.readthedocs.io/en/latest/) is more comprehensive,
but select entries are also listed below, which include some run examples:


[Build and Install Guide](examples/building_and_installing_viame) <> 
[All Examples](https://github.com/Kitware/VIAME/tree/master/examples) <> 
[Core Class and Pipeline Info](http://kwiver.readthedocs.io/en/latest/architecture.html) <> 
[Object Detector Examples](examples/object_detection) <br />
[Stereo Measurement Examples](examples/measurement_using_stereo) <> 
[Embedding Detectors in C++ Code](examples/using_detectors_in_cxx_code) <>
[How to Integrate Your Own Plugin](examples/hello_world_pipeline) <br />
[Example Integrations](plugins) <>
[Example Plugin Templates](plugins/templates) <> 
[GUIs for Visualization and Annotation](examples/annotation_and_visualization) <> 
[Detector Training API](examples/object_detector_training) <br />
[Video Search and Rapid Model Generation](examples/search_and_rapid_model_generation) <> 
[Scoring and Evaluation of Detectors](examples/scoring_and_roc_generation) <>
[KWIVER Overview](https://github.com/Kitware/kwiver)

Pre-Built Binaries
------------------

For a full installation guide, [see here](https://data.kitware.com/api/v1/item/5b4681808d777f2e6225a29f/download).
In summary, extract the binaries and place them in a directory of your choosing, for
example C:\Program Files\VIAME on Windows or /opt/noaa/viame on Linux.
Next, set the PYTHON_INSTALL_DIR and CUDA_INSTALL_DIR variables at the top
of the setup_viame.sh (Linux) or setup_viame.bat (Windows) script in the root install
folder to point to the location of your installed Anaconda and CUDA distributions.
Lastly, run through some of the examples to validate the installation.

**Installation Requirements:** <br>
RHEL/CentOS 7 64-Bit, Ubuntu 16.04 64-Bit, Windows 7, 8, or 10 64-Bit <br>
[Anaconda3 5.2.0 x86_64](https://repo.continuum.io/archive/) (Note: Anaconda**3 x86_64**, not Anaconda2 or x86) <br>
[CUDA 8.0 GA2 x86_64](https://developer.nvidia.com/cuda-toolkit-archive) (If you use GPU support) <br>

**Installation Recommendations:** <br>
A CUDA-enabled GPU with 8 Gb or more VRAM <br>

**Linux Binaries:** <br>
[VIAME v0.9.7.5 Ubuntu 16.04, 64-Bit, GPU Enabled, CUDA 8.0, Python 3.6, Mirror1](https://data.kitware.com/api/v1/item/5b69d8548d777f06857c1f97/download) <br>
[VIAME v0.9.7.5 Ubuntu 16.04, 64-Bit, GPU Enabled, CUDA 8.0, Python 3.6, Mirror2](https://drive.google.com/open?id=1kD3soOyYE6NqHf7L7SOhXDlk0gDP-3JM) <br>
[VIAME v0.9.7.7 RHEL/CentOS 7, 64-Bit, GPU Enabled, CUDA 8.0, Python 3.6, Mirror1](https://data.kitware.com/api/v1/item/5b73145c8d777f06857c45eb/download) <br>
[VIAME v0.9.7.7 RHEL/CentOS 7, 64-Bit, GPU Enabled, CUDA 8.0, Python 3.6, Mirror2](https://drive.google.com/open?id=1tf4OV05NRzv__J-LI4S5-huLpRyQHhWK)

**Windows Binaries:** <br>
[VIAME v0.9.7.6 Windows 7/8/10, 64-Bit, GPU Enabled, CUDA 8.0, Python 3.6, Mirror1](https://data.kitware.com/api/v1/item/5b6a3a208d777f06857c1f9f/download) <br>
[VIAME v0.9.7.6 Windows 7/8/10, 64-Bit, GPU Enabled, CUDA 8.0, Python 3.6, Mirror2](https://drive.google.com/open?id=1dHoK-CUxjY-6dm0-rtlO7OcLyvs1Qm9B) <br>
[VIAME v0.9.7.6 Windows 7/8/10, 64-Bit, CPU Only, Python 3.6, Mirror1](https://data.kitware.com/api/v1/item/5b6a3e968d777f06857c1fa3/download) <br>
[VIAME v0.9.7.6 Windows 7/8/10, 64-Bit, CPU Only, Python 3.6, Mirror2](https://drive.google.com/open?id=1mamA9wEv8M5v2SQyL7zXe_OindIYZnji)


Quick Build Instructions
------------------------

More in-depth build instructions can be found [here](examples/building_and_installing_viame), but
VIAME itself can be built either as a super-build, which builds most of its
dependencies alongside itself, or standalone. To build VIAME requires, at a minimum,
[Git](https://git-scm.com/), [CMake](https://cmake.org/), and a C++ compiler.
If using the command line, run the following commands, only replacing [source-directory] and
[build-directory] with locations of your choice:

	git clone https://github.com/Kitware/VIAME.git [source-directory]

	cd [source-directory] && git submodule update --init --recursive

Next, create a build directory and run the following `cmake` command (or alternatively
use the cmake GUI if you are not using the command line interface):

	mkdir [build-directory] && cd [build-directory]

	cmake -DCMAKE_BUILD_TYPE:STRING=Release [source-directory]

Once your `cmake` command has completed, you can configure any build flags you want
using 'ccmake' or the cmake GUI, and then build with the following command on Linux:

	make -j8

Or alternatively by building it in Visual Studio or your compiler of choice on
Windows. The '-j8' tells the build to run multi-threaded using 8 threads, this is
useful for a faster build though if you get an error it can be difficult to know
here it was, in which case running just 'make' might be more helpful. For Windows,
currently VS2015 (with only some sub-versions of 2017) are supported. If using CUDA,
version 8.0 or 9.0, with CUDNN 6.0 is desired. Other versions have yet to be tested
extensively. On Windows it can also be beneficial to use Anaconda to get multiple
python packages. Boost Python (turned on by default when Python is enabled) requires
Numpy and a few other dependencies.

There are several optional arguments to viame which control which plugins get built,
such as those listed below. If a plugin is enabled that depends on another dependency
such as OpenCV) then the dependency flag will be forced to on.


<center>

| Flag                         | Description                                                                           |
|------------------------------|---------------------------------------------------------------------------------------|
| VIAME_ENABLE_OPENCV          | Builds OpenCV and basic OpenCV processes (video readers, simple GUIs)                 |
| VIAME_ENABLE_VXL             | Builds VXL and basic VXL processes (video readers, image filters)                     |
| VIAME_ENABLE_CAFFE           | Builds Caffe and basic Caffe processes (pixel classifiers, FRCNN dependency)          |
| VIAME_ENABLE_PYTHON          | Turns on support for using python processes                                           |
| VIAME_ENABLE_PYTORCH         | Installs all pytorch processes (detectors, classifiers)                               |
| VIAME_ENABLE_MATLAB          | Turns on support for and installs all matlab processes                                |
| VIAME_ENABLE_SCALLOP_TK      | Builds Scallop-TK based object detector plugin                                        |
| VIAME_ENABLE_YOLO            | Builds YOLO (Darknet) object detector plugin                                          |
| VIAME_ENABLE_FASTER_RCNN     | Builds Faster-RCNN based object detector plugin                                       |
| VIAME_ENABLE_BURNOUT         | Builds Burn-Out based pixel classifier plugin                                         |
| VIAME_ENABLE_UW_CLASSIFIER   | Builds UW fish classifier plugin                                                      |

</center>


And a number of flags which control which system utilities and optimizations are built, e.g.:


<center>

| Flag                         | Description                                                                                 |
|------------------------------|---------------------------------------------------------------------------------------------|
| VIAME_ENABLE_CUDA            | Enables CUDA (GPU) optimizations across all processes (OpenCV, Caffe, etc...)               |
| VIAME_ENABLE_CUDNN           | Enables CUDNN (GPU) optimizations across all processes                                      |
| VIAME_ENABLE_VIVIA           | Builds VIVIA GUIs (tools for making annotations and viewing detections)                     |
| VIAME_ENABLE_KWANT           | Builds KWANT detection and track evaluation (scoring) tools                                 |
| VIAME_ENABLE_DOCS            | Builds Doxygen class-level documentation for projects (puts in install share tree)          |
| VIAME_BUILD_DEPENDENCIES     | Build VIAME as a super-build, building all dependencies (default behavior)                  |
| VIAME_INSTALL_EXAMPLES       | Installs examples for the above modules into install/examples tree                          |
| VIAME_DOWNLOAD_MODELS        | Downloads pre-trained models for use with the examples and training new models              |

</center>

Update Instructions
-------------------

If you already have a checkout of VIAME and want to switch branches or
update your code, it is important to re-run:

	git submodule update --init --recursive

After switching branches to ensure that you have on the correct hashes
of sub-packages within the build (e.g. fletch or KWIVER). Very rarely
you may also need to run:

	git submodule sync

Just in case the address of submodules has changed. You only need to
run this command if you get a "cannot fetch hash #hashid" error.


Quick Run Instructions
----------------------

If building from the source, all final compiled binaries are placed in the
[build-directory]/install directory, which is the same as the root directory
in the pre-built binaries. This will hereby be refered to as the [install-directory].

One way to test the system is to see if you can run the examples in the [install-directory]/examples
folder, for example, the pipelined object detectors. There are some environment variables
that need to be set up before you can run on Linux or Mac, which are all in the
install/setup_viame.sh script. This script is sourced in all of the example run
scripts, and similar paths are added in the generated windows .bat example scripts.

Another good initial test is to run the [install-directory]/bin/plugin_explorer program. It
will generate a prodigious number of log messages and then list all the loadable
algorithms. The output should look as follows:

```
---- Algorithm search path

Factories that create type "image_object_detector"
---------------------------------------------------------------
Info on algorithm type "image_object_detector" implementation "darknet"
  Plugin name: darknet      Version: 1.0
    Description:        Image object detector using darknet
    Creates concrete type: kwiver::arrows::darknet::darknet_detector
    Plugin loaded from file: /user/viame/build/install/lib/modules/kwiver_algo_darknet_plugin.so
    Plugin module name: arrows.darknet

Factories that create type "track_features"
---------------------------------------------------------------
Info on algorithm type "track_features" implementation "core"
  Plugin name: core      Version: 1.0
    Description:        Track features from frame to frame using feature detection, matching, and
    loop closure.
    Creates concrete type: kwiver::arrows::core::track_features_core
    Plugin loaded from file: /user/viame/build/install/lib/modules/kwiver_algo_core_plugin.so
    Plugin module name: arrows.core

Factories that create type "video_input"
---------------------------------------------------------------
Info on algorithm type "video_input" implementation "vxl"
  Plugin name: vxl      Version: 1.0
    Description:        Use VXL (vidl with FFMPEG) to read video files as a sequence of images.
    Creates concrete type: kwiver::arrows::vxl::vidl_ffmpeg_video_input
    Plugin loaded from file: /user/viame/build/install/lib/modules/kwiver_algo_vxl_plugin.so
    Plugin module name: arrows.vxl

etc...
```

The plugin loaded line represents the shared objects that have been detected
and loaded. Each shared object can contain multiple algorithms. The algorithm
list shows each concrete algorithm that could be loaded and declared in pipeline files.
Check the log messages to see if there are any libraries that could not be located.

Each algorithm listed consists of two names. The first name is the type of
algorithm and the second is the actual implementation type. For example the
entry image_object_detector:hough_circle_detector indicates that it implements
the image_object_detector interface and it is a hough_circle_detector.

Algorithms can be instantiated in any program and use a configuration based
approach to select which concrete implementation to instantiate.

For a simple pipeline test, go to -

	cd [install-directory]/examples/hello_world_pipeline/

or

	cd [install-directory]/examples/detector_pipelines/

In those directories, run one of the detector pipelines. Which ENABLE_FLAGS you
enabled will control which detector pipelines you can run, and only run scripts
with all required dependencies enabled will show up in the install tree. Each
script is just performing a call to pipeline runner under the hood, e.g.:

	pipeline_runner -p [pipeline-file].pipe

Output detections can then be viewed in the GUI, e.g., see:

[install-directory]/examples/annotation_and_visualization/


License and Citation
--------------------

VIAME is released under a BSD-3 license.

A system paper summarizing VIAME was published in IEEE WACV 2017 which is available
[here](https://data.kitware.com/api/v1/item/597817fa8d777f16d01e9e7f/download).
