
<img src="http://www.viametoolkit.org/wp-content/uploads/2016/08/viami_logo.png" alt="VIAME Logo" width="200" height="78">
<br>
VIAME is a computer vision library designed to integrate several image and video processing algorithms together in
a common distributed processing framework, majorly targeting marine species analytics. As it contains many common
algorithms and compiles several other popular repositories together as a part of its build process, VIAME is also
useful as a general computer vision toolkit. The core infrastructure connecting different system components is
currently the KWIVER library, which can connect C/C++, python, and matlab nodes together in a graph-like pipeline
architecture. Alongside the pipelined image processing system are a number of standalone utilties for model training,
output detection visualization, groundtruth annotation, detector/tracker evaluation (a.k.a. scoring), image/video search,
and rapid model generation.

Example Capabilities
--------------------
<p align="left">
<br>
<nobr>
<img src="http://www.viametoolkit.org/wp-content/uploads/2018/06/viame-gh-splash-04.png" alt="Search Example" width="297" height="180">
<img src="http://www.viametoolkit.org/wp-content/uploads/2018/06/tracks.png" alt="Tracking Example" width="232" height="180">
<img src="http://www.viametoolkit.org/wp-content/uploads/2018/06/viame-gh-splash-02.png" alt="Detection Example" width="297" height="180">
</nobr>
</p>
<p align="left">
<nobr>
<img src="http://www.viametoolkit.org/wp-content/uploads/2018/06/viame-gh-splash-03.png" alt="Measurement Example" width="408" height="146">
<img src="http://www.viametoolkit.org/wp-content/uploads/2018/06/viame-gh-splash-01.png" alt="Query Example" width="421" height="146">
</nobr>
</p>

Documentation
-------------

The [VIAME manual](http://viame.readthedocs.io/en/latest/) is more comprehensive,
but select entries are also listed below:


[Build and Install Guide](examples/building_and_installing_viame) <> 
[All Examples](https://github.com/Kitware/VIAME/tree/master/examples) <> 
[Core Class and Pipeline Info](http://kwiver.readthedocs.io/en/latest/architecture.html) <> 
[Object Detector Examples](examples/object_detection) <br />
[GUIs for Visualization and Annotation](examples/annotation_and_visualization) <> 
[Detector Training API](examples/object_detector_training) <>
[Example Plugin Templates](plugins/templates) <br />
[Video Search and Rapid Model Generation](examples/search_and_rapid_model_generation) <> 
[Scoring and Evaluation of Detectors](examples/scoring_and_roc_generation) <>
[KWIVER Overview](https://github.com/Kitware/kwiver) <br />
[Stereo Measurement Examples](examples/measurement_using_stereo) <> 
[Embedding Detectors in C++ Code](examples/using_detectors_in_cxx_code) <>
[How to Integrate Your Own Plugin](examples/hello_world_pipeline)

Pre-Built Binaries
------------------

For a full installation guide, [see here](https://data.kitware.com/api/v1/item/5c9aad768d777f072bdcab59/download).
In summary, extract the binaries and place them in a directory of your choosing, for
example C:\Program Files\VIAME on Windows or /opt/noaa/viame on Linux.
Next, set the PYTHON_INSTALL_DIR at the top of the setup_viame.sh (Linux) or
setup_viame.bat (Windows) script in the root install folder to point to the
location of your installed Anaconda distribution. Lastly, run through some of the examples
to validate the installation.

**Installation Requirements:** <br>
RHEL/CentOS 7 64-Bit, Ubuntu 16.04 64-Bit, Windows 7, 8, or 10 64-Bit <br>
[Anaconda3 5.2.0 x86_64](https://repo.continuum.io/archive/) (Note: Anaconda**3 5.2.0 x86_64**, not Anaconda2 or x86 or 5.3.0) <br>
4 Gb of Disk Space for the Full Installation <br>

**Installation Recommendations:** <br>
[NVIDIA Drivers](https://www.nvidia.com/Download/index.aspx?lang=en-us) (375.26+ for CUDA 8, 384.81+ for CUDA 9) <br>
A CUDA-enabled GPU with 8 Gb or more VRAM <br>

**Linux Desktop Binaries:** <br>
[VIAME v0.9.8.8 Ubuntu 16.04, 64-Bit, GPU Enabled, CUDA 8.0, Python 3.6, Mirror1](https://data.kitware.com/api/v1/item/5bd79fd68d777f06b93e7ad5/download) <br>
[VIAME v0.9.8.8 Ubuntu 16.04, 64-Bit, GPU Enabled, CUDA 8.0, Python 3.6, Mirror2](https://drive.google.com/open?id=1Yjs5oSlRkS-8ypIjBuna42qFSxqSGgb7) <br>
[VIAME v0.9.9.3 RHEL/CentOS 7, 64-Bit, GPU Enabled, CUDA 8.0, Python 3.6, Mirror1](https://data.kitware.com/api/v1/item/5c6b0bf78d777f072b51ead3/download) <br>
[VIAME v0.9.9.3 RHEL/CentOS 7, 64-Bit, GPU Enabled, CUDA 8.0, Python 3.6, Mirror2](https://drive.google.com/open?id=1ZqERSOs10awJVMklln8Skx7RCsgqS2Wg)

**Windows Desktop Binaries:** <br>
[VIAME v0.9.9.7 Windows 7/8/10, 64-Bit, GPU Enabled, CUDA 8.0, Python 3.6, Mirror1](https://data.kitware.com/api/v1/item/5c916d278d777f072bc17c01/download) <br>
[VIAME v0.9.9.7 Windows 7/8/10, 64-Bit, GPU Enabled, CUDA 8.0, Python 3.6, Mirror2](https://drive.google.com/open?id=1qsA0XulI2sSeqlftdFG0WJrkn-HxurZ7) <br>
[VIAME v0.9.9.7 Windows 7/8/10, 64-Bit, CPU Only, Python 3.6, Mirror1](https://data.kitware.com/api/v1/item/5c98c5728d777f072bd52c43/download) <br>
[VIAME v0.9.9.7 Windows 7/8/10, 64-Bit, CPU Only, Python 3.6, Mirror2](https://drive.google.com/open?id=167QP2Iyyb862d_d9ebSDzgwLsY3MHl0M)

**Web Annotator**: <br>
[Experimental Detection Annotator](https://images.slide-atlas.org/#collection/5b68666270aaa94f2e5bd975/folder/5b68667670aaa94f2e5bd976) <br>

**Optional Patches:** <br>
[MOUSS Models Add-On, All OS](https://data.kitware.com/api/v1/item/5c58d8f48d777f072b2b980d/download) <br>
[MOUSS Sample Project, All Linux](https://data.kitware.com/api/v1/item/5c58d8f68d777f072b2b9815/download) <br>
[Arctic Seals Models Add-On, All Linux, CUDA 9](https://data.kitware.com/api/v1/item/5ca3e4df8d777f072bf2aa65/download) <br>
[Seal Dual Display GUI, Windows 7/8/10, 64-Bit](https://data.kitware.com/api/v1/item/5cd4968a8d777f072b98c637/download) <br>
[HabCam Models Add-On, All OS](https://data.kitware.com/api/v1/item/5c58d8ea8d777f072b2b97fa/download) <br>
[Low Memory GPU (2 Gb to 4 Gb) Add-On, All OS](https://data.kitware.com/api/v1/item/5c74a0328d777f072b6615f3/download)

Note: To install Add-Ons, copy them into your install. To use project files extract them into your working directory of choice.


Quick Run Instructions
----------------------

If building from the source, all final compiled binaries are placed in the
[build-directory]/install directory, which is the same as the root directory
in the above pre-built binaries. This will hereby be refered to as the [install-directory].

One way to test the system is to see if you can run the examples in the
[[install-directory]/examples](https://github.com/Kitware/VIAME/tree/master/examples)
folder, for example, the pipelined object detectors or annotation GUI. If pursuing this route, we
recommend reading the: 
[Examples Overview](https://viame.readthedocs.io/en/latest/section_links/example_capabilities.html), 
[Annotation Overview](https://viame.readthedocs.io/en/latest/section_links/annotation_and_visualization.html), 
[Deep Model Generation Overview](https://viame.readthedocs.io/en/latest/section_links/object_detector_training.html), and
[Rapid Model Generation Overview](https://viame.readthedocs.io/en/latest/section_links/search_and_rapid_model_generation.html#video-and-image-search-using-viame).

The 'examples' folder is one of two core entry points into running VIAME functionality. The other is
to copy project files for your operating system,
[[install-directory]/configs/prj-linux](https://github.com/Kitware/VIAME/tree/master/configs/prj-linux) or
[[install-directory]/configs/prj-windows](https://github.com/Kitware/VIAME/tree/master/configs/prj-windows)
to a directory of your choice and run things from there. Not all functionality is in the default project
file scripts, however, but it is a good entry point if you just want to get started on training object
detection and/or tracking models. There are some environment variables in these files that need to
be set up before you can run on any OS, which are all in the [install-directory]/setup_viame.sh/.bat script.
This script is sourced in all of the project scripts, so there is no need to modify anything unless you
installed VIAME to a non-default location. For the later case you will need to need to modify the
VIAME_INSTALL path at the top of each run script to point to your installed location. 


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
version 8.0 and above, with CUDNN 6.0 and above is desired. On Windows it can also be
beneficial to use Anaconda to get multiple python packages. Boost Python (turned on by
default when Python is enabled) requires Numpy and a few other dependencies.

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
of sub-packages within the build. Very rarely you may also need to run:

	git submodule sync

Just in case the address of submodules has changed. You only need to
run this command if you get a "cannot fetch hash #hashid" error.


License and Citation
--------------------

VIAME is released under a BSD-3 license.

A non-exhaustive list of relevant papers used within VIAME can be found [here](doc/citations.md).
