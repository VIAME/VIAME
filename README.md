
<img src="http://www.viametoolkit.org/wp-content/uploads/2016/08/viami_logo.png" alt="VIAME Logo" width="200" height="78">
<br>
VIAME is a computer vision application designed for do-it-yourself artificial intelligence including
object detection, object tracking, image/video annotation, image/video search, image mosaicing,
stereo measurement, rapid model generation, and tools for the evaluation of different algorithms.
Originally targetting marine species analytics, it now contains many common algorithms and libraries,
and is also useful as a generic computer vision toolkit. The core infrastructure connecting different
system components is currently the KWIVER library, which can connect C/C++, python, and matlab nodes
together in a graph-like pipeline architecture. Alongside the pipelined image processing system are a
number of standalone tools for accomplishing the above. Both a desktop and web version exists for
deployments in different types of environments.

Documentation
-------------

The [User's Quick-Start Guide](https://data.kitware.com/api/v1/item/5e433c67af2e2eed358fc82b/download),
[Tutorial Videos](https://www.youtube.com/channel/UCpfxPoR5cNyQFLmqlrxyKJw), 
and [Developer's Manual](http://viame.readthedocs.io/en/latest/) are more comprehensive,
but select entries are also listed below broken down by individual functionality:


[Build and Install Guide](examples/building_and_installing_viame) <> 
[All Examples](https://github.com/Kitware/VIAME/tree/master/examples) <> 
[GUIs for Annotation and Visualization](examples/annotation_and_visualization) <>
[Object Detectors](examples/object_detection) <>
[Object Trackers](examples/object_tracking) <>
[Detector Training API](examples/object_detector_training) <>
[Video Search and Rapid Model Generation](examples/search_and_rapid_model_generation) <>
[Scoring of Detectors](examples/scoring_and_roc_generation) <>
[Detection File Formats](examples/detection_file_conversions) <>
[Calibration and Image Enhancement](examples/image_enhancement) <>
[Image Registration and Mosaicing](examples/image_registration)  <>
[Stereo Measurement and Depth Maps](examples/measurement_using_stereo) <>
[KWIVER Overview](https://github.com/Kitware/kwiver) <>
[Core Class and Pipelining Info](http://kwiver.readthedocs.io/en/latest/architecture.html) <>
[Web Interface](https://github.com/VIAME/VIAME-Web) <>
[How to Integrate Your Own Plugin](examples/hello_world_pipeline) <>
[Example Plugin Templates](plugins/templates) <>
[Embedding Detectors in C++ Code](examples/using_detectors_in_cxx_code)

Installations
-------------

For a full installation guide and description of the various flavors of VIAME, see the
quick-start guide, above. In summary, VIAME Desktop is currently provided as either a .zip or
.tar file. Alternatively, docker files are available for both VIAME Desktop and Web (below).
A sample instance of VIAME Web is also online, hosted at [viame.kitware.com](viame.kitware.com).
For desktop installs, extract the binaries and place them in a directory of your choosing,
for example /opt/noaa/viame on Linux or C:\Program Files\VIAME on Windows. If you're using
packages built with GPU support, make sure to have sufficient video drivers installed,
version 418.39 or higher. The best way to install drivers depends on your operating system,
see below. Lastly, run through some of the examples to validate the installation. The binaries
are quite large, in terms of disk space, due to the inclusion of multiple default model files
and programs, but if just building your desired features from source (e.g. for embedded apps)
they are much smaller.

**Installation Requirements:** <br>
RHEL/CentOS 7 64-Bit, Ubuntu 16.04/18.04 64-Bit, Windows 7, 8, or 10 64-Bit <br>
6 Gb of Disk Space for the Full Installation <br>

**Installation Recommendations:** <br>
NVIDIA Drivers (Version 418.39+ 
Windows 
[\[1\]](https://www.nvidia.com/Download/index.aspx?lang=en-us)
[\[2\]](https://developer.nvidia.com/cuda-downloads)
Ubuntu 
[\[1\]](https://linuxhint.com/ubuntu_nvidia_ppa/)
[\[2\]](https://developer.nvidia.com/cuda-downloads)
CentOS 
[\[1\]](https://developer.nvidia.com/cuda-downloads)
[\[2\]](https://www.nvidia.com/Download/index.aspx?lang=en-us)) <br>
A [CUDA-enabled GPU](https://developer.nvidia.com/cuda-gpus) with 8 Gb or more VRAM <br>

**Windows Desktop Binaries:** <br>
[VIAME v0.12.1 Windows 7\*/8/10, GPU Enabled, Mirror1 (.zip)](https://data.kitware.com/api/v1/item/5fb9ff2650a41e3d196659bc/download) <br>
[VIAME v0.12.1 Windows 7\*/8/10, GPU Enabled, Mirror2 (.zip)](https://drive.google.com/file/d/1bQLBtInQWzCj06O_P-5nThcYHEFROaRg/view?usp=sharing) <br>
[VIAME v0.12.1 Windows 7\*/8/10, CPU Only, Mirror1 (.zip)](https://data.kitware.com/api/v1/item/5fb9ff7350a41e3d196659df/download) <br>
[VIAME v0.12.1 Windows 7\*/8/10, CPU Only, Mirror2 (.zip)](https://drive.google.com/file/d/1fb3zcBZbT3tM_wrtgRmnpJKIXnr0UtTR/view?usp=sharing)

**Ubuntu Desktop Binaries:** <br>
[VIAME v0.12.0 Ubuntu 18.04, GPU Enabled, Mirror1 (.tar.gz)](https://data.kitware.com/api/v1/item/5f9f6cc050a41e3d19274b8e/download) <br>
[VIAME v0.12.0 Ubuntu 18.04, GPU Enabled, Mirror2 (.tar.gz)](https://drive.google.com/file/d/1v6E8eT15OlbcFyYKgyGYvYRsaZeyMbqw/view?usp=sharing) <br>
[VIAME v0.12.0 Ubuntu 16.04, GPU Enabled, Mirror1 (.tar.gz)](https://data.kitware.com/api/v1/item/5f9f751350a41e3d19276a29/download) <br>
[VIAME v0.12.0 Ubuntu 16.04, GPU Enabled, Mirror2 (.tar.gz)](https://drive.google.com/file/d/1e9bWPJvwHKCJxsI8DoiVciTiEDlpBSAR/view?usp=sharing)

**CentOS or Other Linux Desktop Binaries:** <br>
[VIAME v0.12.0 RHEL/CentOS 7/8, GPU Enabled, Mirror1 (.tar.gz)](https://data.kitware.com/api/v1/item/5f9f75d350a41e3d19276bff/download) <br>
[VIAME v0.12.0 RHEL/CentOS 7/8, GPU Enabled, Mirror2 (.tar.gz)](https://drive.google.com/file/d/1aQACQAkmu-5QyDNnQ3KVyOa5q29JF43M/view?usp=sharing) <br>
[VIAME v0.12.0 Generic Linux, GPU Enabled, Mirror1 (.tar.gz)](https://data.kitware.com/api/v1/item/5f9f75d350a41e3d19276bff/download) <br>
[VIAME v0.12.0 Generic Linux, GPU Enabled, Mirror2 (.tar.gz)](https://drive.google.com/file/d/1aQACQAkmu-5QyDNnQ3KVyOa5q29JF43M/view?usp=sharing)

\*Windows 7 requires some updates and service packs installed, e.g. [KB2533623](https://www.microsoft.com/en-us/download/details.aspx?id=26764).

**Web Applications**: <br>
[VIAME Online Web Annotator and Public Annotation Archive](https://viame.kitware.com/) <br>
[VIAME Web Local Installation Instructions](https://github.com/VIAME/VIAME-Web/blob/master/docker/README.md) <br>
[VIAME Web Source Repository](https://github.com/VIAME/VIAME-Web)

**Custom Desktop Distributions:** <br>
[Seal Multi-View GUI, Windows 7/8/10, GPU Enabled (.zip)](https://data.kitware.com/api/v1/item/5fbea91550a41e3d19705409/download) <br>
[Seal Multi-View GUI, Windows 7/8/10, CPU Only (.zip)](https://data.kitware.com/api/v1/item/5f87103650a41e3d19ea6c66/download) <br>
[Seal Multi-View GUI, CentOS 7, GPU Enabled (.tar.gz)](https://data.kitware.com/api/v1/item/5f7764cd50a41e3d19c2e35f/download) <br>
[Seal Multi-View GUI, Generic Linux, GPU Enabled (.tar.gz)](https://data.kitware.com/api/v1/item/5f77645650a41e3d19c2e26b/download)

**Optional Patches:** <br>
[Alternative Generic Detector for IQR Add-On, All OS](https://data.kitware.com/api/v1/item/5ceda2d28d777f072bef0c0d/download) <br>
[Arctic Seals Models Add-On, Windows](https://data.kitware.com/api/v1/item/5e30b8ffaf2e2eed3545bff6/download) <br>
[Arctic Seals Models Add-On, Linux](https://data.kitware.com/api/v1/item/5e30b283af2e2eed3545a888/download) <br>
[HabCam Models (Scallop, Skate, Flatfish) Add-On, Windows](https://data.kitware.com/api/v1/item/5f6bb7e850a41e3d19a63047/download) <br>
[HabCam Models (Scallop, Skate, Flatfish) Add-On, Linux](https://data.kitware.com/api/v1/item/5f9e115f50a41e3d19253c84/download) <br>
[Low Memory GPU (For 4+ Gb Cards) Add-On, All OS](https://data.kitware.com/api/v1/item/5d7edbadd35580e6dc170c78/download) <br>
[MOUSS Model Set 1 (Deep 7 Bottomfish) Add-On, All OS](https://data.kitware.com/api/v1/item/5cdec8ac8d777f072bb4457f/download) <br>
[MOUSS Model Set 2 (Deep 7 Bottomfish) Add-On, All OS](https://data.kitware.com/api/v1/item/5ce5af728d777f072bd5836d/download) <br>
[MOUSS Sample Project, All Linux](https://data.kitware.com/api/v1/item/5e8d3ad42660cbefba9dd13c/download) <br>
[Sea Lion Models Add-On, All OS](https://data.kitware.com/api/v1/item/5f750f1c50a41e3d19bc97bb/download)

Note: To install Add-Ons and Patches, copy them into an existing VIAME installation folder.
To use project files extract them into your working directory of choice. Custom Applications
contain a full installation, only with non-default features turned on, and should not be copied
into existing installations because they are a full installation and bad things will happen.

Docker Images
-------------

Docker images are available on: https://hub.docker.com. For a default container with just core
algorithms, runnable via command-line, see:

kitware/viame:gpu-algorithms-latest

This image is headless (ie, it contains no GUI) and contains a VIAME desktop (not web)
installation in the folder /opt/noaa/viame. For links to the VIAME-Web docker containers see the
above section in the installation documentation. Most add-on models are not included in the
instance but can be downloaded via running the script download_viame_addons.sh in the bin folder.

Quick Build Instructions
------------------------

These instructions are intended for developers or those interested in building the latest master
branch. More in-depth build instructions can be found [here](examples/building_and_installing_viame),
but the software can be built either as a super-build, which builds most of its dependencies
alongside itself, or standalone. To build VIAME requires, at a minimum, [Git](https://git-scm.com/),
[CMake](https://cmake.org/), and a [C++ compiler](http://www.cplusplus.com/doc/tutorial/introduction/).
Installing Python and CUDA is also recommended. If using CUDA, version 9.2 or 10.1 are preferred,
with CUDNN 7.0 and above. Other CUDA versions may or may not work. On both Windows and Linux it
is also recommended to use [Anaconda3 5.2.0](https://repo.anaconda.com/archive/) for python,
which is the most tested distribution used by developers. If using other python distributions
having numpy installed, at a minimum, is necessary.

To build on the command line in Linux, use the following commands, only replacing [source-directory]
and [build-directory] with locations of your choice. While these directories can be the same,
it's good practice to have a 'src' checkout then a seperate 'build' directory alongside it:

	git clone https://github.com/VIAME/VIAME.git [source-directory]

	cd [source-directory] && git submodule update --init --recursive

Next, create a build directory and run the following `cmake` command (or alternatively
use the cmake GUI if you are not using the command line interface):

	mkdir [build-directory] && cd [build-directory]

	cmake -DCMAKE_BUILD_TYPE:STRING=Release [source-directory]

Once your `cmake` command has completed, you can configure any build flags you want
using 'ccmake' or the cmake GUI, and then build with the following command on Linux:

	make -j8

Or alternatively by building it in Visual Studio or your compiler of choice on
Windows. On Linux, '-j8' tells the build to run multi-threaded using 8 threads, this
is useful for a faster build though if you get an error it can be difficult to see
it, in which case running just 'make' might be more helpful. For Windows,
currently VS2017 is the desired compiler, though select versions of 2015 and 2019
also work.

There are several optional arguments to viame which control which plugins get built,
such as those listed below. If a plugin is enabled that depends on another dependency
such as OpenCV) then the dependency flag will be forced to on. If uncertain what to turn
on, it's best to just leave the default enable and disable flags which will build most
(though not all) functionalities. These are core components we recommend leaving turned on:


<center>

| Flag                         | Description                                                                        |
|------------------------------|------------------------------------------------------------------------------------|
| VIAME_ENABLE_OPENCV          | Builds OpenCV and basic OpenCV processes (video readers, simple GUIs)              |
| VIAME_ENABLE_VXL             | Builds VXL and basic VXL processes (video readers, image filters)                  |
| VIAME_ENABLE_PYTHON          | Turns on support for using python processes (multiple algorithms)                  |
| VIAME_ENABLE_PYTORCH         | Installs all pytorch processes (detectors, trackers, classifiers)                  |

</center>


And a number of flags which control which system utilities and optimizations are built, e.g.:


<center>

| Flag                         | Description                                                                        |
|------------------------------|------------------------------------------------------------------------------------|
| VIAME_ENABLE_CUDA            | Enables CUDA (GPU) optimizations across all processes (PyTorch, etc...)            |
| VIAME_ENABLE_CUDNN           | Enables CUDNN (GPU) optimizations across all processes                             |
| VIAME_ENABLE_VIVIA           | Builds VIVIA GUIs (tools for making annotations and viewing detections)            |
| VIAME_ENABLE_KWANT           | Builds KWANT detection and track evaluation (scoring) tools                        |
| VIAME_ENABLE_DOCS            | Builds Doxygen class-level documentation for projects (puts in install tree)       |
| VIAME_BUILD_DEPENDENCIES     | Build VIAME as a super-build, building all dependencies (default behavior)         |
| VIAME_INSTALL_EXAMPLES       | Installs examples for the above modules into install/examples tree                 |
| VIAME_DOWNLOAD_MODELS        | Downloads pre-trained models for use with the examples and interfaces              |

</center>


And lastly, a number of flags which build algorithms or interfaces with more specialized functionality:


<center>

| Flag                         | Description                                                                        |
|------------------------------|------------------------------------------------------------------------------------|
| VIAME_ENABLE_SMQTK           | Builds SMQTK plugins for image/video search                                        |
| VIAME_ENABLE_SCALLOP_TK      | Builds Scallop-TK based object detector plugin                                     |
| VIAME_ENABLE_YOLO            | Builds YOLO (Darknet) object detector plugin                                       |
| VIAME_ENABLE_BURNOUT         | Builds Burn-Out based pixel classifier plugin                                      |
| VIAME_ENABLE_ITK             | Builds ITK cross-modality image registration                                       |
| VIAME_ENABLE_UW_CLASSIFIER   | Builds UW fish classifier plugin                                                   |
| VIAME_ENABLE_TENSORFLOW      | Builds TensorFlow object detector plugin                                           |
| VIAME_ENABLE_SEAL_TK         | Builds Seal multi-modality GUI                                                     |
| VIAME_ENABLE_MATLAB          | Turns on support for and installs all matlab processes                             |
| VIAME_ENABLE_LANL            | Builds an additional (Matlab) scallop detector                                     |

</center>


Source Code Layout
------------------
<pre>
 VIAME
   ├── cmake               # CMake configuration files for subpackages
   ├── docs                # Documentation files and manual (pre-compilation)
   ├── configs             # All system-runnable config files and models
   │   ├── pipelines       # All processing pipeline configs
   │   │   └── models      # All models, which only get downloaded based on flags
   │   ├── prj-linux       # Default linux project files
   │   └── prj-windows     # Default windows project files 
   ├── examples            # All runnable examples and example tutorials
   ├── packages            # External projects used by the system
   │   ├── kwiver          # Processing backend infastructure
   │   ├── fletch          # Dependency builder for things which don't change often
   │   ├── kwant           # Scoring and detector evaluation tools
   │   ├── vivia           # Baseline desktop GUIs (v1.0)
   │   └── ...             # Assorted other packages (typically for algorithms)
   ├── plugins             # Integrated algorithms or wrappers around external projects
   │   └── ...             # Assorted plugins (detectors, depth maps, filters, etc.)
   ├── tools               # Standalone tools or scripts, often building on the above
   └── README.md           # Project introduction page that you are reading
   └── RELEASE_NOTES.md    # A list of the latest updates in the system per version
</pre>


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


License, Citations, and Acknowledgements
----------------------------------------

VIAME is released under a BSD-3 license.

A non-exhaustive list of relevant papers used within the project alongside contributors
can be found [here](docs/citations.md).

VIAME was developed with funding from multiple sources, with special thanks
to those listed [here](docs/acknowledgements.md).
