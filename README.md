
<img src="http://www.viametoolkit.org/wp-content/uploads/2016/08/viami_logo.png" alt="VIAME Logo" width="200" height="78">
<br>
VIAME is a computer vision library designed to integrate several image and video processing algorithms together in
a common distributed processing framework, majorly targeting marine species analytics. As it contains many common
algorithms and compiles several other popular repositories together as a part of its build process, VIAME is also
useful as a general computer vision toolkit. The core infrastructure connecting different system components is
currently the KWIVER library, which can connect C/C++, python, and matlab nodes together in a graph-like pipeline
architecture. Alongside the pipelined image processing system are a number of standalone utilties for model training,
output visualization, groundtruth annotation, detector/tracker evaluation (a.k.a. scoring), image/video search,
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

The [User's Quick-Start Guide](https://data.kitware.com/api/v1/item/5dc08673e3566bda4b57a493/download),
[Tutorial Videos](https://www.youtube.com/channel/UCpfxPoR5cNyQFLmqlrxyKJw), 
and [Developer's Manual](http://viame.readthedocs.io/en/latest/) are more comprehensive,
but select entries are also listed below broken down by individual functionality:


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

For a full installation guide see the quick-start slide deck above, but in summary, extract the binaries
and place them in a directory of your choosing, for example C:\Program Files\VIAME on Windows or /opt/noaa/viame
on Linux. Next, run through some of the examples to validate the installation. It is no longer necessary
to install any dependencies of VIAME, they are all packaged inside of it. The binaries are quite large
(in terms of disk space) due to the inclusion of model default model files and programs, but if just
building your desired features from source they can be relatively small.

**Installation Requirements:** <br>
RHEL/CentOS 7 64-Bit, Ubuntu 16.04 64-Bit, Windows 7, 8, or 10 64-Bit <br>
[Anaconda3 5.2.0 x86_64](https://repo.continuum.io/archive/) (Note: Anaconda**3 5.2.0 x86_64**, not Anaconda2 or x86 or 5.3.0) <br>
5 Gb of Disk Space for the Full Installation <br>

**Installation Recommendations:** <br>
NVIDIA Drivers (Version 410.48+ 
Windows 
[\[1\]](https://www.nvidia.com/Download/index.aspx?lang=en-us)
[\[2\]](https://developer.nvidia.com/cuda-downloads)
Ubuntu 
[\[1\]](https://linuxhint.com/ubuntu_nvidia_ppa/)
[\[2\]](https://developer.nvidia.com/cuda-downloads)
CentOS 
[\[1\]](https://developer.nvidia.com/cuda-downloads)
[\[2\]](https://www.nvidia.com/Download/index.aspx?lang=en-us)) <br>
A CUDA-enabled GPU with 8 Gb or more VRAM <br>

**Linux Desktop Binaries:** <br>
[VIAME v0.10.1 Ubuntu 16.04/18.04, 64-Bit, GPU Enabled, Python 3.6, Mirror1](https://data.kitware.com/api/v1/item/5dd2b306af2e2eed35043d67/download) <br>
[VIAME v0.10.1 Ubuntu 16.04/18.04, 64-Bit, GPU Enabled, Python 3.6, Mirror2](https://drive.google.com/open?id=1PmPrdcXmZoJM0vHMslTp_TU1MkBn1m7d) <br>
[VIAME v0.10.1 RHEL/CentOS 7, 64-Bit, GPU Enabled, Python 3.6, Mirror1](https://data.kitware.com/api/v1/item/5dd2b22aaf2e2eed35043af7/download) <br>
[VIAME v0.10.1 RHEL/CentOS 7, 64-Bit, GPU Enabled, Python 3.6, Mirror2](https://drive.google.com/open?id=1JCNr4JppWoe0cH0yHWviT3gl3c93rb6e)

**Windows Desktop Binaries:** <br>
[VIAME v0.10.0 Windows 7\*/8/10, 64-Bit, GPU Enabled, Python 3.6, Mirror1](https://data.kitware.com/api/v1/item/5dcec95caf2e2eed35fa2177/download) <br>
[VIAME v0.10.0 Windows 7\*/8/10, 64-Bit, GPU Enabled, Python 3.6, Mirror2](https://drive.google.com/open?id=1YxQT9EBP8AzAlQYNM9WBykPlvJxBfXW_) <br>
[VIAME v0.10.1 Windows 7\*/8/10, 64-Bit, CPU Only, Python 3.6, Mirror1](https://data.kitware.com/api/v1/item/5dd2b277af2e2eed35043be0/download) <br>
[VIAME v0.10.1 Windows 7\*/8/10, 64-Bit, CPU Only, Python 3.6, Mirror2](https://drive.google.com/open?id=1R4KLFXnUXxsfi_BS6iWzSoOrac0D4b41)

\*Windows 7 requires some updates and service packs installed, e.g. [KB2533623](https://www.microsoft.com/en-us/download/details.aspx?id=26764).

**Web Applications**: <br>
[VIAME Web Repository (Under Development)](https://github.com/VIAME/VIAME-Web) <br>
[VIAME Deprecated Web Annotator](https://images.slide-atlas.org/#collection/5b68666270aaa94f2e5bd975/folder/5b68667670aaa94f2e5bd976)

**Optional Patches:** <br>
[Alternative Generic Detector for IQR Add-On, All OS](https://data.kitware.com/api/v1/item/5ceda2d28d777f072bef0c0d/download) <br>
[Arctic Seals Models Add-On, Linux](https://data.kitware.com/api/v1/item/5d7bb82fd35580e6dc0fb5af/download) <br>
[Arctic Seals Models Add-On, Windows](https://data.kitware.com/api/v1/item/5d7a6c19d35580e6dc0abb0b/download) <br>
[HabCam Models (Scallop, Skate, Flatfish) Add-On, All OS](https://data.kitware.com/api/v1/item/5d016fbf8d777f072b2955c9/download) <br>
[Low Memory GPU (For 4+ Gb Cards) Add-On, All OS](https://data.kitware.com/api/v1/item/5d7edbadd35580e6dc170c78/download) <br>
[MOUSS Model Set 1 (Deep 7 Bottomfish) Add-On, All OS](https://data.kitware.com/api/v1/item/5cdec8ac8d777f072bb4457f/download) <br>
[MOUSS Model Set 2 (Deep 7 Bottomfish) Add-On, All OS](https://data.kitware.com/api/v1/item/5ce5af728d777f072bd5836d/download) <br>
[MOUSS Sample Project, All Linux](https://data.kitware.com/api/v1/item/5d5d84d185f25b11ff427b4e/download)

**Custom Applications:** <br>
[Seal Multi-View GUI, CentOS 7, GPU Enabled](https://data.kitware.com/api/v1/item/5d8c8347d35580e6dc44e7d4/download) <br>
[Seal Multi-View GUI, Windows 7/8/10, GPU Enabled](https://data.kitware.com/api/v1/item/5d8d37bed35580e6dc46a3f8/download) <br>
[Seal Multi-View GUI, Windows 7/8/10, CPU Only](https://data.kitware.com/api/v1/item/5d8d3867d35580e6dc46aad8/download)

Note: To install Add-Ons and Patches, copy them into an existing VIAME installation folder.
To use project files extract them into your working directory of choice. Custom Applications
contain a full installation, only with non-default features turned on, and should not be copied
into existing installations because they are a full installation and bad things will happen.

Quick Build Instructions
------------------------

These instructions are intended for developers or those interested in building the latest
master branch. More in-depth build instructions can be found [here](examples/building_and_installing_viame),
but the software can be built either as a super-build, which builds most of its
dependencies alongside itself, or standalone. To build VIAME requires, at a minimum,
[Git](https://git-scm.com/), [CMake](https://cmake.org/), and a C++ compiler.
If using the command line, run the following commands, only replacing [source-directory] and
[build-directory] with locations of your choice. While these directories can be the same, it's good
practice to have a 'src' checkout then a seperate 'build' directory:

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
Windows. '-j8' tells the build to run multi-threaded using 8 threads, this is
useful for a faster build though if you get an error it can be difficult to see
it, in which case running just 'make' might be more helpful. For Windows,
currently VS2017 is the desired compiler, though select versions of 2015 and 2019
also work. If using CUDA, version 9.0 and above, with CUDNN 7.0 and above is desired,
in particular we recommend CUDA 10. On both Windows and Linux it can also be
beneficial to use Anaconda to get multiple standard python packages. Boost Python
(turned on by default when Python is enabled) requires Numpy and a few other dependencies.

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
| VIAME_ENABLE_DOCS            | Builds Doxygen class-level documentation for projects (puts in install share tree) |
| VIAME_BUILD_DEPENDENCIES     | Build VIAME as a super-build, building all dependencies (default behavior)         |
| VIAME_INSTALL_EXAMPLES       | Installs examples for the above modules into install/examples tree                 |
| VIAME_DOWNLOAD_MODELS        | Downloads pre-trained models for use with the examples and training new models     |

</center>


And lastly, a number of flags which build algorithms with more specialized functionality:


<center>

| Flag                         | Description                                                                        |
|------------------------------|------------------------------------------------------------------------------------|
| VIAME_ENABLE_MATLAB          | Turns on support for and installs all matlab processes                             |
| VIAME_ENABLE_SCALLOP_TK      | Builds Scallop-TK based object detector plugin                                     |
| VIAME_ENABLE_YOLO            | Builds YOLO (Darknet) object detector plugin                                       |
| VIAME_ENABLE_BURNOUT         | Builds Burn-Out based pixel classifier plugin                                      |
| VIAME_ENABLE_ITK             | Builds ITK cross-modality image registration                                       |
| VIAME_ENABLE_UW_CLASSIFIER   | Builds UW fish classifier plugin                                                   |
| VIAME_ENABLE_TENSORFLOW      | Builds TensorFlow object detector plugin                                           |
| VIAME_ENABLE_SEAL_TK         | Builds Seal Multi-Modality GUI                                                     |
| VIAME_ENABLE_LANL            | Builds an additional (Matlab) scallop detector                                     |

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


Source Code Layout
------------------
<pre>
 VIAME
   ├── cmake               # CMake configuration files for subpackages
   ├── doc                 # Documentation files and manual (pre-compilation)
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
   └── README.md           # Project introduction page that you are reading
</pre>


License, Citations, and Acknowledgements
----------------------------------------

VIAME is released under a BSD-3 license.

A non-exhaustive list of relevant papers used within the project alongside contributors
can be found [here](doc/citations.md).

VIAME was developed with funding from multiple sources, with special thanks
to those listed [here](doc/acknowledgements.md).
