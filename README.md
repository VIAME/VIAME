
<img src="http://www.viametoolkit.org/wp-content/uploads/2016/08/viami_logo.png" alt="VIAME Logo" width="200" height="78">

VIAME is a computer vision application designed for do-it-yourself artificial intelligence including
object detection, object tracking, image/video annotation, query-based search, image mosaicing,
image enhancement, size measurement, multi-camera data processing, rapid model generation,
and tools for the evaluation of different algorithms. Originally targeting marine species
analytics, VIAME now contains many common algorithms and libraries, and is also useful as a
generic computer vision toolkit. It contains a number of standalone tools for accomplishing
the above, a pipeline framework which can connect C/C++, python, and matlab nodes together
in a multi-threaded fashion, and multiple algorithms resting on top of the pipeline infrastructure.
Lastly, a portion of the algorithms have been integrated into both desktop and web user interfaces
for deployments in different types of environments, with an open annotation archive and example
of the web platform available at [viame.kitware.com](https://viame.kitware.com).


Documentation
-------------

The [User's Quick-Start Guide](https://data.kitware.com/api/v1/item/5fdaf1dd2fa25629b99843f8/download),
[Tutorial Videos](https://www.youtube.com/channel/UCpfxPoR5cNyQFLmqlrxyKJw), 
and [Developer's Manual](http://viame.readthedocs.io/en/latest/) are more comprehensive,
but select entries are also listed below broken down by individual functionality:


[Documentation Overview](https://viame.readthedocs.io/en/latest/#documentation-overview) <>
[Installation](examples/installing_from_binaries) <>
[Building](examples/building_from_source) <>
[All Examples](https://github.com/Kitware/VIAME/tree/master/examples) <>
[DIVE Interface](https://kitware.github.io/dive) <>
[VIEW Interface](examples/annotation_and_visualization) <>
[Search and Rapid Model Generation](examples/search_and_rapid_model_generation) <>
[Object Detector CLI](examples/object_detection) <>
[Object Tracker CLI](examples/object_tracking) <>
[Detector Training CLI](examples/object_detector_training) <>
[Evaluation of Detectors](examples/scoring_and_evaluation) <>
[Detection File Formats](https://viame.readthedocs.io/en/latest/sections/detection_file_conversions.html) <>
[Calibration and Image Enhancement](examples/image_enhancement) <>
[Registration and Mosaicing](examples/registration_and_mosaicing)  <>
[Stereo Measurement and Depth Maps](examples/size_measurement) <>
[Pipelining Overview](https://github.com/Kitware/kwiver) <>
[Core Class and Pipeline Info](https://kwiver.readthedocs.io/en/latest) <>
[Plugin Integration](examples/example_pipeline) <>
[Example Plugin Templates](plugins/templates) <>
[Embedding Algorithms in C++](examples/using_algorithms_in_code)

Installations
-------------

For a full installation guide and description of the various flavors of VIAME, see the
quick-start guide, above. The full desktop version is provided as either a .msi, .zip or
.tar file. Alternatively, standalone annotators (without any processing algorithms)
are available via smaller installers (see DIVE standalone, below). Lastly, docker files
are available for both VIAME Desktop and Web (below). For full desktop installs, extract
the binaries and place them in a directory of your choosing, for example /opt/noaa/viame
on Linux or C:\Program Files\VIAME on Windows. If using packages built with GPU support,
make sure to have sufficient video drivers installed, version 570.11 or higher. The best
way to install drivers depends on your operating system. This isn't required if just
using manual annotators (or frame classifiers only). The binaries are quite large,
in terms of disk space, due to the inclusion of multiple default model files and
programs, but if just building your desired features from source (e.g. for embedded
apps) they are much smaller.

**Installation Requirements:** <br>
* Up to 8 Gb of Disk Space for the Full Installation <br>
* Windows 7\*, 8, 10, or 11 (64-Bit) or Linux (64-Bit, e.g. RHEL, CentOS, Ubuntu) <br>
  * Windows 7 requires some updates and service packs installed, e.g. [KB2533623](https://www.microsoft.com/en-us/download/details.aspx?id=26764). <br>
  * MacOS is currently only supported running standalone annotation tools, see below.

**Installation Recommendations:** <br>
* NVIDIA Drivers (Version 570.11 or above,
Windows 
[\[1\]](https://www.nvidia.com/Download/index.aspx?lang=en-us)
[\[2\]](https://developer.nvidia.com/cuda-downloads)
Ubuntu 
[\[1\]](https://linuxhint.com/ubuntu_nvidia_ppa/)
[\[2\]](https://developer.nvidia.com/cuda-downloads)
CentOS 
[\[1\]](https://developer.nvidia.com/cuda-downloads)
[\[2\]](https://www.nvidia.com/Download/index.aspx?lang=en-us)) <br>
* A [CUDA-enabled GPU](https://developer.nvidia.com/cuda-gpus) with 8 Gb or more VRAM <br>

**Windows Full Desktop Binaries:** <br>
* [VIAME v0.22.0 Windows, GPU Enabled, Mirror1 (.zip)](https://drive.google.com/file/d/1XfNRw8wxrGYTnGq4bXTVskQmbjj7bhJS/view?usp=sharing) <br>
* [VIAME v0.22.0 Windows, GPU Enabled, Mirror2 (.zip)](https://data.kitware.com/api/v1/item/698ca49c5389b41db2e728b5/download) <br>
* [VIAME v0.21.1 Windows, CPU Only, Mirror1 (.zip)](https://drive.google.com/file/d/1WkFHOdMPkxYrow7Utcyx_RTPdG4EbOan/view?usp=sharing) <br>
* [VIAME v0.21.1 Windows, CPU Only, Mirror2 (.zip)](https://data.kitware.com/api/v1/item/683fbc2bdfcff796fee73cfe/download)

**Linux Full Desktop Binaries:** <br>
* [VIAME v0.21.1 Linux, GPU Enabled, Mirror1 (.tar.gz)](https://drive.google.com/file/d/171i23ZjWjCE8-F0bcMZZFAKF7dbAqio5/view?usp=sharing) <br>
* [VIAME v0.21.1 Linux, GPU Enabled, Mirror2 (.tar.gz)](https://data.kitware.com/api/v1/item/683fc309dfcff796fee73d06/download) <br>
* [VIAME v0.21.1 Linux, CPU Only, Mirror1 (.tar.gz)](https://drive.google.com/file/d/1U2H-AE6IwGkClmNEDw-GAETtJTDfKUuR/view?usp=sharing) <br>
* [VIAME v0.21.1 Linux, CPU Only, Mirror2 (.tar.gz)](https://data.kitware.com/api/v1/item/683fbc82dfcff796fee73d01/download)

**Web Applications**: <br>
* [VIAME Online Web Annotator and Public Annotation Archive](https://viame.kitware.com/) <br>
* [VIAME Web Local Installation Instructions](https://kitware.github.io/dive/Deployment-Overview/) <br>
* [VIAME Web Source Repository](https://github.com/Kitware/dive)

**Additional Packages:** <br>
* [DIVE Standalone Desktop Annotator (Linux, Mac, Windows)](https://github.com/Kitware/dive/releases)
* [Model Zoo and Add-Ons](https://github.com/VIAME/VIAME/wiki/Model-Zoo-and-Add-Ons)
* [Custom and Prior VIAME Versions](https://github.com/VIAME/VIAME/wiki/Custom-VIAME-Installations)


Docker Images
-------------

Docker images are available on: https://hub.docker.com. For a default container with just core
algorithms, runnable via command-line, see:

kitware/viame:gpu-algorithms-default

This image is headless (ie, it contains no GUI) and contains a VIAME desktop (not web)
installation in the folder /opt/noaa/viame. For links to the VIAME-Web docker containers see the
above section in the installation documentation. Most add-on models are not included in the
instance but can be downloaded via running the script download_viame_addons.sh in the bin folder.

Quick Build Instructions
------------------------

These instructions are intended for developers or those interested in building the latest release
branch. Anyone interested in just running the software as-is and not modifying it should use the
installers listed in the prior section, without needing to do a software build. More in-depth
build instructions can be found [here](examples/building_from_source), but the software
can be built either as a super-build, which builds most of its dependencies alongside itself, or
standalone. To build VIAME requires, at a minimum, [Git](https://git-scm.com/),
[CMake](https://cmake.org/), and a [C++ compiler](http://www.cplusplus.com/doc/tutorial/introduction/).
Installing Python and CUDA is also recommended. If using CUDA, version 12.8 with CUDNN 9 is most
tested, though other versions of 11 through 13 also likely work. For python distributions, at a
minimum Python 3.10 or above is necessary, alongside having pip installed.

To build on the command line in Linux, use the following commands, only replacing [source-directory]
and [build-directory] with locations of your choice. While these directories can be the same,
it's good practice to have a 'src' checkout then a seperate 'build' directory alongside it:

	git clone https://github.com/VIAME/VIAME.git [source-directory]

	cd [source-directory] && git submodule update --init

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
currently VS2019 is the most tested compiler.

There are several optional arguments to viame which control which plugins get built,
such as those listed below. If a plugin is enabled that depends on another dependency
such as OpenCV) then the dependency flag will be forced to on. If uncertain what to turn
on, it's best to just leave the default enable and disable flags which will build most
(though not all) functionalities. These are core components we recommend leaving turned on:


<center>

| Flag                         | Description                                                                    |
|------------------------------|--------------------------------------------------------------------------------|
| VIAME_ENABLE_OPENCV          | Builds OpenCV and basic OpenCV processes (video readers, simple GUIs)          |
| VIAME_ENABLE_VXL             | Builds VXL and basic VXL processes (video readers, image filters)              |
| VIAME_ENABLE_PYTHON          | Turns on support for using python processes (multiple algorithms)              |
| VIAME_ENABLE_PYTORCH         | Installs all pytorch processes (detectors, trackers, classifiers)              |

</center>


And a number of flags which control which system utilities and optimizations are built, e.g.:


<center>

| Flag                         | Description                                                                    |
|------------------------------|--------------------------------------------------------------------------------|
| VIAME_ENABLE_CUDA            | Enables CUDA (GPU) optimizations across all packages                           |
| VIAME_ENABLE_CUDNN           | Enables CUDNN (GPU) optimizations across all processes                         |
| VIAME_ENABLE_DIVE            | Enables DIVE GUI (annotation and training on multiple sequences)               |
| VIAME_ENABLE_VIVIA           | Builds VIVIA GUIs (VIEW and SEARCH for annotation and video search)            |
| VIAME_ENABLE_DOCS            | Builds Doxygen class-level documentation (puts in install tree)                |
| VIAME_BUILD_DEPENDENCIES     | Build VIAME as a super-build, building all dependencies (default)              |
| VIAME_INSTALL_EXAMPLES       | Installs examples for the above modules into install/examples tree             |
| VIAME_DOWNLOAD_MODELS        | Downloads pre-trained models for use with the examples and interfaces          |

</center>


And lastly, a number of flags which build algorithms or interfaces with more specialized functionality:


<center>

| Flag                         | Description                                                                    |
|------------------------------|--------------------------------------------------------------------------------|
| VIAME_ENABLE_PYTORCH-*       | Builds a number of PyTorch plugins with different functions                    |
| VIAME_ENABLE_TENSORFLOW      | Builds TensorFlow object detector plugin                                       |
| VIAME_ENABLE_TENSORRT        | Builds TensorRT object detector plugin                                         |
| VIAME_ENABLE_DARKNET         | Builds Darknet (YOLO) object detector plugin                                   |
| VIAME_ENABLE_MATLAB          | Turns on support for and installs all matlab processes                         |
| VIAME_ENABLE_KWANT           | Builds KWANT detection and track evaluation (scoring) tools                    |
| VIAME_ENABLE_SEAL            | Builds Seal multi-modality GUI                                                 |
| VIAME_ENABLE_ITK             | Builds ITK cross-modality image registration                                   |

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
update your code, it is important to occasionally re-run:

	git submodule update --init

After switching branches to ensure that you have on the correct hashes
of sub-packages within the build. Very rarely you may also need to run:

	git submodule sync && git submodule update

Just in case the address of submodules has changed. You only need to
run this command if you get a "cannot fetch hash #hashid" error. Lastly,
in the advanced case of running extra manual builds for certain dependencies,
a recursive module update is required:

	git submodule update --init --recursive


License, Citations, and Acknowledgements
----------------------------------------

The core of VIAME is released under a BSD-3 license (see LICENSE.txt).

A non-exhaustive list of relevant papers used within the project alongside contributors
can be found [here](docs/citations.md).

VIAME was developed with funding from multiple sources, with special thanks
to those listed [here](docs/acknowledgements.md).
