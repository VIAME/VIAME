Darknet Detection
=================

The following pipelines will take in a set of images or a video file.
Each frame will be evaluated by the Darknet yolo algorithm with a weight file that was trained on the virat data set.
This weight file will identify any 'person' or 'vehicle' objects in the image.
The detections will then be drawn on the input image and displayed to the user, and written to disk.


Setup
-----

In order to execute pipeline files, follow `these <https://github.com/kwiver#running-kwiver>`_ steps to set up KWIVER

In order to run the pipelines associated with this tutorial you will need to download the associated data package.
The download process is done via targets created in the build process.
In a bash terminal in your KWIVER build directory, make the following targets::

  make external_darknet_example
  make setup_darknet_example

If you are using Visual Studio, manually build the external_darknet_example project, followed by the setup_darknet_example project.

This will pull, place, and configure all the data associated with thise exampe into <your KWIVER build directory>/examples/pipeline/darknet folder

The following files will be in the <build directory>/examples/pipelines/darknet folder:

  - images - Directory containing images used in this example
  - models - Directory containing configuration and weight files needed by Darknet
  - output - Directory where new images will be placed when the pipeline executes
  - video - Directory containing the video used in this example
  - configure.cmake - CMake script to set configure *.in files specific to your system
  - darknet_image.pipe - The pipe file to run Darknet on the provided example images
  - darknet_image.pipe.in - The pipe file to be configured to run on your system
  - darknet_video.pipe - The pipe file to run Darknet on the provided example video
  - darknet_video.pipe.in - The pipe file to be configured to run on your system
  - image_list.txt - The images to be used by the darknet_image.pipe file
  - image_list.txt.in - The list file to be configured to run on your system
  - readme.txt - This tutorial supersedes content in this file

Execution
---------

Run the following command from the kwiver build\bin directory (bin/release on windows)
Relativly point to the darknet_image.pipe or darknet_video.pipe file like this::

  # Windows Example :
  kwiver runner ..\..\examples\pipelines\darknet\darknet_image.pipe
  # Linux Example :
  ./kwiver runner ../examples/pipelines/darknet/darknet_image.pipe

  # Windows Example :
  kwiver runner ..\..\examples\pipelines\darknet\darknet_video.pipe
  # Linux Example :
  ./kwiver runner ../examples/pipelines/darknet/darknet_video.pipe

NOTE, you will need to supply a video file for the darknet_video pipe at this time.
We will update the zip contents ASAP.

The darknet_image.pipe file will put all generated output to the examples/pipelines/darknet/output/images

The darknet_video.pipe file will put all generated output to the examples/pipelines/darknet/output/video


Image Detection
~~~~~~~~~~~~~~~

Process Graph
-------------

darknet_image
~~~~~~~~~~~~~

The following image displays the pipeline graph.
Each process is linked to its associated definition page to learn more about it and the algorithms it uses.

.. graphviz:: ../_generated/graphviz/darknet_image.gv

darknet_video
~~~~~~~~~~~~~

The following image displays the pipeline graph.
Each process is linked to its associated definition page to learn more about it and the algorithms it uses.

.. graphviz:: ../_generated/graphviz/darknet_video.gv
