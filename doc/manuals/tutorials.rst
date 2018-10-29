Tutorials
================

The following links describe a set of kwiver tutorials.
All the source code mentioned here is provided by the `repository <https://github.com/Kitware/kwiver/tree/master/examples>`_.

Visit the `repository <https://github.com/Kitware/kwiver>`_ on how to get and build the KWIVER code base.

Ensure you select the KWIVER_ENABLE_EXAMPLES option during CMake configuration.
This will create a kwiver_examples executable that you use to execute and step any code in the example library.
The kwiver_examples executable is made up of multiple cpp files.
Each file is designed to demonstrait a particular feature in kwiver.
Each file will provide a single entry method for execution.
Each of these entry methods are called from the kwiver_examples main.cpp file.
This main method is intended for you to be able to select and step specific methods by commenting out other methods.

As always, we would be happy to hear your comments and receive your contributions on any tutorial.

Basic Image and Video
---------------------

.. toctree::
  :hidden:

  vital/images
  pipelines/image_display
  pipelines/video_display

Images and video are the most fundamental data needed for computer vision.
The following tutorials will demonstrate the basic functionality provided in kwiver associated with getting image and video data into the framework.

The basic image types and algorithms are defined :doc:`here</vital/images>`

The kwiver_examples file `source/examples/cpp/how_to_part_01_images.cpp <https://github.com/Kitware/kwiver/blob/master/examples/cpp/how_to_part_01_images.cpp>`_ constains code associated with these types and algorithms.
This file demonstrates instantiating and executing various algorithms to load, view, and get data from image and video files on disk.

The following example sprokit pipelines are provided to demonstrait using these algorithms and types in a streaming process.

========================================================== ====================================================================
:doc:`Image Display</pipelines/image_display>`             A pipe that loads and displays several images
:doc:`Video Display</pipelines/video_display>`             A pipe that loads and displays a video file
========================================================== ====================================================================


Detection Types and Algorithms
------------------------------

.. toctree::
  :hidden:

  vital/detectors
  pipelines/example_detection
  pipelines/hough_detection
  pipelines/darknet_detection

Object dectection is the first step in tracking and identifying an activity.
The following tutorials will demonstrate the basic functionality provided in kwiver associated with detecting objets in images and video.

The basic detection types and algorithms are defined :doc:`here</vital/detectors>`

The kwiver_examples file `source/examples/cpp/how_to_part_02_detections.cpp <https://github.com/Kitware/kwiver/blob/master/examples/cpp/how_to_part_02_detections.cpp>`_ constains code associated with these types and algorithms.
This example demonstrates instantiating and executing various detections algorithms on images and video.

The following example sprokit pipelines are provided to demonstrait using these algorithms and types in a streaming process.

========================================================== ====================================================================
:doc:`Example Detection</pipelines/example_detection>`     A very basic implementation of the dection algorithm
:doc:`Hough Detection</pipelines/hough_detection>`         Detect circles in images using a hough detector
:doc:`Darknet Detection</pipelines/darknet_detection>`     Object detection using the Darnket library
========================================================== ====================================================================

Tracking Types and Algorithms
-----------------------------

Coming Soon!

Activity Types and Algorithms
-----------------------------

Coming Soon!
