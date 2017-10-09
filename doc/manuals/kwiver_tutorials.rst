KWIVER Tutorials
================
The following links describe a set of kwiver tutorials. 
All the source code mentioned here is provided by the `repository <https://github.com/Kitware/kwiver/examples>`_. 

Visit the `repository <https://github.com/Kitware/kwiver>`_ on how to get and build the KWIVER code base

As always, we would be happy to hear your comments and receive your contributions on any tutorial.

Fundamental Types and Algorithms
--------------------------------

The following tutorials will demonstrate the basic functionality provided in kwiver.
They will focus on the vital types available in kwiver and the various algorithm interfaces currelty supported.
Each example highlights an area of functionality provied in KWIVER.
The KWIVER examples directory contains executable code demonstrating the use of these types with various arrow implementations of the highligted algorithms.

====================================== ==============================================================================
:doc:`Images</vital/images>`            Learn about the fundamental image types and some basic I/O and algorithms    
:doc:`Detection</vital/detections>`     Focus on the data structures and algorithms used by object detection         
====================================== ==============================================================================

Sprokit Pipelines
-----------------

The following tutorials will use Sprokit pipeline files to chain together various algorithms to demonstrate applied examples.
The KWIVER examples directory contains executable pipe files for each of the table entries below.
In order to execute the provided pipeline file, follow the steps to set up KWIVER `here <https://github.com/kwiver#running-kwiver>`_

========================================================== ====================================================================
:doc:`Numbers Flow</pipelines/numbers_flow>`               A simple 'Hello World' pipeline that outputs numbers to a file
:doc:`Image Display</pipelines/image_display>`             A pipe that loads and displays several images
:doc:`Video Display</pipelines/video_display>`             A pipe that loads and displays a video file
:doc:`Hough Detection</pipelines/hough_detection>`         Detect circles in images using a hough detector 
:doc:`Darknet Detection</pipelines/darknet_detection>`     Object detection using the Darnket library
:doc:`Image Stabilization</pipelines/image_stabilization>` Something cool that Matt Brown has done 
========================================================== ====================================================================

.. toctree::
  :hidden:
  
  pipelines/numbers_flow
  pipelines/image_display
  pipelines/video_display
  pipelines/hough_detection
  pipelines/darknet_detection
  pipelines/image_stabilization
  