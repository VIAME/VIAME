Darknet Detection
=================

Setup
-----

In order to execute pipeline files, follow `these <https://github.com/kwiver#running-kwiver>`_ steps to set up KWIVER

In order to run the pipelines associated with this tutorial you will need to download the associated data package.
The download process is done via targets created in the build process.
In a bash terminal in your KWIVER build directory, make the following targets::

  make external_darknet_data
  make setup_darknet_example

If you are using Visual Studio, manually build the external_darknet_data project, followed by the setup_darknet_example project.

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

Run the following command from the kwiver build\bin directory (bin\release on windows)
Relativly point to the darknet_image.pipe or darknet_video.pipe file like this::
  
  # Windows Example : 
  pipeline_runner -p ..\..\examples\pipelines\darknet\darknet_image.pipe
  # Linux Example : 
  ./pipeline_runner -p ../examples/pipelines/darknet/darknet_image.pipe

The darknet_image.pipe file will put all generated output to the examples/pipelines/darknet/output/images

The darknet_video.pipe file will put all generated output to the examples/pipelines/darknet/output/video

We will dig into more details for each pipeline file in the following sections.
  
Image Detection
~~~~~~~~~~~~~~~

The darknet_image.pipe file will run a pre-trained YOLO v2 object detector from darknet against the provided image files.
The detector is trained to identify people and vehicles in images.

Follow these links for more information about pipeline design and files.

This pipefile will execute the following processes for each image specified:

+----------------------------------------------------------------------------------------------------------------------+
| Processes                                                                                                            |
+======================================================================================================================+
|:Name: input                                                                                                          |
|:Type: :doc:`frame_list_input<../sprokit/processes/frame_list_input>`                                                 |
|:Description: Reads the images in the image_list.txt file                                                             |
+----------------------------------------------------------------------------------------------------------------------+
|:Name: yolo_v2                                                                                                        |
|:Type: :doc:`image_object_detector<../sprokit/processes/image_object_detector>`                                       |
|:Description: Configured to use the darknet implementation of image_object_detector                                   |
+----------------------------------------------------------------------------------------------------------------------+
|:Name: draw                                                                                                           |
|:Type: :doc:`draw_detected_object_boxes<../sprokit/processes/draw_detected_object_boxes>`                             |
|:Description: Creates a copy of the current image, then draw the detection boxes on it created by the yolo_v2 process |
+----------------------------------------------------------------------------------------------------------------------+
|:Name: disp                                                                                                           |
|:Type: :doc:`image_viewer<../sprokit/processes/image_viewer>`                                                         |
|:Description: Shows the new image copy with detection boxes in a window as the pipeline runs                          |
+----------------------------------------------------------------------------------------------------------------------+
|:Name: write                                                                                                          |
|:Type: :doc:`image_writer<../sprokit/processes/image_writer>`                                                         |
|:Description: Writes the new image copy with detection boxes to the specified directory                               |
+----------------------------------------------------------------------------------------------------------------------+
|:Name: yolo_v2_kw18_writer                                                                                            |
|:Type: :doc:`detected_object_output<../sprokit/processes/detected_object_output>`                                     |
|:Description: Writes the detected_object_set object to an ascii file in kw18 format                                   |
+----------------------------------------------------------------------------------------------------------------------+
|:Name: yolo_v2_csv_writer                                                                                             |
|:Type: :doc:`detected_object_output<../sprokit/processes/detected_object_output>`                                     |
|:Description: Writes the detected_object_set object to an ascii file in csv format                                    |
+----------------------------------------------------------------------------------------------------------------------+


Video Detection
~~~~~~~~~~~~~~~

TODO


