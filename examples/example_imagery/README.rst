
==============
Example Imagery
==============

This folder contains sample image datasets used by the other examples in
this directory. Each subfolder contains images from a specific camera
system or domain, along with any associated ground truth annotations.


Available Datasets
------------------

+--------------------------------+--------------------------------------------+
| Dataset                        | Description                                |
+================================+============================================+
| arctic_seal_example_set1       | Aerial thermal/color imagery for seal      |
|                                | detection                                  |
+--------------------------------+--------------------------------------------+
| camtrawl_example_image_set1    | Underwater trawl camera imagery for        |
|                                | fish species classification                |
+--------------------------------+--------------------------------------------+
| habcam_example_image_set1      | HabCam underwater survey images for        |
|                                | scallop and fish detection                 |
+--------------------------------+--------------------------------------------+
| mouss_example_image_set1       | MOUSS (Modular Optical Underwater Survey   |
|                                | System) bottom fish imagery                |
+--------------------------------+--------------------------------------------+
| raw_auv_image_set1             | Raw AUV (Autonomous Underwater Vehicle)    |
|                                | imagery in raw format                      |
+--------------------------------+--------------------------------------------+
| small_example_image_set1       | Small general-purpose test image set       |
+--------------------------------+--------------------------------------------+


Dataset Details
---------------

**arctic_seal_example_set1**

Aerial imagery from the CHESS (Calibrated High-resolution Enhanced Survey
System) project for detecting Arctic seals. Contains paired thermal (16-bit)
and color (8-bit) images. Used for demonstrating multi-modal detection.

**camtrawl_example_image_set1**

Images from underwater trawl camera systems. Contains fish images with
associated metadata. Used for species classification training examples.

**habcam_example_image_set1**

Images from the HabCam system used for scallop surveys. Filename format
includes timestamp information (YYYYMM.YYYYMMDD.HHMMSSMMM.frame.png).
Used for scallop detection and measurement examples.

**mouss_example_image_set1**

Bottom camera imagery from the MOUSS system. Used for training and testing
fish detection models. Filename format includes timestamp and frame number.

**raw_auv_image_set1**

Raw images from AUV missions that may require Bayer pattern decoding or
other preprocessing. Used for image enhancement examples.

**small_example_image_set1**

A small, general-purpose image set for quick testing. Useful for verifying
that pipelines and algorithms are working before processing larger datasets.


Using with Examples
-------------------

Other example directories reference these datasets. For example:

.. code-block:: bash

   # In object_detection examples:
   ./run_detector_on_habcam.sh
   # Uses: example_imagery/habcam_example_image_set1/

Most example scripts automatically locate these datasets relative to their
own location. If you move datasets, update the paths in the scripts.

