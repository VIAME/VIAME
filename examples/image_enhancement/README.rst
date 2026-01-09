
===============================
Image Enhancement and Filtering
===============================

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/09/color_correct.png
   :scale: 50
   :align: center
   :target: http://www.viametoolkit.org/wp-content/uploads/2018/09/color_correct.png

This document corresponds to the `Image Enhancement`_ example folder within a VIAME
desktop installation. This directory stores assorted scripts for debayering, color
correction, illumination normalization, and general image contrast enhancement.

.. _Image Enhancement: https://github.com/VIAME/VIAME/blob/master/examples/image_enhancement

******************
Build Requirements
******************

These are the build flags required to run this example, if building from the source.

In the pre-built binaries they are all enabled by default.

| VIAME_ENABLE_OPENCV set to ON (required)
| VIAME_ENABLE_VXL set to ON (optional)

********************
Code Used in Example
********************

| plugins/opencv/ocv_debayer_filter.cxx
| plugins/opencv/ocv_debayer_filter.h
| plugins/opencv/ocv_image_enhancement.cxx
| plugins/opencv/ocv_image_enhancement.h
