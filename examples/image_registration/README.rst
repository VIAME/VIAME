===============================
Image Registration
===============================

This document corresponds to `this example online`_, in addition to the
image_registration example folder in a VIAME installation. This directory stores
assorted scripts for performing registration, either temporally across an image sequence
with a certain amount of overlap, or across modalities (e.g. optical and thermal
imagery).

.. _this example online: https://github.com/VIAME/VIAME/blob/master/examples/image_registration


******************
Build Requirements
******************

These are the build flags required to run this example, if building from the source.

In the pre-built binaries OpenCV is enabled by default, though not ITK which is required
for cross-modality registration.

| VIAME_ENABLE_OPENCV set to ON (optional)
| VIAME_ENABLE_ITK set to ON (optional)

********************
Code Used in Example
********************

| plugins/itk/
| plugins/opencv/
