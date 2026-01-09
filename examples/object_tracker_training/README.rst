
=======================
Object Tracker Training
=======================

********
Overview
********

This document corresponds to the `object tracker training`_ example within a VIAME
desktop installation folder.

.. _object detector training: https://github.com/VIAME/VIAME/tree/master/examples/object_tracker_training

Object tracker training can be performed in a similar way to `object detection training`_,
via formatting the directory structure in the same format and launching the train script. Unlike
detection no labels.txt file is necessary, and by default, all images will be used to train
updated tracking models. The necessary format is:

| [root_training_dir] (default: training_data)
| ...folder1
| ......image001.png
| ......image002.png
| ......image003.png
| ......groundtruth.csv
| ...folder2
| ......image001.png
| ......image002.png
| ......groundtruth.csv
| ...etc...
|

.. _object detection training: https://github.com/VIAME/VIAME/tree/master/examples/object_detector_training

******************
Build Requirements
******************

These are the build flags required to run this example, if building from
the source.

In the pre-built binaries they are all enabled by default.

| VIAME_ENABLE_OPENCV set to ON
| VIAME_ENABLE_PYTHON set to ON
| VIAME_ENABLE_PYTORCH set to ON
| VIAME_ENABLE_PYTORCH-SIAMMASK set to ON


********************
Code Used in Example
********************

| plugins/core
| plugins/pytorch
| packages/kwiver/arrows/pytorch
