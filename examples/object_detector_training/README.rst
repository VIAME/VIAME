
==========================
Detector Training Examples
==========================

This document corresponds to `this example online`_, in addition to the
object_detector_training example folder in a VIAME installation.

.. _this example online: https://github.com/VIAME/VIAME/tree/master/examples/object_detector_training

********
Overview
********

There are a number of ways that models can be trained within the VIAME platform:

1) In the DIVE interface, either through the web or desktop
2) In the command-line interface examples contained within this example
3) SEARCH interface...

*******************
Configuration Files
*******************

Run model training on ground truth annotations. Currently,
training configurations are available to do object
detection, object classification, and full-frame
classification. Tracker training will be added in a future
update.

-  Full-frame classifiers can be trained on arbitrary
   multi-class labels. It's helpful to start with
   ``empty frame lbls`` utility pipe and add type
   annotations to each generated frame.
-  Object classifiers and detectors are trained on bounding
   boxes with arbitrary multi-class labels.

.. rubric:: Overview
   :name: overview

-  SVM (`Support Vector
   Machine <https://en.wikipedia.org/wiki/Support-vector_machine>`__)
   configurations are usable with the smallest amount of
   ground-truth and train relatively quickly.
-  `NetHarn <https://gitlab.kitware.com/computer-vision/netharn>`__
   is a pytorch deep learning framework that requires more
   input data: on the order of thousands of target examples.
   There are two architectures used. Netharn models can take
   up to several days to train.

   -  Cascade Faster R-CNN (cfrnn) for training box
      detectors
   -  Mask R-CNN for training pixel classification and box
      detection
   -  ResNet (Residual Network) for training full frame or
      secondary object classifiers

.. rubric:: Options
   :name: options

|Training options dialog|

.. rubric:: New Model Name
   :name: new-model-name

A recognizable name for the pipeline that results from the
training run.

.. rubric:: Configuration File
   :name: configuration-file

One of the configuration options in the table below.

.. rubric:: Labels.txt file
   :name: labelstxt-file

This **optional** file controls the output classes that a
newly trained model will generate.

-  Use if you annotated using higher granularity labels
   (such as species names) and want to train a classifier
   using more
-  Or you want to restrict your training session to only
   train on certain kinds of ground-truth data.

The following example ``labels.txt`` shows how to train a
``FISH`` classifier by combining ``redfish`` and
``bluefish``, preserve the ``ROCK`` label, and omit every
other label.

.. container:: highlight

   +-----------------------------------+-----------------------------------+
   | .. container:: linenodiv          | .. container::                    |
   |                                   |                                   |
   |    ::                             |    ::                             |
   |                                   |                                   |
   |       1                           |       FISH redfish bluefish       |
   |       2                           |       ROCK                        |
   +-----------------------------------+-----------------------------------+

By default, all classes from all input datasets are
preserved in the output model.

.. rubric:: Use annotation frames only
   :name: use-annotation-frames-only

By default, training runs include all frames from the chosen
input datasets, and frames without annotations are
considered negative examples. If you choose to use annotated
frames only, frames or images with zero annotations will be
discarded. This option is useful for trying to train on
datasets that are only partially annotated.


**********************
Command Line Interface
**********************

The common detector training API is used for training multiple object
detectors from the same input format for both experimentation and
deployment purposes. By default, each detector has a default training
process that handles issues such as automatically reconfiguring networks
for different output category labels, while simulatenously allowing for
more customization by advanced users.

Future releases will also include the ability to use stereo depth
maps in training, alongside additional forms of data augmentation
and more easily definable data source nodes for alternative input
file structures.

| Input data used for training should be put in the following format:
|
| [root_training_dir]
| ...labels.txt
| ...folder1
| ......image001.png
| ......image002.png
| ......image003.png
| ......groundtruth.csv
| ...folder2
| ......image001.png
| ......image002.png
| ......groundtruth.csv
|
where groundtruth can be in any file format for which a
"detected_object_set_input" implementation exists (e.g. viame_csv, kw18, habcam),
and labels.txt contains a list of output categories (one per line) for
the trained detection model. "labels.txt" can also contain any alternative
names in the groundtruth which map back to the same output category label.
For example, see training_data/labels.txt for the corresponding groundtruth
file in training_data/seq1. The "labels.txt" file allows the user to selectively
train models for certain sub-categories or super-categories of object by specifying
only the categories of interest to train a model for, and any synonyms for the
same category on the same line.


After formatting data, a model can be trained via the 'viame_train_detector'
tool, the only modification required from the scripts in this folder being
setting your .conf files to the correct groundtruth file format type.


******************
Build Requirements
******************

These are the build flags required to run this example, if building from
the source.

In the pre-built binaries they are all enabled by default.

| VIAME_ENABLE_OPENCV set to ON
| VIAME_ENABLE_PYTHON set to ON
| VIAME_ENABLE_DARKNET set to ON (for yolo_v2 training)
| VIAME_ENABLE_SCALLOP_TK set to ON (for scallop_tk training)


********************
Code Used in Example
********************

| plugins/core/viame_train_detector.cxx
| packages/kwiver/vital/algo/train_detector.h
| packages/kwiver/vital/algo/train_detector.cxx
| packages/kwiver/vital/algo/detected_object_set_input.h
| packages/kwiver/vital/algo/detected_object_set_input.cxx
| packages/kwiver/arrows/darknet/darknet_trainer.h
| packages/kwiver/arrows/darknet/darknet_trainer.cxx
| plugins/core/detected_object_set_input_habcam.h
| plugins/core/detected_object_set_input_habcam.cxx
