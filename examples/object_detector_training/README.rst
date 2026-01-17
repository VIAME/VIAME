
==========================
Detector Training Examples
==========================

This document corresponds to the `Object Detector Training`_ example folder
within a VIAME desktop installation.

.. _Object Detector Training: https://github.com/VIAME/VIAME/tree/master/examples/object_detector_training

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


After formatting data, a model can be trained via the 'viame' tool with a
training configuration file, the only modification required from the scripts
in this folder being setting your .conf files to the correct groundtruth
file format type.

************
Labels Files
************

labels.txt files offer the ability to modify which categories models
are being trained over, specify synonyms for the same output category,
and lastly specify class hierachies. Synonyms can be specified on the 
same line, categories to not train offer ommitted, and parent classes
indicated via the ":parent=classname" indicator. 

For example, in the below the model will have two output classes, "speciesA"
and "speciesD" with "speciesA" comprised of species "speciesA", "speciesB",
and "speciesC" 

| speciesA speciesB speciesC
| speciesD

In the below three classes will be output, but speciesB will not be present

| speciesA
| speciesC
| speciesD

In the below three classes will also be output, and a hierachical specifier
is included for models which make use of hierachical information. In this
case, speciesC is indicated as a child class of genusA (e.g. if class A is
a higher order, e.g. genus, and classC a particular species in this case).

| genusA
| speciesC :parent=genusA
| speciesD

If no labels.txt is specified during training, then all unique labels across
all inputs will be used across training.

******************
Build Requirements
******************

These are the build flags required to run this example, if building from
the source. In the pre-built binaries they are all enabled by default.

| VIAME_ENABLE_OPENCV set to ON
| VIAME_ENABLE_PYTHON set to ON
| VIAME_ENABLE_DARKNET set to ON (for yolo_v2 training)


********************
Code Used in Example
********************

| tools/train.cxx
| packages/kwiver/vital/algo/train_detector.h
| packages/kwiver/vital/algo/train_detector.cxx
| packages/kwiver/vital/algo/detected_object_set_input.h
| packages/kwiver/vital/algo/detected_object_set_input.cxx
| packages/kwiver/arrows/darknet/darknet_trainer.h
| packages/kwiver/arrows/darknet/darknet_trainer.cxx
| plugins/core/detected_object_set_input_habcam.h
| plugins/core/detected_object_set_input_habcam.cxx
