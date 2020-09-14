
==========================
Frame Level Classification
==========================

********
Overview
********

This document corresponds to `this online example`_, in addition to the
frame_level_classification example folder in a VIAME installation.

.. _this online example: https://github.com/VIAME/VIAME/tree/master/examples/frame_level_classification

Frame level classification is useful for computing properties such as whether or
not an organism is just within a frame (as opposed to counting instances of it)
or for performing techniques such as background (substrate) classification.

Two methods are provided, training SVM models which is useful for cases with
less training data, and deep (ResNet50) models for standard deep learning
classification when lots of training samples are provided. The third option
for generating full frame classifiers is using the search and rapid model
generation to perform it during video search. When there are lots of full
frame labels, the deep learning method generally yields the best performance.

Training data must be supplied in a similar format to object detector training,
that is the below directory structure (where '...' indicates a subdirectory):
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
