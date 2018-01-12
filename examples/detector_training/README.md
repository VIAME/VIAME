Detector Training API
---------------------

The common detector training API is used for training multiple object
detectors from the same input format for both experimentation and
deployment purposes. By default, each detector has a default training
process that handles issues such as automatically reconfiguring networks
for different output category labels, while simulatenously allowing for
more customization by advanced users. <br>

Future releases will also include the ability to use stereo depth
maps in training, alongside additional forms of data augmentation
and more easily definable data source nodes for alternative input
file structures. <br>

Input data used for training should be put in the following format: <br>

[root_training_dir] <br>
...labels.txt <br>
...folder1 <br>
......image001.png <br>
......image002.png <br>
......image003.png <br>
......groundtruth.gt <br>
...folder2 <br>
......image001.png <br>
......image002.png <br>
......groundtruth.gt <br>

where groundtruth can be in any file format for which a
"detected_object_set_input" implementation exists (e.g. kw18, habcam),
and labels.txt contains a list of output categories (one per line) for
the trained detection model. "labels.txt" can also contain any alternative
names in the groundtruth which map back to the same output category label.
For example, see training_data/labels.txt for the corresponding groundtruth
file in training_data/seq1. <br>

After formatting data, a model can be trained via the 'viame_train_detector'
tool, the only modification required from the scripts in this folder being
setting your .conf files to the correct groundtruth file format type. <br>


Build Requirements
------------------

VIAME_ENABLE_OPENCV set to ON <br>
VIAME_ENABLE_PYTHON set to ON <br>
VIAME_ENABLE_YOLO set to ON (for yolo_v2 training) <br>
VIAME_ENABLE_SCALLOP_TK set to ON (for scallop_tk training) <br>


Code Used in Example
--------------------

plugins/core/viame_train_detector.cxx <br>
packages/kwiver/vital/algo/train_detector.h <br>
packages/kwiver/vital/algo/train_detector.cxx <br>
packages/kwiver/vital/algo/detected_object_set_input.h <br>
packages/kwiver/vital/algo/detected_object_set_input.cxx <br>
packages/kwiver/arrows/darknet/darknet_trainer.h <br>
packages/kwiver/arrows/darknet/darknet_trainer.cxx <br>
plugins/core/detected_object_set_input_habcam.h <br>
plugins/core/detected_object_set_input_habcam.cxx <br>
