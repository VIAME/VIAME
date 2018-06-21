
**********************************
Video and Image Search using VIAME
**********************************

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/02/video_query_start.png
   :scale: 30
   :align: center
   :target: https://github.com/Kitware/VIAME/tree/master/examples/search_and_rapid_model_generation/viame_ingest

This section corresponds to `this example online`_, in addition to the
viame_ingest example folder in a VIAME installation. This folder contains examples
covering image search on top of an archive of videos or images.

.. _this example online: https://github.com/Kitware/VIAME/tree/master/examples/search_and_rapid_model_generation/viame_ingest

| WARNING: This example is a work in progress, and should only be attempted
  by advanced users and developers for the time being. 
|
| Building and running this examples requires: 
|
|  (a) The python packages: numpy, pymongo, torch, torchvision, matplotlib, and python-tk
|  (b) A VIAME build with VIAME_ENABLE_SMQTK, BURNOUT, YOLO, OPENCV, PYTORCH, VXL, and VIVIA enabled.

An arbitrary tracking pipeline is used to first generate spatio-temporal object tracks
representing object candidate locations in video. Descriptors are generated around these
object tracks, which get indexed into a database and can be queried upon. By indicating
which query results are correct, a model can be trained for a new object category and
saved to an output file to be reused again in future pipelines or query requests.

First, create_searchable_index.sh should be called to initialize a new database, and populate it
with descriptors around generic objects to be queried upon. Next 'launch_query_gui.sh' should be
called to launch the GUI, new query->image exemplar should be selected, and finally, results
annotated as either correct or incorrect.
