
**********************************
Video and Image Search using VIAME
**********************************

-------
Summary
-------

This section corresponds to `this example online`_, in addition to the
viame_ingest example folder in a VIAME installation. This folder contains examples
covering image search on top of an archive of videos or images.

.. _this example online: https://github.com/Kitware/VIAME/tree/master/examples/search_and_rapid_model_generation/viame_ingest

| Building and running this examples requires either a VIAME install or a build from source with: 
|
|  (a) The python packages: numpy, pymongo, torch, torchvision, matplotlib, and python-tk
|  (b) A VIAME build with VIAME_ENABLE_SMQTK, BURNOUT, YOLO, OPENCV, PYTORCH, VXL, and VIVIA enabled.
|
An arbitrary detection and/or tracking pipeline is used to first generate spatio-temporal
object tracks representing object candidate locations in video or imagery. Descriptors are
generated around these object tracks, which get indexed into a database and can be queried upon.
By indicating which query results are correct, a model can be trained for a new object
category (or sub-category attribute) and saved to an output file to be reused again in future
pipelines or query requests.

-------------
Initial Setup
-------------

First, you should decide where you want to run this example from. Doing it in the example folder
tree is fine as a first pass, but if it is something you plan on running a few times or on multiple
datasets, you probably want to select a different place in your user space to store databases and
model files. This can be accomplished by making a new folder in your directory and either copying
the scripts (.sh, .bat) from this example into this new directory, or via copying the scripts
from [viame-install]/configs/prj-linux (or prj-windows) to this new directory which features
these example scripts alongside others.

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_0_new_project.png
   :scale: 30
   :align: center
   :target: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_0_new_project.png

--------------------------
Ingest Image or Video Data
--------------------------

First, create_index.*.sh should be called to initialize a new database, and populate it
with descriptors around generic objects to be queried upon. Next 'launch_query_gui.sh' should be
called to launch the GUI, new query->image exemplar should be selected, and finally, results
annotated as either correct or incorrect.

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_1_ingest.png
   :scale: 30
   :align: center
   :target: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_1_ingest.png

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_2_ingest.png
   :scale: 30
   :align: center
   :target: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_2_ingest.png

----------------------
Perform an Image Query
----------------------

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_3_launch_gui.png
   :scale: 30
   :align: center
   :target: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_3_launch_gui.png

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_4_new_query.png
   :scale: 30
   :align: center
   :target: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_4_new_query.png

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_5_query_result.png
   :scale: 30
   :align: center
   :target: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_5_query_result.png

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_6_select_fish.png
   :scale: 30
   :align: center
   :target: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_6_select_fish.png

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_7_crop_fish.png
   :scale: 30
   :align: center
   :target: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_7_crop_fish.png

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_8_cropped_fish.png
   :scale: 30
   :align: center
   :target: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_8_cropped_fish.png

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_9_select_fish_again.png
   :scale: 30
   :align: center
   :target: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_9_select_fish_again.png

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_10_initial_results.png
   :scale: 30
   :align: center
   :target: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_10_initial_results.png

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_11_initial_results.png
   :scale: 30
   :align: center
   :target: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_11_initial_results.png

-----------------
Train a IQR Model
-----------------

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_12_adjudacation.png
   :scale: 30
   :align: center
   :target: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_12_adjudacation.png

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_13_feedback.png
   :scale: 30
   :align: center
   :target: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_13_feedback.png

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_14_next_n_results.png
   :scale: 30
   :align: center
   :target: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_14_next_n_results.png

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_15_next_n_results.png
   :scale: 30
   :align: center
   :target: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_15_next_n_results.png

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_16_next_n_results.png
   :scale: 30
   :align: center
   :target: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_16_next_n_results.png

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_17_saved_models.png
   :scale: 30
   :align: center
   :target: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_17_saved_models.png

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_18_produced_detections.png
   :scale: 30
   :align: center
   :target: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_18_produced_detections.png

--------------
Filter Results
--------------

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_19_edited_detections.png
   :scale: 30
   :align: center
   :target: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_19_edited_detections.png

--------------------
Train a Better Model
--------------------

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_20_edited_detections.png
   :scale: 30
   :align: center
   :target: http://www.viametoolkit.org/wp-content/uploads/2018/07/iqr_20_edited_detections.png
