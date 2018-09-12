
========================
Examples Folder Overview
========================

In the '[install]/examples' folder, there are a number of subfolders, with each folder corresponding
to a different core functionality. The scripts in each of these folders can be copied to and run
from any directory on your computer, the only item requiring change being the 'VIAME_INSTALL' path at
the top of the each run script. These scripts can be opened and edited in any text editor to point
the VIAME_INSTALL path to the location of your installed (or built) binaries. This is true on both
Windows, Linux, and Mac.

Each example is run in a different fashion, but there are 3 core commands you need to know in
order to run them on Linux:

'bash' - for running commands, e.g. 'bash run_annotation_gui.sh' which launches the application

'ls' - for making file lists of images to process, e.g. 'ls *.png > input_list.txt' to list all
png image files in a folder

'cd' - go into an example directory, e.g. 'cd annotation_and_visualization' to move down into the
annotation_and_visualization example directory. 'cd ..' is another useful command which moves one
directory up, alongside a lone 'ls' command to list all files in the current directory.

To run the examples on Windows, you just need to be able to run (double click) the .bat scripts
in the given directories. Additionally, knowing how to make a list of files, e.g. 'dir > filename.txt'
on the windows command line can also be useful for processing custom image lists.


========================
Key Toolkit Capabilities
========================

****************
Object Detection
****************

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/02/many_scallop_detections_gui.png
   :scale: 50
   :align: center
   :target: https://github.com/Kitware/VIAME/tree/master/examples/object_detection

***********************************
Measuring Fish Lengths Using Stereo
***********************************

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/02/fish_measurement_example.png
   :scale: 50
   :align: center
   :target: https://github.com/Kitware/VIAME/tree/master/examples/measurement_using_stereo

*************************************************
Image and Video Search for Rapid Model Generation
*************************************************

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/01/search_ex.png
   :scale: 50
   :align: center
   :target: https://github.com/Kitware/VIAME/tree/master/examples/search_and_rapid_model_generation

*************************************
GUIs for Visualization and Annotation
*************************************

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/02/annotation_example.png
   :scale: 50
   :align: center
   :target: https://github.com/Kitware/VIAME/tree/master/examples/annotation_and_visualization

***********************************************
Illumination Normalization and Color Correction
***********************************************

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/09/color_correct.png
   :scale: 50
   :align: center
   :target: https://github.com/Kitware/VIAME/tree/master/examples/image_enhancement

*******************************
Detector and Tracker Evaluation
*******************************

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/02/scoring-2.png
   :scale: 50
   :align: center
   :target: https://github.com/Kitware/VIAME/tree/master/examples/scoring_and_roc_generation
