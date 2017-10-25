frame_list_input
================

.. _frame_list_input:

Reads a list of image file names and generates stream of images and associated
time stamps

Configuration
-------------

.. csv-table:: 
   :header: "Variable", "Default", "Tunable", "Description"
   :align: left
   :widths: auto

   "frame_time", 0.03333333, "No", "Inter frame time in seconds." 
   "image_list_file", (no default value), "No", "Name of file that contains list of image file names. Each line in the file 
   specifies the name of a single image file."
   "image_reader", (no default value), "No", ":ref:`image_io impl_name option<algo_image_io>`" 
   "path", (no default value), "No", "Path to search for image file. The format is the same as the standard path 
   specification, a set of directories separated by a colon (':')"

Input Ports
~~~~~~~~~~~

*There are no input port's for this process*

Output Ports
~~~~~~~~~~~~

.. csv-table:: 
   :header: "Variable", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "image", "kwiver:image", "(none)", "Single frame image." 
   "image_file_name", kwiver:image_file_name, "(none)", "Name of an image file. The file name may contain leading path components."
   "timestamp", kwiver:timestamp, "(none)", "Timestamp for input image." 

Pipefile Usage
--------------
The following sections describe the blocks needed to use this process in a pipe file

Process Declaration
~~~~~~~~~~~~~~~~~~~

.. code::

 # ================================================================
 process input
   :: frame_list_input
 # Input file containing new-line separated paths to sequential image files.
   image_list_file = C:/Programming/KWIVER/builds/release/examples/pipelines/image_list.txt
   frame_time = .9
 # image_io algorithm to use for 'image_reader'.
  image_reader:type = ocv
 # ================================================================

Available :ref:`image_io impl_name options<algo_image_io>` for image_reader:type

Process connections
~~~~~~~~~~~~~~~~~~~~

The following Input ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # There are no input port's for this process

        
The follwing Output ports are available from this process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will provide these vital objects to any processes
 connect from input.timestamp
         to   processX.timestamp
 connect from input.image
         to   processX.image

Class Description
-----------------
        
..  doxygenclass:: kwiver::frame_list_process
    :project: kwiver
    :members: