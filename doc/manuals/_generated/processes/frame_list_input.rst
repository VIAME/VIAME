  .. |br| raw:: html

   <br />

Configuration
-------------

.. csv-table::
   :header: "Variable", "Default", "Tunable", "Description"
   :align: left
   :widths: auto

   "frame_time", "0.03333333", "NO", "Inter frame time in seconds. The generated timestamps will have the specified |br|\ number of seconds in the generated timestamps for sequential frames. This can be |br|\ used to simulate a frame rate in a video stream application."
   "image_list_file", "(no default value)", "NO", "Name of file that contains list of image file names. Each line in the file |br|\ specifies the name of a single image file."
   "image_reader", "(no default value)", "NO", "Algorithm configuration subblock"
   "path", "(no default value)", "NO", "Path to search for image file. The format is the same as the standard path |br|\ specification, a set of directories separated by a colon (':')"

Input Ports
-----------

There are no input ports for this process.


Output Ports
------------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "image", "kwiver:image", "(none)", "Single frame image."
   "image_file_name", "kwiver:image_file_name", "(none)", "Name of an image file. The file name may contain leading path components."
   "timestamp", "kwiver:timestamp", "(none)", "Timestamp for input image."

Pipefile Usage
--------------

The following sections describe the blocks needed to use this process in a pipe file.

Pipefile block
--------------

.. code::

 # ================================================================
 process <this-proc>
   :: frame_list_input
 # Inter frame time in seconds. The generated timestamps will have the specified
 # number of seconds in the generated timestamps for sequential frames. This can
 # be used to simulate a frame rate in a video stream application.
   frame_time = 0.03333333
 # Name of file that contains list of image file names. Each line in the file
 # specifies the name of a single image file.
   image_list_file = <value>
 # Algorithm configuration subblock
   image_reader = <value>
 # Path to search for image file. The format is the same as the standard path
 # specification, a set of directories separated by a colon (':')
   path = <value>
 # ================================================================

Process connections
~~~~~~~~~~~~~~~~~~~

The following Input ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # There are no input port's for this process


The following Output ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will produce the following output ports
 connect from <this-proc>.image
          to   <downstream-proc>.image
 connect from <this-proc>.image_file_name
          to   <downstream-proc>.image_file_name
 connect from <this-proc>.timestamp
          to   <downstream-proc>.timestamp

Class Description
-----------------

.. doxygenclass:: kwiver::frame_list_process
   :project: kwiver
   :members:

