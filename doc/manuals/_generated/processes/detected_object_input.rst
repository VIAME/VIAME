  .. |br| raw:: html

   <br />

Configuration
-------------

.. csv-table::
   :header: "Variable", "Default", "Tunable", "Description"
   :align: left
   :widths: auto

   "file_name", "(no default value)", "NO", "Name of the detection set file to read."
   "reader", "(no default value)", "NO", "Algorithm type to use as the reader."

Input Ports
-----------

There are no input ports for this process.


Output Ports
------------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "detected_object_set", "kwiver:detected_object_set", "(none)", "Set of detected objects."
   "image_file_name", "kwiver:image_file_name", "(none)", "Name of an image file. The file name may contain leading path components."

Pipefile Usage
--------------

The following sections describe the blocks needed to use this process in a pipe file.

Pipefile block
--------------

.. code::

 # ================================================================
 process <this-proc>
   :: detected_object_input
 # Name of the detection set file to read.
   file_name = <value>
 # Algorithm type to use as the reader.
   reader = <value>
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
 connect from <this-proc>.detected_object_set
          to   <downstream-proc>.detected_object_set
 connect from <this-proc>.image_file_name
          to   <downstream-proc>.image_file_name

Class Description
-----------------

.. doxygenclass:: kwiver::detected_object_input_process
   :project: kwiver
   :members:

