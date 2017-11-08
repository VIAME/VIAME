  .. |br| raw:: html

   <br />

Configuration
-------------

.. csv-table::
   :header: "Variable", "Default", "Tunable", "Description"
   :align: left
   :widths: auto

   "file_name", "(no default value)", "NO", "Name of the detection set file to write."
   "writer", "(no default value)", "NO", "Block name for algorithm parameters. e.g. writer:type would be used to specify |br|\ the algorithm type."

Input Ports
-----------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "detected_object_set", "kwiver:detected_object_set", "_required", "Set of detected objects."
   "image_file_name", "kwiver:image_file_name", "(none)", "Name of an image file. The file name may contain leading path components."

Output Ports
------------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto


Pipefile Usage
--------------

The following sections describe the blocks needed to use this process in a pipe file.

Pipefile block
--------------

.. code::

 # ================================================================
 process <this-proc>
   :: detected_object_output
 # Name of the detection set file to write.
   file_name = <value>
 # Block name for algorithm parameters. e.g. writer:type would be used to
 # specify the algorithm type.
   writer = <value>
 # ================================================================

Process connections
~~~~~~~~~~~~~~~~~~~

The following Input ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will consume the following input ports
 connect from <this-proc>.detected_object_set
          to   <upstream-proc>.detected_object_set
 connect from <this-proc>.image_file_name
          to   <upstream-proc>.image_file_name

The following Output ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will produce the following output ports

Class Description
-----------------

.. doxygenclass:: kwiver::detected_object_output_process
   :project: kwiver
   :members:

