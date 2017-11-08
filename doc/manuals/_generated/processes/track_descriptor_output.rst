  .. |br| raw:: html

   <br />

Configuration
-------------

.. csv-table::
   :header: "Variable", "Default", "Tunable", "Description"
   :align: left
   :widths: auto

   "file_name", "(no default value)", "NO", "Name of the track descriptor set file to write."
   "writer", "(no default value)", "NO", "Block name for algorithm parameters. e.g. writer:type would be used to specify |br|\ the algorithm type."

Input Ports
-----------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "image_file_name", "kwiver:image_file_name", "(none)", "Name of an image file. The file name may contain leading path components."
   "track_descriptor_set", "kwiver:track_descriptor_set", "_required", "Set of track descriptors."

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
   :: track_descriptor_output
 # Name of the track descriptor set file to write.
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
 connect from <this-proc>.image_file_name
          to   <upstream-proc>.image_file_name
 connect from <this-proc>.track_descriptor_set
          to   <upstream-proc>.track_descriptor_set

The following Output ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will produce the following output ports

Class Description
-----------------

.. doxygenclass:: kwiver::track_descriptor_output_process
   :project: kwiver
   :members:

