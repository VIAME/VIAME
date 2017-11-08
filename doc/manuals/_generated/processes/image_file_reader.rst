  .. |br| raw:: html

   <br />

Configuration
-------------

.. csv-table::
   :header: "Variable", "Default", "Tunable", "Description"
   :align: left
   :widths: auto

   "error_mode", "fail", "NO", "How to handle file not found errors. Options are 'abort' and 'skip'. Specifying |br|\ 'fail' will cause an exception to be thrown. The 'pass' option will only log a |br|\ warning and wait for the next file name."
   "image_reader", "(no default value)", "NO", "Algorithm configuration subblock."
   "path", "(no default value)", "NO", "Path to search for image file. The format is the same as the standard path |br|\ specification, a set of directories separated by a colon (':')"

Input Ports
-----------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "image_file_name", "kwiver:image_file_name", "_required", "Name of an image file. The file name may contain leading path components."

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
   :: image_file_reader
 # How to handle file not found errors. Options are 'abort' and 'skip'.
 # Specifying 'fail' will cause an exception to be thrown. The 'pass' option
 # will only log a warning and wait for the next file name.
   error_mode = fail
 # Algorithm configuration subblock.
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

 # This process will consume the following input ports
 connect from <this-proc>.image_file_name
          to   <upstream-proc>.image_file_name

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

.. doxygenclass:: kwiver::image_file_reader_process
   :project: kwiver
   :members:

