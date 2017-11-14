  .. |br| raw:: html

   <br />

Configuration
-------------

.. csv-table::
   :header: "Variable", "Default", "Tunable", "Description"
   :align: left
   :widths: auto

   "file_name_template", "image%04d.png", "NO", "Template for generating output file names. The template is interpreted as a |br|\ printf format with one format specifier to convert an integer increasing image |br|\ number. The image file type is determined by the file extension and the concrete |br|\ writer selected."
   "image_writer", "(no default value)", "NO", "Config block name to configure algorithm. The algorithm type is selected with |br|\ "image_writer:type". Specific writer parameters depend on writer type selected."

Input Ports
-----------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "image", "kwiver:image", "_required", "Single frame image."
   "timestamp", "kwiver:timestamp", "(none)", "Timestamp for input image."

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
   :: image_writer
 # Template for generating output file names. The template is interpreted as a
 # printf format with one format specifier to convert an integer increasing
 # image number. The image file type is determined by the file extension and the
 # concrete writer selected.
   file_name_template = image%04d.png
 # Config block name to configure algorithm. The algorithm type is selected with
 # "image_writer:type". Specific writer parameters depend on writer type
 # selected.
   image_writer = <value>
 # ================================================================

Process connections
~~~~~~~~~~~~~~~~~~~

The following Input ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will consume the following input ports
 connect from <this-proc>.image
          to   <upstream-proc>.image
 connect from <this-proc>.timestamp
          to   <upstream-proc>.timestamp

The following Output ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will produce the following output ports

Class Description
-----------------

.. doxygenclass:: kwiver::image_writer_process
   :project: kwiver
   :members:

