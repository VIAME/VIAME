  .. |br| raw:: html

   <br />

Configuration
-------------

.. csv-table::
   :header: "Variable", "Default", "Tunable", "Description"
   :align: left
   :widths: auto

   "footer", "bottom", "NO", "Footer text for image display. Displayed centered at bottom of image."
   "gsd", "3.14159", "NO", "Meters per pixel scaling."
   "header", "top", "NO", "Header text for image display."

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

   "image", "kwiver:image", "(none)", "Single frame image."

Pipefile Usage
--------------

The following sections describe the blocks needed to use this process in a pipe file.

Pipefile block
--------------

.. code::

 # ================================================================
 process <this-proc>
   :: template
 # Footer text for image display. Displayed centered at bottom of image.
   footer = bottom
 # Meters per pixel scaling.
   gsd = 3.14159
 # Header text for image display.
   header = top
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
 connect from <this-proc>.image
          to   <downstream-proc>.image

Class Description
-----------------

.. doxygenclass:: group_ns::template_process
   :project: kwiver
   :members:

