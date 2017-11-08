  .. |br| raw:: html

   <br />

Configuration
-------------

.. csv-table::
   :header: "Variable", "Default", "Tunable", "Description"
   :align: left
   :widths: auto

   "draw_algo", "(no default value)", "NO", "Name of drawing algorithm config block."

Input Ports
-----------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "detected_object_set", "kwiver:detected_object_set", "_required", "Set of detected objects."
   "image", "kwiver:image", "_required", "Single frame image."

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
   :: draw_detected_object_set
 # Name of drawing algorithm config block.
   draw_algo = <value>
 # ================================================================

Process connections
~~~~~~~~~~~~~~~~~~~

The following Input ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will consume the following input ports
 connect from <this-proc>.detected_object_set
          to   <upstream-proc>.detected_object_set
 connect from <this-proc>.image
          to   <upstream-proc>.image

The following Output ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will produce the following output ports
 connect from <this-proc>.image
          to   <downstream-proc>.image

Class Description
-----------------

.. doxygenclass:: kwiver::draw_detected_object_set_process
   :project: kwiver
   :members:

