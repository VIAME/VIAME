  .. |br| raw:: html

   <br />

Configuration
-------------

.. csv-table::
   :header: "Variable", "Default", "Tunable", "Description"
   :align: left
   :widths: auto

   "computer", "(no default value)", "NO", "Algorithm configuration subblock"

Input Ports
-----------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "left_image", "kwiver:image", "_required", "Single frame left image."
   "right_image", "kwiver:image", "_required", "Single frame right image."

Output Ports
------------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "depth_map", "kwiver:image", "(none)", "Depth map stored in image form."

Pipefile Usage
--------------

The following sections describe the blocks needed to use this process in a pipe file.

Pipefile block
--------------

.. code::

 # ================================================================
 process <this-proc>
   :: compute_stereo_depth_map
 # Algorithm configuration subblock
   computer = <value>
 # ================================================================

Process connections
~~~~~~~~~~~~~~~~~~~

The following Input ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will consume the following input ports
 connect from <this-proc>.left_image
          to   <upstream-proc>.left_image
 connect from <this-proc>.right_image
          to   <upstream-proc>.right_image

The following Output ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will produce the following output ports
 connect from <this-proc>.depth_map
          to   <downstream-proc>.depth_map

Class Description
-----------------

.. doxygenclass:: kwiver::compute_stereo_depth_map_process
   :project: kwiver
   :members:

