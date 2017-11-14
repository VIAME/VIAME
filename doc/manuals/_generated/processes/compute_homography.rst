  .. |br| raw:: html

   <br />

Configuration
-------------

.. csv-table::
   :header: "Variable", "Default", "Tunable", "Description"
   :align: left
   :widths: auto


Input Ports
-----------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "feature_track_set", "kwiver:feature_track_set", "_required", "Set of feature tracks."
   "timestamp", "kwiver:timestamp", "_required", "Timestamp for input image."

Output Ports
------------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "homography_src_to_ref", "kwiver:s2r_homography", "(none)", "Source image to ref image homography."

Pipefile Usage
--------------

The following sections describe the blocks needed to use this process in a pipe file.

Pipefile block
--------------

.. code::

 # ================================================================
 process <this-proc>
   :: compute_homography
 # ================================================================

Process connections
~~~~~~~~~~~~~~~~~~~

The following Input ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will consume the following input ports
 connect from <this-proc>.feature_track_set
          to   <upstream-proc>.feature_track_set
 connect from <this-proc>.timestamp
          to   <upstream-proc>.timestamp

The following Output ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will produce the following output ports
 connect from <this-proc>.homography_src_to_ref
          to   <downstream-proc>.homography_src_to_ref

Class Description
-----------------

.. doxygenclass:: kwiver::compute_homography_process
   :project: kwiver
   :members:

