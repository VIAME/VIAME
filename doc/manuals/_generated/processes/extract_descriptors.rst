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

   "feature_set", "kwiver:feature_set", "_required", "Set of detected image features."
   "image", "kwiver:image", "_required", "Single frame image."
   "timestamp", "kwiver:timestamp", "_required", "Timestamp for input image."

Output Ports
------------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "descriptor_set", "kwiver:descriptor_set", "(none)", "Set of descriptors."

Pipefile Usage
--------------

The following sections describe the blocks needed to use this process in a pipe file.

Pipefile block
--------------

.. code::

 # ================================================================
 process <this-proc>
   :: extract_descriptors
 # ================================================================

Process connections
~~~~~~~~~~~~~~~~~~~

The following Input ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will consume the following input ports
 connect from <this-proc>.feature_set
          to   <upstream-proc>.feature_set
 connect from <this-proc>.image
          to   <upstream-proc>.image
 connect from <this-proc>.timestamp
          to   <upstream-proc>.timestamp

The following Output ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will produce the following output ports
 connect from <this-proc>.descriptor_set
          to   <downstream-proc>.descriptor_set

Class Description
-----------------

.. doxygenclass:: kwiver::extract_descriptors_process
   :project: kwiver
   :members:

