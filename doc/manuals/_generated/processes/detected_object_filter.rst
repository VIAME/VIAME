  .. |br| raw:: html

   <br />

Configuration
-------------

.. csv-table::
   :header: "Variable", "Default", "Tunable", "Description"
   :align: left
   :widths: auto

   "filter", "(no default value)", "NO", "Algorithm configuration subblock."

Input Ports
-----------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "detected_object_set", "kwiver:detected_object_set", "_required", "Set of detected objects."

Output Ports
------------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "detected_object_set", "kwiver:detected_object_set", "(none)", "Set of detected objects."

Pipefile Usage
--------------

The following sections describe the blocks needed to use this process in a pipe file.

Pipefile block
--------------

.. code::

 # ================================================================
 process <this-proc>
   :: detected_object_filter
 # Algorithm configuration subblock.
   filter = <value>
 # ================================================================

Process connections
~~~~~~~~~~~~~~~~~~~

The following Input ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will consume the following input ports
 connect from <this-proc>.detected_object_set
          to   <upstream-proc>.detected_object_set

The following Output ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will produce the following output ports
 connect from <this-proc>.detected_object_set
          to   <downstream-proc>.detected_object_set

Class Description
-----------------

.. doxygenclass:: kwiver::detected_object_filter_process
   :project: kwiver
   :members:

