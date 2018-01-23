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

   "d_vector", "kwiver:d_vector", "_required", "Vector of doubles from descriptor"

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
   :: read_d_vector
 # ================================================================

Process connections
~~~~~~~~~~~~~~~~~~~

The following Input ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will consume the following input ports
 connect from <this-proc>.d_vector
          to   <upstream-proc>.d_vector

The following Output ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will produce the following output ports

Class Description
-----------------

.. doxygenclass:: kwiver::read_descriptor_process
   :project: kwiver
   :members:

