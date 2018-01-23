  .. |br| raw:: html

   <br />

Configuration
-------------

.. csv-table::
   :header: "Variable", "Default", "Tunable", "Description"
   :align: left
   :widths: auto

   "reject", "false", "NO", "Whether to reject type setting requests or not."

Input Ports
-----------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "input", "_flow_dependent/tag", "(none)", "An input port with a flow dependent type."

Output Ports
------------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "output", "_flow_dependent/tag", "(none)", "An output port with a flow dependent type"

Pipefile Usage
--------------

The following sections describe the blocks needed to use this process in a pipe file.

Pipefile block
--------------

.. code::

 # ================================================================
 process <this-proc>
   :: flow_dependent
 # Whether to reject type setting requests or not.
   reject = false
 # ================================================================

Process connections
~~~~~~~~~~~~~~~~~~~

The following Input ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will consume the following input ports
 connect from <this-proc>.input
          to   <upstream-proc>.input

The following Output ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will produce the following output ports
 connect from <this-proc>.output
          to   <downstream-proc>.output

Class Description
-----------------

.. doxygenclass:: sprokit::flow_dependent_process
   :project: kwiver
   :members:

