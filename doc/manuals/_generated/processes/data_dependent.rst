  .. |br| raw:: html

   <br />

Configuration
-------------

.. csv-table::
   :header: "Variable", "Default", "Tunable", "Description"
   :align: left
   :widths: auto

   "reject", "false", "NO", "Whether to reject type setting requests or not."
   "set_on_configure", "true", "NO", "Whether to set the type on configure or not."

Input Ports
-----------

There are no input ports for this process.


Output Ports
------------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "output", "_data_dependent", "(none)", "An output port with a data dependent type"

Pipefile Usage
--------------

The following sections describe the blocks needed to use this process in a pipe file.

Pipefile block
--------------

.. code::

 # ================================================================
 process <this-proc>
   :: data_dependent
 # Whether to reject type setting requests or not.
   reject = false
 # Whether to set the type on configure or not.
   set_on_configure = true
 # ================================================================

Process connections
~~~~~~~~~~~~~~~~~~~

The following Input ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # There are no input port's for this process


The following Output ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will produce the following output ports
 connect from <this-proc>.output
          to   <downstream-proc>.output

Class Description
-----------------

.. doxygenclass:: sprokit::data_dependent_process
   :project: kwiver
   :members:

