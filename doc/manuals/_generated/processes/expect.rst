  .. |br| raw:: html

   <br />

Configuration
-------------

.. csv-table::
   :header: "Variable", "Default", "Tunable", "Description"
   :align: left
   :widths: auto

   "expect", "(no default value)", "NO", "The expected value."
   "expect_key", "false", "NO", "Whether to expect a key or a value."
   "tunable", "(no default value)", "YES", "A tunable value."

Input Ports
-----------

There are no input ports for this process.


Output Ports
------------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "dummy", "_none", "(none)", "A dummy port."

Pipefile Usage
--------------

The following sections describe the blocks needed to use this process in a pipe file.

Pipefile block
--------------

.. code::

 # ================================================================
 process <this-proc>
   :: expect
 # The expected value.
   expect = <value>
 # Whether to expect a key or a value.
   expect_key = false
 # A tunable value.
   tunable = <value>
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
 connect from <this-proc>.dummy
          to   <downstream-proc>.dummy

Class Description
-----------------

.. doxygenclass:: sprokit::expect_process
   :project: kwiver
   :members:

