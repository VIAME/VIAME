  .. |br| raw:: html

   <br />

Configuration
-------------

.. csv-table::
   :header: "Variable", "Default", "Tunable", "Description"
   :align: left
   :widths: auto

   "offset", "0", "NO", "The offset from the first datum to use as the output."
   "skip", "1", "NO", "The number of inputs to skip for each output."

Input Ports
-----------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "input", "_flow_dependent/tag", "_required", "A stream with extra data at regular intervals."

Output Ports
------------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "output", "_flow_dependent/tag", "_required", "The input stream sampled at regular intervals."

Pipefile Usage
--------------

The following sections describe the blocks needed to use this process in a pipe file.

Pipefile block
--------------

.. code::

 # ================================================================
 process <this-proc>
   :: skip
 # The offset from the first datum to use as the output.
   offset = 0
 # The number of inputs to skip for each output.
   skip = 1
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

.. doxygenclass:: sprokit::skip_process
   :project: kwiver
   :members:

