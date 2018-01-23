  .. |br| raw:: html

   <br />

Configuration
-------------

.. csv-table::
   :header: "Variable", "Default", "Tunable", "Description"
   :align: left
   :widths: auto

   "copies", "1", "NO", "The number of copies to make of each input."

Input Ports
-----------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "input", "_flow_dependent/tag", "_required", "Arbitrary input data."

Output Ports
------------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "duplicate", "_flow_dependent/tag", "_required, _shared", "Duplicated input data."

Pipefile Usage
--------------

The following sections describe the blocks needed to use this process in a pipe file.

Pipefile block
--------------

.. code::

 # ================================================================
 process <this-proc>
   :: duplicate
 # The number of copies to make of each input.
   copies = 1
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
 connect from <this-proc>.duplicate
          to   <downstream-proc>.duplicate

Class Description
-----------------

.. doxygenclass:: sprokit::duplicate_process
   :project: kwiver
   :members:

