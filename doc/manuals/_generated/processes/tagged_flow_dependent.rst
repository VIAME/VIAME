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

   "tagged_input", "_flow_dependent/tag", "(none)", "A tagged input port with a flow dependent type."
   "untagged_input", "_flow_dependent/", "(none)", "An untagged input port with a flow dependent type."

Output Ports
------------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "tagged_output", "_flow_dependent/tag", "(none)", "A tagged output port with a flow dependent type"
   "untagged_output", "_flow_dependent/", "(none)", "An untagged output port with a flow dependent type"

Pipefile Usage
--------------

The following sections describe the blocks needed to use this process in a pipe file.

Pipefile block
--------------

.. code::

 # ================================================================
 process <this-proc>
   :: tagged_flow_dependent
 # ================================================================

Process connections
~~~~~~~~~~~~~~~~~~~

The following Input ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will consume the following input ports
 connect from <this-proc>.tagged_input
          to   <upstream-proc>.tagged_input
 connect from <this-proc>.untagged_input
          to   <upstream-proc>.untagged_input

The following Output ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will produce the following output ports
 connect from <this-proc>.tagged_output
          to   <downstream-proc>.tagged_output
 connect from <this-proc>.untagged_output
          to   <downstream-proc>.untagged_output

Class Description
-----------------

.. doxygenclass:: sprokit::tagged_flow_dependent_process
   :project: kwiver
   :members:

