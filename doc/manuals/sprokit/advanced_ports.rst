********************
Advanced Port Topics
********************

This section discusses advanced port concepts.

Setting type for Port
---------------------

The the majority of cases, the type passed through a port is known
when the process is written and does not change. There are other cases
where the handling of types passed through ports has to be more
flexible in order to support some use cases.

**Any Data Type**

There are some cases where a process just manages the flow of the data
and does not actually look at any data taken from a port, such as the
multiplexer process. In these cases, a specific port type,
``type_any`` can be supplied when the port is declared. Ports with
this type selected can connect to any port with out regard for data
type.

**Data Dependent**

Some processes that can determine the ports type, but must be
configured before the actual data type is known. In these cases, the
port is declared with the data type ``type_data_dependent``. These
processes need to override the ``_set_output_port_type()`` method to
supply the type string for the port. The types for output ports can be
set on a port by port basis as determined by the port name.

All data dependent port data types must be resolved during the
processes ``_configure()`` method.

**Flow Dependent**

There are processes that can handle different data types but require
multiple ports to all have the same type. When a port is declared with
a ``type_flow_dependent`` specification, the data type for that port
is set dynamically based on the connection that is made made. Typical
usage of this type of port is as follows.

.. code-block:: c++

    process::port_t const port_input = port_t("pass");

    declare_input_port(
       port_input, // port name
       type_flow_dependent + "tag",
       required,
       port_description_t("The datum to pass."));

The "tag" that is appended into the flow dependent attribute created a
group of all ports declared with the same tag. The tag is just a
string, so this can be flexible and descriptive. The first port that
is connected from a tagged group establishes the port type for all
other ports (input and output) that have the same tag. Now all ports
in that group must be connected to another port with the first type or
the connection will fail.

The difference between flow dependent and data dependent is that the
type for flow dependent ports are set automatically and the process
must directly set the data dependent ones.

Port Flags
----------

When a port is declared, there is a set of option flags that can be
associated with the new port. These flags can describe the limitations
of the output data provided and the assumptions of the input data. The
individual port flags are discussed in the following sections.

Static port input value
^^^^^^^^^^^^^^^^^^^^^^^

When an input port is declared with the ``flag_input_static`` flag, it
specifies that this input port does not require a connection and if
there is no connection, a static value can be supplied to this port
from the configuration. When a port is declared static, the value
supplied to the port can be configured as follows. ::

   process my_process
    :: my_process_type
       static/port = value

This config specification supplies constant "value" to the port.
If a static port is connected and also has a static value configured,
the configured static value is ignored.

Backward Connections
^^^^^^^^^^^^^^^^^^^^

There are cases when building a pipeline where a backwards edge is
need to procide feedback to a previous stage. This is indicated by
adding the ``flag_input_nodep`` flag when declaring a input port.

This flag indicates that the port is expected to be a *backwards* edge
within the pipeline so that when the pipeline is topologically sorted
(either for initialization or execution order by a scheduler), the
edge can be ignored for such purposes.

Mutability of data passed through ports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Data is passed either by value or by reference. When passed by
reference, multiple porocesses can all be looking at the same data
structure so it is important to declare if a process will modify that
data item or not.

If the mutability of the ports being connected are not compatible,
then the connection is denied.

**output constant**
  Marks that an output is "const" and may not be modified by
  receivers of the data. This is usually used for data that is
  sent, and a reference is kept internally to help reduce memory
  usage.

**output shared**
  A shared port may be connected to exactly one mutable port, any
  number of non-mutable ports, or nothing. Any other usage is a
  data sharing violation and not allowed.

  The downstream processes all share the same instance of the
  port data. A change made from the mutable port is seen by all
  others.

**input mutable**
  Marks that an input is modified within the process and that
  other receivers of the data may see the changes if the data is
  not handled carefully.

If no flags are set, then the ports can be connected if the data types
are compatible.

Required Ports
^^^^^^^^^^^^^^

Ports that have the ``flag_required`` are required to be connected to
another port. If these ports are not connected, the pipeline can not
be constructed. This flag can be applied to both input and output
ports.

Required ports also interact with the stepping of a process. The
KWIVER framework will not step a process unless there is a data item
available on all required input ports. In addition, a process will not
complete untill all data from required output ports has been consumed.

If you have input ports that are not required, ``_step()`` will be
called even if the expected data for those ports have not been
produced by upstream processes and therefore not available.

**************************
Dynamically Creating Ports
**************************

There are use cases where the number and names of input or output
ports are not known until the pipeline is being set up. This situation
can be handled by having a process override the ``input_port_undefined()``
and/or ``output_port_undefined()`` methods. These methods are called
when a connection is being attempted and the requested port has not
been declared. The following code snippet illustrates how to create a
port on demand.

.. code-block:: c++

    input_port_undefined(port_t const& port_name)
    {
      LOG_TRACE( logger(), "Processing undefined input port: \"" << port_name << "\"" );

      if (! kwiver::vital::starts_with( port_name, "_" ) )
      {
        if ( port_not_created_yet( port_name ) )
        {
          // Create input port
          port_flags_t required;
          required.insert( flag_required );

          LOG_TRACE( logger(), "Creating input port: \"" << port_name << "\"" );

          // Open an input port for the name
          declare_input_port(
            port_name,                  // port name
            type_flow_dependent,        // port_type
            required,
            port_description_t( "data type to be serialized" ) );
        }
      }
    }

Output ports can be created in a similar manner.
