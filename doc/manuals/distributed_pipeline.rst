Distributed Processing
=============================

The architecture of the sprokit pipeline system is intended to run in a single application where data items
can be passed through memory references. There are times where sprokit based applications need to communicate
with other applications.

Supports communicating between multiple sprokit based applications.
Also supports communicating with applications that use different serialization and messaging packages such as protobuf, ROS, and ZeroMQ.

1) Why would you want to use distributed processing

    - spread processing over multiple computers for better performance
    - interface with other message based systems such as ROS of ZeroMQ

2) How this can be done: differernt appropaches to applying processing

    - case 1: communicating between two sprokit pipeline based applications. Show a large pipeline topology and then break into multiple applications by inserting serialization support into selected edges.
    - case 2: connecting a sprokit pipeline to a different messaging system (e.g. ROS)

3) Describe the different topologies (include diagrams). Show serialization processes, algorithm amd transport processes. Show messages with notional contents for case 1 & 2

4) Describe how the serialization processes work in that they handle multiple threough streams for efficiency. Also cover the cases where multiple serializers are required (based on pipeline topology).

Examples
---------

First example is case 1.

- Show the two pipe files (one for each end of the transport link.)
- Describe how the port names work
- Describe how the serialization algo is selected
- Describe the transport process (notionally?)

Second example is case 2.

- Show the two pipelines but the ports should have groups and items
- Describe how the port names work
- Describe how the serialization algo is selected
- Describe requirements for item names and types
- Describe the transport process (notionally?)


How to create a new data_serializer algorithm
----------------------------------------------

1) describe interface (base class)

All serializer algorithms must be derived from the `data_serializer`
algorithm.  All derived classes must implement the `deserialize()` and
`serialize()` methods, in addition to the configuration support methods.

The `serialize()` method converts one or more data items into a
serialized byte stream. The format of the byte stream depends on the
serialization_type being implemented. Examples of serialization types
are json and protobuf.

The `serialize()` and `deserialize()` methods must be compatible so
that the `deserialize()` method can take the output of the `serialize()`
method and reproduce the original data items.

The `serialize()` method takes in a map of one or more named data
items and the `deserialize()` method produces a similar map. It is up
to the data_serializer implementation to define what these names are
and the associated data types.

Basic one element data_serializer implementations usually do not
require any configuration, but more complicated multi-input
serializers can require an arbitrary amount of config data. These
config parameters are supplied

2) take some example code for an existing algo.

- best practices (adding tags to output to assist in error checking)
- Single element algorithm (e.g. json using *cereal* package)
- Multiple element algorithm (message packing)


How to connect to a [de]serializer process
------------------------------------------

The serializer process dynamically creates serialization algorithms
based on the ports being connected. There are two use cases for
serializing; single data type and multiple data types.

The syntax for serializer input ports and deserializer output ports is
as follows:

    1) <group>
    2) <group>/<algorithm>/<element>

Format 1 is for serializing a single data item. The *group* name will
also be the name of the output port. Format 2 is used when multiple
data items are to be serialized into one output. This ensures that all
the data items arrive together at the other end. The *group* name
specifies the output port name. The *algorithm* specifies the name of
the serializer implementation to use. The *element* specifies the name
of the data item that is passed to the serializer algorithm. Usually
this name means something to the serializer so this can't be a random
name.

Single data type
^^^^^^^^^^^^^^^^
::
    process ser :: serializer
      # Select serialization type
      serialization_type = json

      # specify algorithm config parameters if needed
      serialize-json:image:foo = 16RGB

      connect from input.image to ser.image

    process sink :: ~~~~~
      connect from ser.image  to sink.sink


In this example, the serializer process will instantiate a
serialization algorithm based on the data type associated with input
connection. So, if the data type from `input.image` is a
'kwiver:image', then it will try to instantiate the serializer for
*serialize-json* with an implementation name *kwiver:image*.


Multiple data type
^^^^^^^^^^^^^^^^^^
::
    process ser :: serializer
      # Select serialization type
      serialization_type = protobuf

      # specify algorithm config parameters
      # these are algo specific
      serialize-protobuf:ImageTS:foo = 3.1415

      connect from input.timestamp to ser.first/ImageTS/timestamp
      connect from input.image     to ser.first/ImageTS/image

    process sink :: ~~~~~
      connect from ser.first  to sink.sink

In this example, the serializer process will instantiate a
serialization algorithm based on the algorithm name
(e.g. ImageTS) since it has more than one input port specified. The
data items passed to the `serialize()` method will be "timestamp" and
"image", as taken from the `connect` line in the pipe file.

The [de]serializer process can support multiple serialization streams
if specific data element synchronization is needed.::

  process ser :: serializer
    # Select serialization type
    serialization_type = protobuf

    # Specify algorithm config parameters. These are algo specific.
    # The block name "serialize-protobuf" is specific to the
    # serialization_type. This may not be that useful since it will
    # be applied to instances of that algorithm.
    block serialize-protobuf
      ImageTS:foo = 3.1415
    endblock

    connect from input.timestamp to ser.first/ImageTS/timestamp
    connect from input.image     to ser.first/ImageTS/image
    # output port is "first"

    # additional serialization using the same algorithm, but a different output.
    connect from mask.timestamp to ser.second/ImageTS/timestamp
    connect from mask.image     to ser.second/ImageTS/image
    # output port is "second"

    # additional serialization path based on port data type.
    connect from image_src.image to ser.image
    # output port is "image"


[de]serializer process details
------------------------------

The serializer process always requires the `serialization_type` config
entry. The value supplied is used to select the set of data_serializer
algorithms. If the type specified is `json`, then the data_serializer
will be selected from the 'serialize-json' group. The list of
data_serializer algorithms can be displayed with the following command

`plugin_explorer --fact serialize`


Transport Processes
===================

Transport processes take a serialized message (byte buffer) and
interface to a specific data transport. There are two types of
transport processes, *send* and *receive*. The **send** type processes
take the byte buffer from a serializer process and put it on the
transport.  The **receive** type processes take a message from the
transport and put it on the output port to go to a deserializer
process.. The port name for both types of processes is
"serialized_message".
