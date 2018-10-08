Distributed Processing
======================


A key component required in  KWIVER to enable the
construction of fully elaborated computer vision systems is a strategy for
multi-processing by distributing Sprokit pipelines across multiple computing
nodes.   This a critical requirement since modern computer vision algrorithms
tend to be resource hungry, especially in the case of deep learning based
alrorithms which require extensive GPU support to run optimally.  KWIVER has
been utilized in systems using the
`The Robotics Operating System (ROS) <http://www.ros.org>>`_ and `Apache
Kafka <https://kafka.apache.org/>`_ among others.

KWIVER, howver,  can use a built-in mechanism for constructing multi-computer
processing systems, with message passing becoming an integral part of the
KWIVER framework. We have chosen to use `ZeroMQ <https://zeromq.org>`_ for the
message passing architecture, because it readily scales from small brokerless
prototype systems to more complex broker based architectures that span many
dozens of communicating elements.

The current ZeroMQ system focuses on “brokerless” processing, relying on pipeline
configuration settings to establish communication topologies. What this means
in practice is that pipelines must be constructed in a way that “knows” where
their communication partners are located in terms of networking (hostname and
ports). While this is sufficient to stand up a number of interesting and useful
systems  it is expected that KWIVER will evolve to providing limited brokering
services to enable more flexibility and dynamism when constructing KWIVER based
multi-processing systems.

KWIVER's multi-processing support is composed of two components:

#. Serialiation
#. Transport

In keeping with KWIVER's architecture, both of these are represented as
abstractions, under which specific implementations (JSON, Protocol Buffers,
ZeroMQ, ROS etc.) can be constructed.


Serialization
-------------

KWIVER’s serialization strategy is based on KWIVER’s arrows. There is a
serialization arrow for each of the VITAL data types. Then there are
implementations for various serialization protocols.  KWIVER supports JSON
based serialization and binary serialization. For binary serialization,
Google’s Protocol Buffers are used. While JSON based serialization makes
reading and debugging types like detected_object_set easy, binary serialization
is used to serialize data heavy elements like images.    As with other KWIVER
arrows, providing new implementations supporting other protocols is
straightforward.

Constructing a Serialization Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Serialization Example
'''''''''''''''''''''

- best practices (adding tags to output to assist in error checking)
- Single element algorithm (e.g. json using *cereal* package)
- Multiple element algorithm (message packing)


Serialization and Deserialization Processes
-------------------------------------------

The KWIVER serialization infrastructure is designed to allow the use
of multiple cooperating Sprokit pipelines to intereact with one another
in a multiprocessig environment.  The purpose of serialization is to package
data in a format (a byte string) that can easily be transmitted or
received via a *transport*.  The purpose of the serialization and
deserialization processes is to convert one or more Sprokit data
ports to and from a single byte string for transmission by a transport
process.

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
''''''''''''''''
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
''''''''''''''''''
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The serializer process always requires the `serialization_type` config
entry. The value supplied is used to select the set of data_serializer
algorithms. If the type specified is `json`, then the data_serializer
will be selected from the 'serialize-json' group. The list of
data_serializer algorithms can be displayed with the following command

`plugin_explorer --fact serialize`


Transport Processes
-------------------

KWIVER’s transport strategy is structured as Sprokit end-caps (the needs and
requirements for the transport implementations are somewhat dependent on
Sprokit’s stream processing implementation and so don’t lend themselves to
implementation as KWIVER arrows). The current implementation focuses on a
one-to-many and many-to-one topologies for VITAL data types.

Transport processes take a serialized message (byte buffer) and
interface to a specific data transport. There are two types of
transport processes, *send* and *receive*. The **send** type processes
take the byte buffer from a serializer process and put it on the
transport.  The **receive** type processes take a message from the
transport and put it on the output port to go to a deserializer
process.. The port name for both types of processes is
"serialized_message".

ZeroMQ Transport
''''''''''''''''

The cannonical implementation of the Sprokit transport processes is based
on ZeroMQ, specifically ZeroMQ’s PUB/SUB pattern with REQ/REP synchronization.

The Sprokit ZeroMQ implementation is contained in two Sprokit processes,
`zmq_transport_send_process` and `zmq_transport_receive_process`:

zmq_transport_send_process
^^^^^^^^^^^^^^^^^^^^^^^^^^

..  doxygenclass:: kwiver::zmq_transport_send_process
    :project: kwiver
    :members:

zmq_transport_receive_process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..  doxygenclass:: kwiver::zmq_transport_receive_process
    :project: kwiver
    :members:


Distributed Pipelines Examples
------------------------------

To demonstrate the use of Sprokit's ZeroMQ distributed processing capabilities,
we'll first need some simple Sprokit pipline files.  The first will generate
some synthetic `detected_object_set` data, serialize it into Protocol Buffers
and transmit the result with ZeroMQ.  Here is a figure that illustrates the
pipeline

.. _zmqmultipubblock:
.. figure:: /_images/zmq_send_pipeline.png
   :align: center

And here is the actual `.pipe` file that implements it::

	process sim :: detected_object_input
					file_name = none
					reader:type = simulator
					reader:simulator:center_x = 100
					reader:simulator:center_y = 100
					reader:simulator:dx = 10
					reader:simulator:dy = 10
					reader:simulator:height = 200
					reader:simulator:width = 200
					reader:simulator:detection_class = "simulated"

	# --------------------------------------------------
	process ser :: serializer
					serialization_type = protobuf

	connect from sim.detected_object_set to ser.dos

	# --------------------------------------------------
	process zmq :: zmq_transport_send
					port = 5560

	connect from ser.dos to zmq.serialized_message

To recieve the data, we'll create another pipeline that recieves the ZeroMQ data,
deserializes it from the Protocoal Buffer container and then writes the resulting
data to a CSV file.  This pipeline looks something like this:

.. _zmqmultipubblock:
.. figure:: /_images/zmq_receive_pipeline.png
   :align: center

The actual `.pipe` file looks like this::

	process zmq :: zmq_transport_receive
		port = 5560
		num_publishers = 1

	# --------------------------------------------------
	process dser :: deserializer
					serialization_type = protobuf

	connect from zmq.serialized_message to dser.dos

	# --------------------------------------------------
	process sink :: detected_object_output
					file_name = received_dos.csv
					writer:type = csv

	connect from dser.dos to sink.detected_object_set

We'll use `pipeline_runner` to start these pipelines.  First, we'll start the
send pipeline::

	pipeline_runner --pipe test_zmq_send.pipe

In a second terminal, we'll start the reciever::

	pipeline_runner --pipe test_zmq_receive.pipe

When the reciever is started, the data flow will start immediately.  At the end of execution
the file `recevied_dos.csv` should contain the transmitted, synthesized `detected_object_set`
data.

Multiple Publishers
^^^^^^^^^^^^^^^^^^^

With the current implementation of the ZeroMQ transport and Sprokit's dynamic configuration
capabilities, we can use these pipelines to create more complex topologies as well.  For
example, we can set up a system with multiple publishers and a reciever that merges the results.
Here is a diagram of such a topology:

.. _zmqmultipubblock:
.. figure:: /_images/zmq_multi_pub.png
   :align: center

We can use the same `.pipe` files by reconfiguring the pipeline on the command line using
`pipeline_runner`.  Here's how we'll start the first sender.  In this case we're simply
changing the `detection_class` configuration for the simulator so that we can identify
this sender's output in the resulting CSV file::

	pipeline_runner --pipe test_zmq_send.pipe --set sim:reader:simulator:detection_class=detector_one

In another terminal we can start a second sender.  In this case we also change the `detection_class`
configuration and we change the ZeroMQ `port` to be two above the default port of `5560`.  This leaves
room for the synchronization port of the first sender and sets up the two senders in the configuration
expected by a multi-publisher receiver::

	pipeline_runner --pipe test_zmq_send.pipe --set sim:reader:simulator:detection_class=detector_two --set zmq:port=5562

Finally, we'll start the reciever.  We'll simply change the `num_publishers` parameter to `2`
so that it connects to both publishers, starting at port `5560` for the first and automatically
adding two to get to `5562` for the second::

	pipeline_runner --pipe test_zmq_recv.pipe --set zmq:num_publishers=2


Multiple Subscribers
^^^^^^^^^^^^^^^^^^^^

In a similar fashion, we can construct topologies where multiple subscribers subscribe
to a single publisher.  Here is a diagram of this topology:

.. _zmqmultisubblock:
.. figure:: /_images/zmq_multi_sub.png
   :align: center

First we'll start our publisher, reconfiguring it to expect `2` subscribers before starting::

	pipeline_runner --pipe test_zmq_send.pipe  --set zmq:expected_subscribers=2

Then, we'll start our first subscriber, changing the output file name to `received_dos_one.csv`::

	pipeline_runner --pipe test_zmq_recv.pipe --set sink::file_name=received_dos_one.csv

Finally, we'll start out second subscriber, this time changing the output file name to `received_dos_two.csv`::

	pipeline_runner --pipe test_zmq_recv.pipe --set sink::file_name=received_dos_two.csv

Worked examples of these pipelines using the `TMUX <https://github.com/tmux/tmux>`_ terminal multiplexor can
be found in `test_zmq_multi_pub_tmux.sh` and `test_zmq_multi_sub_tmus.sh` in the `sprokit/tests/pipelines`
of the KWIVER repository.
