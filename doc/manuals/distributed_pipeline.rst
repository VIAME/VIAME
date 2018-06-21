

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

- serializer and deser are matched pairs and must be able to work together.

2) take some example code for an existing algo.

- best practices (adding tags to assist in error checking)
- Single element algorithm (e.g. json using *cereal* package)
- Multiple element algorithm (message packing)



