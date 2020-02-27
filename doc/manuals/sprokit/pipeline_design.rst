Pipeline Design
===============

Overview
--------

The design of the new pipeline is meant to address issues that have come up
before and to add functionality that has been wanted for a while including
Python support, interactive pipeline debugging, better concurrency support,
and more.

Type Safety
-----------

The codebase strives for type safety where possible. This is achieved by
using ``typedef`` to rename types. When applicable, ``typedef`` types also
expose objects through only a ``shared_ptr`` to prevent unintentional deep
copies from occurring and simplify memory management.

The use of ``typedef`` within the codebase also simplifies changing core types
if necessary (e.g., replacing ``std::shared_ptr`` with a different managed
pointer class).

Some of the core classes (i.e., ``sprokit::datum`` and ``sprokit::stamp``) are
immutable through their respective ``typedef`` and can only be created with
static methods of the respective class which enforce that they can only be
constructed in specific ways.

.. doxygenclass:: sprokit::datum
                  :project: kwiver
                  :members:

Introspection
-------------

Processes are designed to be introspected so that information about a process
can be given at runtime. It also allows processes to be created at runtime
and pipelines created dynamically. By abstracting out C++ types, language
bindings do not need to deal with templates, custom bindings for every
plugin, and other intricacies that bindings to C++ libraries usually entail.

Thread safety
-------------

Processes within the new pipeline are encouraged to be thread safe. When
thread safety cannot be ensured, it must be explicitly marked. This is so
that any situation where data is shared across threads where more than one
thread expects to be able to modify the data is detected as an error.

Error Handling
--------------

Errors within the pipeline are indicated with exceptions. Exceptions allow
the error to be handled at the appropriate level and if the error is not
caught, the message will reach the user. This forces ignoring errors to be
explicit since not all compilers allow decorating functions to warn when
their return value is ignored.

Control Flow
------------

The design of the ``sprokit::process`` class is such that the heavy lifting is
done by the base class and specialized computations are handled as needed by
a subclass. This allows a new process to be written with a minimum amount of
boilerplate. Where special logic is required, a subclass can implement a
``virtual`` method which can add supplemental logic to support a feature.

For example, when information about a port is requested, the
``sprokit::process::input_port_info`` method is called which delegates logic to the
``sprokit::process::_input_port_info`` method which can be overridden. By
default, it returns information about the port if it has been declared,
otherwise it calls ``sprokit::process::input_port_undefined``.
To create ports on the fly, a process can reimplement
``sprokit::process::input_port_undefined`` to dynamically create the port
so that it exists and an exception is not thrown.

The same applies to ``sprokit::process::output_port_undefined``.

The rationale for not making ``sprokit::process::input_port_info`` ``virtual``
is to enforce that API specifications are met. For example, when connecting
edges, the main method makes sure that the edge is not ``NULL`` and that the
process has not been initialized yet.

Data Flow
---------

Data flows within the pipeline via the ``sprokit::edge`` class which ensures
thread-safe communication between processes. A process communicates with
edges via its input and output ports. Ports are named communication sockets
where edges may be connected to so that a process can send and receive data.
Input ports may have at most one edge sending data to it while output ports
may feed into any number of edges.

Ports
-----

Ports are declared within a process and managed by the base
``sprokit::process`` class to minimize the amount of code that needs
to be written to handle communication within the pipeline.

A port has a "type" associated with it which is used to detect errors
when connecting incompatible ports with each other. These types are
*logical* types, not a type within a programming language. A
*double* can represent a distance or a time interval (or even a
distance is a different unit!), but a port which uses a *double* to
a distance would have a type of *distance_in_meters*, *not*
*double*.
In addition to comcrete port types, there are two special types,
one of which indicates that
any type is accepted on the port and another which indicates that no
data is ever expected on the port.

Ports can also have flags associated with them. Flags give extra information
about the data that is expected on a port. A flag can indicate that the data
on the port must be present to make any sense (either it's required for a
computation or that if the result is ignored, there's no point in doing the
computation in the first place), the data on the port should not be modified
(because it is only a shallow copy and other processes modifying the data
would invalidate results), or that the data for the port will be modified
(used to cause errors when connected to a port with the previous flag). Flags
are meant to be used to bring attention to the fact that more is happening to
data that flows through the port than normal.

Packets
-------

Each data packet within an edge is made up of two parts: a status packet and
a stamp. The stamp is used to ensure that the various flows through the
pipeline are synchronized.

The status packet indicates the result of the computation that creates the
result available on a port. It can indicate that the computation succeeded
(with the result), failed (with the error message), could not be completed
for some reason (e.g., not enough data), or complete (the input data is
exhausted and no more results can be made). Having a status message for each
result within the pipeline allows for more fine-grained data dependencies to
be made. A process which fails to get some extra data related to its main
data stream (e.g., metadata on a video frame) does not have to create invalid
objects nor indicate failure to other, unrelated, ports.

A stamp consists of a step count and an increment. If two stamps have the
same step count. A stamp's step count is incremented at the source for each
new data element. Step counts are unitless and should only be used for
ordering information. In fact, the ``sprokit::stamp`` interface enforces this
and only provides a comparison operator between stamps. Since step counts
are unitless and discrete, inserting elements into the stream requires that
the step counts change.

The base ``sprokit::process`` class handles the common case for incoming and
outgoing data. The default behavior is that if an input port is marked as
being "required", its status message is aggregated with other required
inputs:

- If a required input is complete, then the current process' computation is
  considered to be complete as well.
- Otherwise, if a required input is an error message, then the current
  process' computation is considered an error due to an error as input
  (following the GIGO principle).
- Otherwise, if a required input is empty, then the current process'
  computation is considered empty (the computation is missing data and
  cannot be completed).
- Then, since all of the required inputs are available, the stamps are
  checked to ensure that they are on the same step count.

If custom logic is required to manage ports or data, this control flow can be
disabled piecemeal and handled manually. The status can check can be disabled
on a per-process basis so that it can be managed in a special way.

<Need to describe synchronization modes>
- How do optional input ports interact with the above?



Pipeline Execution
------------------

The execution of a pipeline is separate from the construction and
verification. This allows specialized schedulers to be used in situations
where some resource is constrained (one scheduler to keep memory usage low,
another to minimize CPU contention, another for an I/O-heavy pipeline, and
others).
