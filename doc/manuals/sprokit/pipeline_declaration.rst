Pipeline Declaration Files
==========================

Pipeline declaration files allow a pipeline to be loaded from a plain
text description. They provide all of the information necessary to
create and run a pipeline and may be composed of files containing
pipeline specification information that are included into the main
file

The '#' character is used to introduce a comment. All text from the
'#' to the end of the line are considered comments.

A pipeline declaration file is made up of the following sections:

- Configuration Section
- Process Definition Section
- Connection Definition


Configuration Entries
---------------------

Configuration entries are statements which add an entry to the
configuration block for the pipeline. The general form for a
configuration entry is a key / value pair, as shown below:

``key = value``

The key specification can be hierarchical and be specified with
multiple components separated by a ':' character. Key components are
described by the following regular expression ``[a-zA-Z0-9_-]+``.

``key:component:list = value``

Each leading key component (the name before the ':') establishes a
subblock in the configuration. These subblocks are used to group
configuration entries for different sections of the application.

The value for a configuration entry is the character string that
follows the '=' character. The value has leading and trailing blanks
removed. Embedded blanks are preserved without the addition of
enclosing quotes. If quotes are used in the value portion of the
configuration entry, they are not processed in any way and remain part
of the value string. That is, if you put quotes in the value component
of a configuration entry, they will be there when the value is
retrieved in the program.

Configuration items can have their values replaced or modified by
subsequent configuration statements, unless the read-only flag is
specified (see below).

The value component may also contain macro references that are
replaced with other text as the config entry is processed. Macros can
be used to dynamically adapt a config entry to its operating
environment without requiring the entry to be hand edited. The macro
substitution feature is described below.

Configuration entry attributes
''''''''''''''''''''''''''''''

Configuration keys may have attributes associated with them. These
attributes are specified immediately after the configuration key. All
attributes are enclosed in a single set of brackets (e.g. []). If a
configuration key has more than one attribute they are all in the same
set of brackets separated by a comma.

Currently the only understood flags are:

``flag{ro}`` Marks the configuration value as read-only. A configuration
that is marked as read only may not have the value subsequently
modified in the pipeline file or programatically by the program.

``flag{tunable}`` Marks the configuration value as tunable. A
configuration entry that is marked as tunable can have a new value
presented to the process during a reconfigure operation.

Examples::

  foo[ro] = bar # results in foo = "bar"
  foo[ro, tunable] = bar


Macro Substitution
------------------

The values for configuration elements can be composed from static text
in the config file and dynamic text supplied by macro providers. The
format of a macro specification is ``$TYPE{name}`` where **TYPE** is the
name of macro provider and **name** requests a particular value to be
supplied. The **name** entry is specific to each provider.

The text of the macro specification is only replaced. Any leading or
trailing blanks will remain.  If the value of a macro is not defined,
the macro specification will be replaced with the null string.

Macro Providers
'''''''''''''''

The macro providers are listed below and discussed in the following sections.

- LOCAL - locally defined values
- ENV - program environment
- CONFIG - values from current config block
- SYSENV - system environment

LOCAL Macro Provider
''''''''''''''''''''

This macro provider supplies values that have been stored previously
in the config file.  Local values are specified in the config file
using the ":=" operator. For example the config entry ``mode := online``
makes ``$LOCAL{mode}`` available in subsequent configuration
entries.::

  mode := online
  ...
  config_file = data/$LOCAL{mode}/model.dat


This type of macro definition can appear anywhere in a config file and
becomes available for use on the next line.  The current block context
has no effect on the name of the macro.

ENV Macro Provider
''''''''''''''''''

This macro provides gives access to the current program
environment. The values of environment variables such as "HOME" can be
used by specifying ``$ENV{HOME}`` in the config file.

CONFIG Macro Provider
'''''''''''''''''''''

This macro provider gives access to previously defined configuration entries.
For example::

  config foo
    bar = baz

makes the value available by specifying ``$CONFIG{foo:bar}`` to following lines in the config file
as shown below.::

   value = mode-$CONFIG{foo:bar}ify


SYSENV Macro Provider
'''''''''''''''''''''

This macro provider supports the following symbols derived from the
current host operating system environment.

- curdir - current working directory
- homedir - current user's home directory
- pid - current process id
- numproc - number of processors in the current system
- totalvirtualmemory - number of KB of total virtual memory
- availablevirtualmemory - number of KB of available virtual memory
- totalphysicalmemory - number of KB of total physical memory
- availablephysicalmemory - number of KB of physical virtual memory
- hostname - name of the host computer
- domainname - name of the computer in the domain
- osname - name of the host operating system
- osdescription - description of the host operating system
- osplatform - platorm name (e.g. x86-64)
- osversion - version number for the host operating system
- iswindows - TRUE if running on Windows system
- islinux - TRUE if running on Linux system
- isapple - TRUE if running on Apple system
- is64bits - TRUE if running on a 64 bit machine

Block Specification
-------------------

In some cases the fully qualified configuration key can become long and unwieldy.
The block directive can be used to establish a configuration context to be applied
to the enclosed configuration entries.
``block alg``
Starts a block with the *alg* block name and all entries within the block will have ``alg:``
prepended to the entry name.::

  block alg
     mode = red      # becomes alg:mode = red
  endblock

Blocks can be nested to an arbitrary depth with each providing context for the enclosed
entries.::

  block foo
    block bar:fizzle
      mode = yellow     # becomes foo:bar:fizzle:mode = yellow
    endblock
  endblock

Including Files
---------------

The include directive logically inserts the contents of the specified
file into the current file at the point of the include
directive. Include files provide an easy way to break up large
configurations into smaller reusable pieces.

``include filename``

The ``filename`` specified may contain references to an ENV or SYSENV
macro. The macro reference is expanded before the file is located. No
other macro providers are supported.

If the file name is not an absolute path, it is located by scanning
the current config search path.  The manner in which the config
include path is created is described in a following section.  If the
file is still not found, the stack of include directories is scanned
from the current include file back to the initial config file. Macro
substitution, as described below, is performed on the file name string
before the searching is done.

Block specifications and include directives can be used together to
build reusable and shareable configuration snippets.::

  block main
    block alg_one
      include alg_foo.config
    endblock

    block alg_two
      include alg_foo.config
    endblock
  endblock

In this case the same configuration structure can be used in two
places in the overall configuration.

Include files can be nested to an arbitrary depth.

Relativepath Modifier
---------------------

There are cases where an algorithm needs an external file containing
binary data that is tied to a specific configuration.  These data
files are usually stored with the main configuration files.
Specifying a full hard coded file path is not portable between
different users and systems.

The solution is to specify the location of these external files
relative to the configuration file and use the *relativepath* modifier
construct a full, absolute path at run time by prepending the
configuration file directory path to the value. The relativepath keyword
appears before the *key* component of a configuration entry.::

  relativepath data_file = ../data/online_dat.dat

If the current configuration file is
``/home/vital/project/config/blue/foo.config``, the resulting config
entry for **data_file** will be
``/home/vital/project/config/blue/../data/online.dat``

The *relativepath* modifier can be applied to any configuration entry,
but it only makes sense to use it with relative file specifications.

Configuration Section
---------------------

Configuration sections introduce a named configuration subblock that
can provide configuration entries to runtime components or make the
entries available through the $CONFIG{key} macro.

The configuration blocks for *_pipeline* and *_scheduler* are
described below.

The form of a configuration section is as follows::

  config <key-path> <line-end>
        <config entries>

Examples
''''''''
todo Explain examples.::

  config common
    uncommon = value
    also:uncommon = value


Creates configuration items::

    common:uncommon = value
    common:also:uncommon = value


Another example::

  config a:common:path
    uncommon:path:to:key = value
    other:uncommon:path:to:key = value

Creates configuration items::

    a:common:path:uncommon:path:to:key = value
    a:common:path:other:uncommon:path:to:key = value

Process definition Section
--------------------------

A process block adds a process to the pipeline with optional
configuration items. Processes are added as an instance of registered
process type with the specified name. Optional configuration entries
can follow the process declaration. These configuration entries are
made available to that process when it is started.

Specification
'''''''''''''
A process specification is as follows. An instance of the specified process-type
is created and is available in the pipeline under the specified process-name::

  process <process-name> :: <process-type>
    <config entries>

Examples
''''''''

An instance of my_processes_type is created and named my_process::

  process my_process :: my_process_type

  process another_process
    :: awesome_process
       some_param = some_value


Non-blocking processes
''''''''''''''''''''''
A process can be declared as non-blocking which indicates that input
data is to be dropped if the input port queues are full. This is
useful for real-time processing where a process is the bottleneck.

The non-blocking behaviour is a process attribute that is specified as
a configuration entryin the pipeline file. The syntax for this
configuration option is as follows::

  process blocking_process
    :: awesome_process
     _non_blocking = 2

The special "_non_blocking" configuration entry specifies the
capacity of all incoming edges to the process. When the edges are
full, the input data are dropped. The input edge size is set to two
entries in the above example. This capacity specification overrides
all other edge capacity controls for this process only.

Static port values
''''''''''''''''''

Declaring a port static allows a port to be supplied a constant value
from the config in addition to the option of it being connected in the
normal way. Ports are declared static when they are created by a
process by adding the \c flag_input_static option to the \c
declare_input_port() method.

When a port is declared as static, the value at this port may be
supplied via the configuration using the special static/ prefix
before the port name. The syntax for specifying static values is::

 :static/<port-name> <key-value>

If a port is connected and also has a static value configured, the
configured static value is ignored.

The following is an example of configuring a static port value.::

  process my_process
    :: my_process_type
       static/port = value

Instrumenting Processes
'''''''''''''''''''''''

A process may request to have its instrumentation calls handled by an external provider. This
is done by adding the _instrumentation block to the process config.::

  process my_process
    :: my_process_type
    block _instrumentation
       type = foo
       block  foo
         file = output.dat
         buffering = optimal
       endblock
    endblock


The type parameter specifies the instrumentation provider, "foo" in
this case. If the special name "none" is specified, then no
instrumentation provider is loaded. This is the same as not having the
config block present. The remaining configuration items that start
with "_instrumentation:<type>" are considered configuration data for
the provider and are passed to the provider after it is loaded.

Connection Definition
---------------------

A connection definition specifies how the output ports from a process
are connected to the input ports of another process. These connections
define the data flow of the pipeline graph.::


  connect from <process-name> . <input-port-name> to <process-name> . <output-port-name>


Examples
''''''''

This example connects a timestamp port to two different processes.::

 connect from input.timestamp      to   stabilize  .timestamp
 connect from input.timestamp      to   writer     .timestamp


Pipeline Edge Configuration
---------------------------

A pipeline edge is a connection between two ports. The behaviour of
the edges can be configured if the defaults are not appropriate.  Note
that defining a process as non-blocking overrides all input edge
configurations for that process only.

Pipeline edges are configured in a hierarchical manner. First there is
the _pipeline:_edge config block which establishes the basic
configuration for all edges. This can be specified as follows::

  config _pipeline:_edge
         capacity = 30     # set default edge capacity


Currently the only attribute that can be configured is "capacity".

The config for the edge type overrides the default configuration so
that edges used to transport specific data types can be configured as
a group. This edge type configuration is specified as follows::

  config _pipeline:_edge_by_type
         image_container:capacity = 30
         timestamp:capacity = 4


Where *image_container* and  *timestamp* are the type names used when
defining process ports.

After this set of configurations have been applied, edges can be
more specifically configured based on their connection description. An
edge connection is described in the config as follows::

  config _pipeline:_edge_by_conn
          <process>:<up_down>:<port> <value>


Where:

- <process> is the name of the process that is being connected.
- <up_down> is the direction of the connection. This is either "up" or "down".
- <port> is the name of the port.

For the example, the following connection::

  connect from input.timestamp
          to   stabilize.timestamp


can be described as follows::

  config _pipeline:_edge_by_conn
     input:up:timestamp:capacity = 20
     stabilize:down:timestamp:capacity = 20


Both of these entries refer to the same edge, so in real life, you
would only need one.

These different methods of configuring pipeline edges are applied
in a hierarchial manner to allow general defaults to be set, and
overridden using more specific edge attributes. This order is
default capacity, edge by type, then edge by connection.

Scheduler configuration
-----------------------

Normally the pipeline is run with a default scheduler that assigns
one thread to each process. A different scheduler can be specified
in the config file. Configuration parameters for the scheduler can
be specified in this section also.::

  config _scheduler
     type = <scheduler-type>


Available scheduler types are:

- sync - Runs the pipeline synchronously in one thread.
- thread_per_process - Runs the pipeline using one thread per process.
- pythread_per_process - Runs the pipeline using one thread per process and supports processes written in python.
- thread_pool - Runs pipeline with a limited number of threads (not implemented).

The pythread_per_process is the only scheduler that supports processes written python.

Scheduler specific configuration entries are in a sub-block named as
the scheduler.  Currently these schedulers do not have any
configuration parameters, but when they do, they would be configured
as shown in the following example.

Example
'''''''

The pipeline scheduler can selected with the pipeline configuration as follows::

  config _scheduler
   type = thread_per_process

   # Configuration for thread_per_process scheduler
   thread_per_process:foo = bar

   # Configuration for sync scheduler
   sync:foos = bars


Clusters Definition File
------------------------

A cluster is a collection of processes which can be treated as a
single process for connection and configuration purposes. Clusters are
defined in a slngle file with one cluster per file.

A cluster definition starts with the *cluster* keyword followed by
the name of the cluster. A documentation section must follow the
cluster name definition. Here is where you describe the purpose and
function of the cluster in addition to any other important
information about limitations or assumptions. Comments start
with ``--`` and continue to the end of the line. These comments
are included with the cluster definition and are displayed by the
plugin explorer as part of the cluster documentation. The '#' style
comments can still be used to annotate the file but are not included
as part of the cluster documnetation.

The body of the cluster definition is made up of three types of
declarations that may appear multiple times and in any order. These
are:

  - config specifier
  - input mapping
  - output mapping

A description is required after each one of these entries. The
description starts with "--" and continues to the end of the
line. These descriptions are different from typical comments you would
put in a pipe file in that they are associated with the cluster
elements and serve as user documentation for the cluster.

After the cluster has been defined, the constituent processes are
defined. These processes are contained within the cluster and can be
interconnected in any valid configuration.

config specifier
''''''''''''''''

A configuration specification defines a configuration key with a value
that is bound to the cluster. These configuration items are available
for use within the cluster definition file and are referenced as
<cluster-name>:<config-key>::

     cluster_key = value
     -- Describe configuration entry


Input mapping
'''''''''''''

The input mapping specification creates an input port on the cluster
and defines how it is connected to a process (or processes) within the
cluster. When a cluster is instantiated in a pipeline, connections can
be made to these ports.::

    imap from cport
         to   proc1.port
         to   proc2.port
    -- Describe input port expected data type and
    -- all other interesting details.


Output mapping
''''''''''''''

The output mappinc specification creates an output port on the cluster
and defines how the data is supplied. When a cluster is instantiated,
these output ports can be connected to downstream processes in the
usual manner.::

    omap from proc2.oport   to  cport
    -- Describe output port data type and
    -- all other interesting details.


An example cluster definition is as follows::

  cluster <name>
    -- Description fo cluster.
    -- May extend to multiple lines.

    cluster_key = value
    -- Describe the config entry here.

    imap from cport
         to   proc1.port
         to   proc2.port
    -- Describe input port. Input port can be mapped
    -- to multiple process ports

    omap from proc2.oport    to  coport
    -- describe output port

The following is a more complicated example::

  cluster configuration_provide
    -- Multiply a number by a constant factor.

    factor = 20
    -- The constant factor to multiply by.

    imap from factor  to   multiply.factor1
    -- The factor to multiply by.

    omap from multiply.product    to   product
    -- The product.

   # The following defines the contained processes
  process const
    :: const_number
    value[ro]= $CONFIG{configuration_provide:factor}

  process multiply
    :: multiplication

  connect from const.number        to   multiply.factor2
