How To Make a Process
=====================

.. \page how_to_process Creating a process
..
.. The \ref sprokit::process class is the basic building block of a pipeline. In
.. general, there are three kinds of processes:
..
.. \li \b source — Reads data from a file, creates data, or otherwise acts as a
..     way for data to enter the pipeline.
.. \li \b compute — Uses input data to perform some computation or
..     transformation on the data and then outputs it.
.. \li \b sink — Accepts data and writes it to a file, verifies it, or some
..     other task which does not have any (pipeline) output.
..
.. This page will go through the steps of creating a compute process and will
.. include notes for each of the general types where things may differ. In this
.. example, a process which compares two input strings will be created.
..
.. \section how_to_process_starting Getting started
..
.. \dontinclude examples/how_to_process.cxx.ex
..
.. First up is to declare your class:
..
.. \until };
..
.. Processes must inherit from \ref sprokit::process and the constructor must be
.. able to pass the configuration to it. A destructor is declared for
.. completion's sake.
..
.. The process needs to reimplement the \ref sprokit::process::_step method to be
.. able to do useful transformations.
..
.. \note Sink processes will usually not have any output ports and can ignore
.. those methods.
..
.. In this example, the process will have a private class in which to store
.. information. A private class does not need to be used, but in order to keep
.. things clean, the example will use it.
..
.. \until };
..
.. Here, the \tt{compare_string_process::priv} class is declared with the
.. variables that will be needed later. There is a variable to store the
.. configuration value in, a declaration of the name of the configuration
.. variable name, and a default for it.
..
.. \until port_output
..
.. Definitions of defaults and names for ports and configuration values. This
.. helps to reduce spelling errors (although there are test cases to ensure that
.. introspection on processes is valid).
..
.. \note In general, naming things with a leading underscore is highly
.. discouraged. A leading underscore is used for names used internally to help
.. avoid future conflicts.
..
.. \section how_to_process_config Configuration
..
.. The first step in creating a process is the configuration. Processes will
.. receive its configuration in the constructor. Processes should not check the
.. configuration for validity at this point. The rationale is that processes
.. support introspection and a process which cannot be constructed cannot be
.. inspected.
..
.. In the example process, a configuration value that may be wanted would be a
.. flag for case insensitivity. To support introspection, information about the
.. configuration must also be provided.
..
.. Here, the configuration name and defaults are defined. The key is wrapped in
.. the \ref sprokit::config::key_t constructor to future-proof the code.
..
.. \until ));
..
.. Processes also need to offer introspection for the configuration values. In
.. order to do this, the \ref sprokit::process::_available_config and \ref
.. sprokit::process::_config_info methods need to be reimplemented.
..
.. Subclasses may declare configuration keys for management by the base class.
.. This is done with the \ref sprokit::process::declare_configuration_key method.
..
.. \section how_to_process_ports Ports
..
.. Ports are where edges are connected to a process for it to be able to send
.. and receive data.
..
.. Input ports are where a process receives data and output ports are where data
.. should be output. Input and output do not need to be 1:1, but in most cases
.. they will be.
..
.. Data packets along an edge contain both a stamp and the actual data packet.
.. Stamps are used for synchronization within the pipeline and the data packet
.. is a way to pass status messages between pipeline nodes.
..
.. Stamps can be managed manually by a process (those that upsample or
.. downsample would need to do this, input processes will also generally use
.. their own stamping logic), but most can reply on the step-counting
.. management of the base class.
..
.. Data packets can indicate a few status messages for the upstream port (\em
.. not necessarily the process, as the status message is per port). In order
.. from highest to lowest priority (i.e., if it occurs within a group of
.. packets, it takes precedence for determining the group's status) are:
..
.. \li \ref sprokit::datum::invalid — An invalid type. This should never occur.
.. \li \ref sprokit::datum::error — An error occurred computing the expected data.
.. \li \ref sprokit::datum::flush — Indicates that the data stream is being
..     "reset" and any internal state within the process should be cleared. This
..     should only affect processes which keep state in between steps.
.. \li \ref sprokit::datum::complete — Indicates that no more data will be
..     available on the port. This should be used for the end of a file, data
..     stream, or that required ports are complete as well and nothing more can
..     be computed.
.. \li \ref sprokit::datum::empty — No data available. The value was not
..     computable or a recoverable error occurred which prevented computation.
.. \li \ref sprokit::datum::data — Contains actual data.
..
.. The \ref sprokit::process::data_info method can be used to determine the
.. highest priority status within a group of packets.
..
.. \until insert
..
.. The \ref sprokit::process::flag_required flag indicates that the port \em must
.. be connected for the process to run. If there is any configuration where a
.. port can be disconnected and the process still successfully run, it should be
.. checked during initialization.
..
.. The other flags which are currently understood are:
..
.. \li \tt{flag_input_mutable} — Indicates that the input received on the port
..     will be modified externally by another process.
.. \li \tt{flag_output_const} — Indicates that the data sent on the output port
..     may not be modified. This is usually used for data that is sent, but a
..     reference is kept internally to help reduce memory usage.
..
.. \until }
..
.. When the class is initialized, it declares its ports for the base class to
.. manage. The types on the ports should be chosen carefully. If the strings do
.. not compare exactly, the \ref sprokit::pipeline::connect method will fail when
.. connecting two ports. There are also some types which have special meaning:
..
.. \li \ref sprokit::process::type_any — Indicates that the data on the port is
..     not inspected. Processes which do tasks such as load balancing and
..     joining should use this type on its ports.
.. \li \ref sprokit::process::type_none — Indicates that the data on the port is
..     never created with \ref sprokit::datum::new_datum and is only useful for
..     triggers and status messages.
..
.. \until }
..
.. \todo Need use cases for ports described. Describe why and when you want these
.. types of ports. How they interact with scheduler stopping criteria.
..
.. - required input port
.. - optional input port
.. - optional, static input port. How to specify static data for this port in the config.
.. - optional output port
.. - required output port
..
.. - how to determine if optional input is connected and why you would want to check.
..
.. The private class constructor is straightforward.
..
.. \section how_to_process_init Initialization
..
.. \until }
.. \until }
..
.. When initializing \tt{compare_string_process}, the configuration for the
.. process is read and the private class is initialized. This is safe because
.. the \ref sprokit::process class will ensure that \ref sprokit::process::step
.. cannot be called before \ref sprokit::process::init.
..
.. The process can check to see if its configuration is valid. If there is some
.. invalid value or a logic error, the \ref
.. sprokit::invalid_configuration_value_exception and \ref
.. sprokit::invalid_configuration_exception exceptions can be thrown,
.. respectively.
..
.. Initialization is where files should be opened (for reading or writing),
.. bounds checked, and so on.
..
.. \section how_to_process_run Running
..
.. The methods of a process are called in the following order:
..
.. Constructor()
.. configure()
.. all pipeline connections are made
.. init
..
.. The base \ref sprokit::process class will do checking for input ports that have
.. the \flag{required} flag to ensure that they are synchronized and all have
..
..
..
.. data available on them. The default behavior is meant to handle the common
.. case for processes that receive inputs and compute a single result based on
.. it. The default behavior can be enabled or disabled for each case using the
.. \c ensure_* methods on the \ref sprokit::process class.
..
.. \until include
..
.. The process will use Boost to compare strings and ignoring case.
..
.. \until str2
..
.. First, data is grabbed from the input ports.
..
.. \until }
..
.. The case insensitive comparison is only made if the case sensitive compare
.. failed and the option for it is set.
..
.. \until push_to_port_as
..
.. The result is pushed into the output port.
..
.. \until }
..
.. Finally, any post-step processing that the base class may need to handle is
.. called.
..
.. \section how_to_process_termination Termination
..
.. \until }
.. \until }
..
.. The process is terminated when the destructor is called. It may be called at
.. any time (i.e., before the \ref sprokit::process::mark_process_as_complete
.. method is called) if the pipeline is terminated before completion.
..
.. \section how_to_process_registration Registration
..
.. Processes are registered with the process registry so that they can be
.. dynamically created by other tools. Modules are found by the \ref
.. sprokit::load_known_modules function which loads up dynamic libraries and calls
.. a registration function. The module must have a symbol by the name of
.. \tt{register_processes} which is called to register process types.
..
.. In \tt{register_processes}, a function is given to the registry along with a
.. typename for the process and a short description. There is a template
.. function which does the default creation of a process, \ref
.. sprokit::create_process.
..
.. \until }
.. \until }
..
.. The \tt{register_processes} function then adds a new type to the registry and
.. gives the function to create a new instance but only if this registration
.. function has not already been run.

