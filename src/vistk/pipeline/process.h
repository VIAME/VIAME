/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_PROCESS_H
#define VISTK_PIPELINE_PROCESS_H

#include "pipeline-config.h"

#include "edge.h"
#include "config.h"
#include "datum.h"
#include "types.h"

#include <boost/cstdint.hpp>
#include <boost/noncopyable.hpp>
#include <boost/rational.hpp>
#include <boost/scoped_ptr.hpp>

#include <set>
#include <string>
#include <utility>
#include <vector>

/**
 * \file process.h
 *
 * \brief Header for \link vistk::process processes\endlink.
 */

namespace vistk
{

/// A group of processes.
typedef std::vector<process_t> processes_t;

/**
 * \class process process.h <vistk/pipeline/process.h>
 *
 * \brief A node within a \ref pipeline which runs computations on data.
 *
 * \oports
 *
 * \oport{_heartbeat} Carries the status of the process.
 *
 * \section initialization Initialization Routine
 *
 * <ol>
 *   <li>Configuration is given to the process when constructed.</li>
 *   <li>Connections are made.</li>
 *   <li>The \ref process::_init() method is called.</li>
 * </ol>
 *
 * Exceptions for misconfiguration should be thrown from \ref process::_init()
 * reimplementations. This is to facilitate querying of processes.
 *
 * \section destruction Destruction Routine
 *
 * <ol>
 *   <li>The destructor is called.</li>
 * </ol>
 *
 * Processes should be ready for destruction at any time.
 *
 * \ingroup base_classes
 */
class VISTK_PIPELINE_EXPORT process
  : boost::noncopyable
{
  public:
    /// The type for the type of a process.
    typedef std::string type_t;
    /// A group of types.
    typedef std::vector<type_t> types_t;
    /// The type for the name of a process.
    typedef std::string name_t;
    /// The type for a group of process names.
    typedef std::vector<name_t> names_t;
    /// The type for a constraint on a process.
    typedef std::string constraint_t;
    /// The type for a set of constraints on a process.
    typedef std::set<constraint_t> constraints_t;
    /// The type for a description of a port.
    typedef std::string port_description_t;
    /// The type for the name of a port on a process.
    typedef std::string port_t;
    /// The type for a group of ports.
    typedef std::vector<port_t> ports_t;
    /// The type for the type of data on a port.
    typedef std::string port_type_t;
    /// The type for the component of a frequency.
    typedef uint64_t frequency_component_t;
    /// The type for the frequency of data on a port.
    typedef boost::rational<frequency_component_t> port_frequency_t;
    /// The type for a flag on a port.
    typedef std::string port_flag_t;
    /// The type for a group of port flags.
    typedef std::set<port_flag_t> port_flags_t;
    /// The type for the address of a port within the pipeline.
    typedef std::pair<name_t, port_t> port_addr_t;
    /// The type for a group of port addresses.
    typedef std::vector<port_addr_t> port_addrs_t;

    /**
     * \class port_info process.h <vistk/pipeline/process.h>
     *
     * \brief Information about a port.
     */
    class VISTK_PIPELINE_EXPORT port_info
    {
      public:
        /**
         * \brief Constructor.
         *
         * \param type_ The type of the port.
         * \param flags_ Flags for the port.
         * \param description_ A description of the port.
         * \param frequency_ The frequency of the port relative to the step.
         */
        port_info(port_type_t const& type_,
                  port_flags_t const& flags_,
                  port_description_t const& description_,
                  port_frequency_t const& frequency_);
        /**
         * \brief Destructor.
         */
        ~port_info();

        /// The type of the port.
        port_type_t const type;
        /// Flags for the port.
        port_flags_t const flags;
        /// A description of the port.
        port_description_t const description;
        /// The port's frequency.
        port_frequency_t const frequency;
    };
    /// Type for information about a port.
    typedef boost::shared_ptr<port_info const> port_info_t;

    /**
     * \class conf_info process.h <vistk/pipeline/process.h>
     *
     * \brief Information about a configuration parameter.
     */
    class VISTK_PIPELINE_EXPORT conf_info
    {
      public:
        /**
         * \brief Constructor.
         *
         * \param def_ The default value for the parameter.
         * \param description_ A description of the value.
         */
        conf_info(config::value_t const& def_,
                  config::description_t const& description_);
        /**
         * \brief Destructor.
         */
        ~conf_info();

        /// The default value for the parameter.
        config::value_t const def;
        /// A description of the value.
        config::description_t const description;
    };
    /// Type for information about a configuration parameter.
    typedef boost::shared_ptr<conf_info const> conf_info_t;

    /**
     * \class data_info process.h <vistk/pipeline/process.h>
     *
     * \brief Information about a set of data.
     */
    class VISTK_PIPELINE_EXPORT data_info
    {
      public:
        /**
         * \brief Constructor.
         *
         * \param in_sync_ Whether the data is synchonized.
         * \param max_status_ The highest priority status of the data.
         */
        data_info(bool in_sync_,
                  datum::type_t max_status_);
        /**
         * \brief Destructor.
         */
        ~data_info();

        /// True if the data is synchonized.
        bool const in_sync;
        /// The highest priority status in the set.
        datum::type_t const max_status;
    };
    /// Type for information about a set of data.
    typedef boost::shared_ptr<data_info const> data_info_t;

    /**
     * \brief Data checking levels. All levels include lower levels.
     *
     * \note This is only exposed for easier access from bindings.
     *
     * All levels include lower levels.
     */
    typedef enum
    {
      /// Check nothing about incoming data.
      check_none,
      /// Check to ensure incoming data is synchronized.
      check_sync,
      /// Check to ensure incoming data is valid.
      check_valid
    } data_check_t;

    /**
     * \brief Pre-connection initialization.
     *
     * \throws reconfigured_exception Thrown if called multiple times.
     *
     * \postconds
     *
     * \postcond{\c this is ready to be initialized}
     *
     * \endpostconds
     */
    void configure();

    /**
     * \brief Post-connection initialization.
     *
     * \throws unconfigured_exception Thrown if called before \ref configure.
     * \throws reinitialization_exception Thrown if called multiple times.
     *
     * \postconds
     *
     * \postcond{\c this is ready to be stepped}
     *
     * \endpostconds
     */
    void init();

    /**
     * \brief Reset the process.
     *
     * Calling this removes all edges from the process.
     */
    void reset();

    /**
     * \brief Step through one iteration of the process.
     *
     * \preconds
     *
     * \precond{\c this was initialized}
     *
     * \endpreconds
     *
     * \throws unconfigured_exception Thrown if called before \ref configure.
     * \throws uninitialized_exception Thrown if called before \ref init.
     */
    void step();

    /**
     * \brief Query for the constraints on the process.
     *
     * \returns The set of constraints on the process.
     */
    virtual constraints_t constraints() const;

    /**
     * \brief Connect an edge to an input port on the process.
     *
     * \preconds
     *
     * \precond{\p edge}
     * \precond{The input port \p port exists}
     * \precond{The input port \p port has not been connected to before}
     *
     * \endpreconds
     *
     * \throws null_edge_port_connection_exception Thrown when \p edge is \c NULL.
     * \throws connect_to_initialized_process_exception Thrown if called after \ref init.
     * \throws no_such_port_exception Thrown when \p port does not exist on the process.
     *
     * \postconds
     *
     * \postcond{The input port \p port is connected via the edge \p edge}
     *
     * \endpostconds
     *
     * \param port The port to connect to.
     * \param edge The edge to connect to the port.
     */
    void connect_input_port(port_t const& port, edge_t edge);
    /**
     * \brief Connect an edge to an output port on the process.
     *
     * \preconds
     *
     * \precond{\p edge}
     * \precond{The output port \p port exists}
     *
     * \endpreconds
     *
     * \throws null_edge_port_connection_exception Thrown when \p edge is \c NULL.
     * \throws no_such_port_exception Thrown when \p port does not exist on the process.
     *
     * \postconds
     *
     * \postcond{The input port \p port is connected via the edge \p edge}
     *
     * \endpostconds
     *
     * \param port The port to connect to.
     * \param edge The edge to connect to the port.
     */
    void connect_output_port(port_t const& port, edge_t edge);

    /**
     * \brief Query for a list of input ports available on the process.
     *
     * \returns The names of all input ports available.
     */
    ports_t input_ports() const;
    /**
     * \brief Query for a list of output ports available on the process.
     *
     * \returns The names of all output ports available.
     */
    ports_t output_ports() const;

    /**
     * \brief Query for information about an input port on the process.
     *
     * \warning The returned pointer is not updated if the information for a
     * port changes.
     *
     * \throws no_such_port_exception Thrown when \p port does not exist on the process.
     *
     * \param port The port to return information about.
     *
     * \returns Information about the input port.
     */
    port_info_t input_port_info(port_t const& port);
    /**
     * \brief Query for information about an output port on the process.
     *
     * \warning The returned pointer is not updated if the information for a
     * port changes.
     *
     * \throws no_such_port_exception Thrown when \p port does not exist on the process.
     *
     * \param port The port to return information about.
     *
     * \returns Information about the output port.
     */
    port_info_t output_port_info(port_t const& port);

    /**
     * \brief Set the type of a flow-dependent input port type.
     *
     * \throws no_such_port_exception Thrown when \p port does not exist on the process.
     * \throws static_type_reset_exception Thrown when the \p port's current type is not dependent on other types.
     * \throws set_type_on_initialized_process_exception Thrown when the \p port's type is set after initialization.
     *
     * \param port The name of the port.
     * \param new_type The type of the port.
     *
     * \returns True if the type can work, false otherwise.
     */
    bool set_input_port_type(port_t const& port, port_type_t const& new_type);
    /**
     * \brief Set the type of a flow-dependent output port type.
     *
     * \throws no_such_port_exception Thrown when \p port does not exist on the process.
     * \throws static_type_reset_exception Thrown when the \p port's current type is not dependent on other types.
     * \throws set_type_on_initialized_process_exception Thrown when the port type is set after initialization.
     *
     * \param port The name of the port.
     * \param new_type The type of the port.
     *
     * \returns True if the type can work, false otherwise.
     */
    bool set_output_port_type(port_t const& port, port_type_t const& new_type);

    /**
     * \brief Request available configuration options for the process.
     *
     * \returns The names of all available configuration keys.
     */
    config::keys_t available_config() const;

    /**
     * \brief Retrieve information about a configuration parameter.
     *
     * \throws unknown_configuration_value_exception Thrown when \p key is not a valid configuration key.
     *
     * \param key The name of the configuration value.
     *
     * \returns Information about the parameter.
     */
    conf_info_t config_info(config::key_t const& key);

    /**
     * \brief The name of the process.
     *
     * \returns The name of the process.
     */
    name_t name() const;
    /**
     * \brief The type of the process.
     *
     * \returns A name for the type of the process.
     */
    type_t type() const;

    /// A constraint which indicates that the process cannot be run in a thread of its own.
    static constraint_t const constraint_no_threads;
    /// A constraint which indicates that the process is used through the Python bindings.
    static constraint_t const constraint_python;
    /// A constraint which indicates that the process is not reentrant.
    static constraint_t const constraint_no_reentrancy;
    /// A constraint which indicates that the input of the process is not synchronized.
    static constraint_t const constraint_unsync_input;
    /// A constraint which indicates that the output of the process is not synchronized.
    static constraint_t const constraint_unsync_output;
    /// The name of the heartbeat port.
    static port_t const port_heartbeat;
    /// The name of the configuration value for the name.
    static config::key_t const config_name;
    /// The name of the configuration value for the type.
    static config::key_t const config_type;
    /// A type which means that the type of the data is irrelevant.
    static port_type_t const type_any;
    /// A type which indicates that no actual data is ever created.
    static port_type_t const type_none;
    /// A type which indicates that the type is dependent on data.
    static port_type_t const type_data_dependent;
    /// A type which indicates that the type depends on the connected port's type.
    static port_type_t const type_flow_dependent;
    /// A flag which indicates that the output cannot be modified.
    static port_flag_t const flag_output_const;
    /// A flag which indicates that the input may be defined as a configuration value.
    static port_flag_t const flag_input_static;
    /// A flag which indicates that the input may be modified.
    static port_flag_t const flag_input_mutable;
    /// A flag which indicates that a connection to the port does not imply a dependency.
    static port_flag_t const flag_input_nodep;
    /// A flag which indicates that the port is required to be connected.
    static port_flag_t const flag_required;
  protected:
    /**
     * \brief Constructor.
     *
     * \warning Configuration errors must \em not throw exceptions here.
     *
     * \param config Contains configuration for the process.
     */
    process(config_t const& config);
    /**
     * \brief Destructor.
     */
    virtual ~process();

    /**
     * \brief Pre-connection initialization for subclasses.
     */
    virtual void _configure();

    /**
     * \brief Post-connection initialization for subclasses.
     */
    virtual void _init();

    /**
     * \brief Reset logic for subclasses.
     */
    virtual void _reset();

    /**
     * \brief Method where subclass data processing occurs.
     */
    virtual void _step();

    /**
     * \brief Subclass constraint query method.
     *
     * \returns Constraints on the subclass.
     */
    virtual constraints_t _constraints() const;

    /**
     * \brief Subclass input ports.
     *
     * \returns The names of all input ports available in the subclass.
     */
    virtual ports_t _input_ports() const;
    /**
     * \brief Subclass output ports.
     *
     * \returns The names of all output ports available in the subclass.
     */
    virtual ports_t _output_ports() const;

    /**
     * \brief Subclass input port information.
     *
     * \param port The port to get information about.
     *
     * \returns Information about an output port.
     */
    virtual port_info_t _input_port_info(port_t const& port);
    /**
     * \brief Subclass output port information.
     *
     * \param port The port to get information about.
     *
     * \returns Information about an output port.
     */
    virtual port_info_t _output_port_info(port_t const& port);

    /**
     * \brief Subclass input port type setting.
     *
     * \param port The name of the port.
     * \param new_type The type of the connected port.
     *
     * \returns True if the type can work, false otherwise.
     */
    virtual bool _set_input_port_type(port_t const& port, port_type_t const& new_type);
    /**
     * \brief Subclass output port type setting.
     *
     * \param port The name of the port.
     * \param new_type The type of the connected port.
     *
     * \returns True if the type can work, false otherwise.
     */
    virtual bool _set_output_port_type(port_t const& port, port_type_t const& new_type);

    /**
     * \brief Subclass available configuration keys.
     *
     * \returns The names of all available configuration keys.
     */
    virtual config::keys_t _available_config() const;

    /**
     * \brief Subclass configuration information.
     *
     * \param key The name of the configuration value.
     *
     * \returns Information about the parameter.
     */
    virtual conf_info_t _config_info(config::key_t const& key);

    /**
     * \brief Declare an input port for the process.
     *
     * \throws null_input_port_info_exception Thrown if \p info is \c NULL.
     *
     * \param port The port name.
     * \param info Information about the port.
     */
    void declare_input_port(port_t const& port, port_info_t const& info);
    /**
     * \brief Declare an output port for the process.
     *
     * \throws null_output_port_info_exception Thrown if \p info is \c NULL.
     *
     * \param port The port name.
     * \param info Information about the port.
     */
    void declare_output_port(port_t const& port, port_info_t const& info);

    /**
     * \brief Declare an input port for the process.
     *
     * \param port The port name.
     * \param type_ The type of the port.
     * \param flags_ Flags for the port.
     * \param description_ A description of the port.
     * \param frequency_ The frequency of the port relative to the step.
     */
    void declare_input_port(port_t const& port,
                            port_type_t const& type_,
                            port_flags_t const& flags_,
                            port_description_t const& description_,
                            port_frequency_t const& frequency_ = port_frequency_t(1));
    /**
     * \brief Declare an output port for the process.
     *
     * \param port The port name.
     * \param type_ The type of the port.
     * \param flags_ Flags for the port.
     * \param description_ A description of the port.
     * \param frequency_ The frequency of the port relative to the step.
     */
    void declare_output_port(port_t const& port,
                             port_type_t const& type_,
                             port_flags_t const& flags_,
                             port_description_t const& description_,
                             port_frequency_t const& frequency_ = port_frequency_t(1));

    /**
     * \brief Set the frequency of an input port.
     *
     * \throws no_such_port_exception Thrown when \p port does not exist on the process.
     * \throws set_frequency_on_initialized_process_exception Thrown when the \p port's frequency is set after initialization.
     *
     * \param port The name of the port.
     * \param new_frequency The frequency of the port.
     */
    void set_input_port_frequency(port_t const& port, port_frequency_t const& new_frequency);
    /**
     * \brief Set the frequency of an output port.
     *
     * \throws no_such_port_exception Thrown when \p port does not exist on the process.
     * \throws set_frequency_on_initialized_process_exception Thrown when the \p port's frequency is set after initialization.
     *
     * \param port The name of the port.
     * \param new_frequency The frequency of the port.
     */
    void set_output_port_frequency(port_t const& port, port_frequency_t const& new_frequency);

    /**
     * \brief Remove an input port from the process.
     *
     * \param port The input port to remove.
     */
    void remove_input_port(port_t const& port);
    /**
     * \brief Remove an output port from the process.
     *
     * \param port The output port to remove.
     */
    void remove_output_port(port_t const& port);

    /**
     * \brief Declare a configuration value for the process.
     *
     * \throws null_conf_info_exception Thrown if \p info is \c NULL.
     *
     * \param key The configuration key.
     * \param info Information about the port.
     */
    void declare_configuration_key(config::key_t const& key, conf_info_t const& info);

    /**
     * \brief Declare a configuration value for the process.
     *
     * \param key The configuration key.
     * \param def_ The default value for the parameter.
     * \param description_ A description of the value.
     */
    void declare_configuration_key(config::key_t const& key,
                                   config::value_t const& def_,
                                   config::description_t const& description_);

    /**
     * \brief Mark the process as complete.
     */
    void mark_process_as_complete();

    /**
     * \brief Get whether there is an edge connected to an input port.
     *
     * \param port The port to get the edge for.
     *
     * \return True if there is an edge connected to the \p port, or false if there is none.
     */
    bool has_input_port_edge(port_t const& port) const;
    /**
     * \brief Get the number of connected edges for an output port.
     *
     * \param port The port to get the count for.
     *
     * \returns The number of edges connected to the \p port.
     */
    size_t count_output_port_edges(port_t const& port) const;

    /**
     * \brief Grab an edge datum packet from a port.
     *
     * \param port The port to get data from.
     *
     * \returns The datum available on the port.
     */
    edge_datum_t grab_from_port(port_t const& port) const;
    /**
     * \brief Grab a datum packet from a port.
     *
     * \param port The port to get data from.
     *
     * \returns The datum available on the port.
     */
    datum_t grab_datum_from_port(port_t const& port) const;
    /**
     * \brief Grab a datum from a port as a certain type.
     *
     * \param port The port to get data from.
     *
     * \returns The datum from the port.
     */
    template <typename T>
    T grab_from_port_as(port_t const& port) const;
    /**
     * \brief Grab an input as a certain type.
     *
     * \param port The port to get data from.
     *
     * \returns The input datum.
     */
    template <typename T>
    T grab_input_as(port_t const& port) const;
    /**
     * \brief Output an edge datum packet on a port.
     *
     * \param port The port to push to.
     * \param dat The edge datum to push.
     */
    void push_to_port(port_t const& port, edge_datum_t const& dat) const;
    /**
     * \brief Output a datum packet on a port.
     *
     * \param port The port to push to.
     * \param dat The datum to push.
     */
    void push_datum_to_port(port_t const& port, datum_t const& dat) const;
    /**
     * \brief Output a result on a port.
     *
     * \param port The port to push to.
     * \param dat The result to push.
     */
    template <typename T>
    void push_to_port_as(port_t const& port, T const& dat) const;

    /**
     * \brief The configuration for the process.
     *
     * \returns The configuration for the process.
     */
    config_t get_config() const;
    /**
     * \brief Retrieve a configuration key
     *
     * \throws no_such_configuration_key_exception Thrown if \p key was not declared for the process.
     *
     * \param key The key to request for the value.
     *
     * \returns The value of the configuration.
     */
    template <typename T>
    T config_value(config::key_t const& key) const;

    /**
     * \brief Set whether synchronization checking is enabled before stepping.
     *
     * If set to \ref check_none, no checks on incoming data is performed.
     *
     * If set to \ref check_sync, the input ports which are marked as
     * \flag{required} are guaranteed to be synchronized. When the inputs are
     * not synchronized, an error datum is pushed to all output ports and all
     * input ports will be grabbed from based on the relative frequency of the
     * ports. If this behavior is not wanted, it must be manually handled. The
     * default is that it is enabled.
     *
     * If set to \ref check_valid, the input ports which are marked as
     * \flag{required} are guaranteed to have valid data available. When the
     * inputs are not available, a default corresponding datum packet is
     * generated and pushed to all of the output edges and all input edges will
     * be grabbed from. This implies the \ref check_sync behavior as well.
     *
     * The default is \ref check_valid.
     *
     * \param check The level of validity to check incoming data for.
     */
    void set_data_checking_level(data_check_t check);

    /**
     * \brief Check a set of edge data for certain properties.
     *
     * \param data The data to inspect.
     *
     * \returns Information about the data given.
     */
    static data_info_t edge_data_info(edge_data_t const& data);
  private:
    config::value_t config_value_raw(config::key_t const& key) const;

    bool is_static_input(port_t const& port) const;
    static config::key_t const static_input_prefix;

    friend class pipeline;
    void set_core_frequency(port_frequency_t const& frequency);

    class priv;
    boost::scoped_ptr<priv> d;
};

template <typename T>
T
process
::config_value(config::key_t const& key) const
{
  return config_cast<T>(config_value_raw(key));
}

template <typename T>
T
process
::grab_from_port_as(port_t const& port) const
{
  return grab_datum_from_port(port)->get_datum<T>();
}

template <typename T>
T
process
::grab_input_as(port_t const& port) const
{
  if (is_static_input(port) && !has_input_port_edge(port))
  {
    return config_value<T>(static_input_prefix + port);
  }

  return grab_from_port_as<T>(port);
}

template <typename T>
void
process
::push_to_port_as(port_t const& port, T const& dat) const
{
  push_datum_to_port(port, datum::new_datum(dat));
}

}

#endif // VISTK_PIPELINE_PROCESS_H
