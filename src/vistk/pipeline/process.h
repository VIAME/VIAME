/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_PROCESS_H
#define VISTK_PIPELINE_PROCESS_H

#include "pipeline-config.h"

#include "edge.h"
#include "config.h"
#include "datum.h"
#include "process_registry.h"
#include "types.h"

#include <boost/utility.hpp>

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
 * \oport{heartbeat} Carries the status of the process.
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
    /// The type for the name of a process.
    typedef std::string name_t;
    /// A group of process names.
    typedef std::vector<name_t> names_t;
    /// A type for a description of a port.
    typedef std::string port_description_t;
    /// The type for the name of a port on a process.
    typedef std::string port_t;
    /// A group of ports.
    typedef std::vector<port_t> ports_t;
    /// The type for the type of data on a port.
    typedef std::string port_type_t;
    /// The type for a flag on a port.
    typedef std::string port_flag_t;
    /// A group of port flags.
    typedef std::set<port_flag_t> port_flags_t;
    /// Type for the address of a port within the pipeline.
    typedef std::pair<name_t, port_t> port_addr_t;
    /// A group of port addresses.
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
         */
        port_info(port_type_t const& type_,
                  port_flags_t const& flags_,
                  port_description_t const& description_);
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
     * \brief Post-connection initialization.
     */
    void init();

    /**
     * \brief Steps through one iteration of the process.
     */
    void step();

    /**
     * \brief Query if a process is reentrant.
     *
     * \warning The base class is not yet reentrant.
     *
     * If \c true, it indicates that the \ref process::_step() method is
     * reentrant and does data collation of its edges properly. The default
     * implementation returns \c false for safety.
     *
     * \warning Please be sure that your process is \em actually reentrant
     * before returning \c true from this method.
     *
     * \returns True if the process can be \ref process::step()'d recursively, false otherwise.
     */
    virtual bool is_reentrant() const;

    /**
     * \brief Connects an edge to an input port on the process.
     *
     * \throws null_edge_port_connection_exception Thrown when \p edge is \c NULL.
     * \throws no_such_port_exception_exception Thrown when \p port does not exist on the process.
     *
     * \param port The port to connect to.
     * \param edge The edge to connect to the port.
     */
    void connect_input_port(port_t const& port, edge_t edge);
    /**
     * \brief Connects an edge to an output port on the process.
     *
     * \throws null_edge_port_connection_exception Thrown when \p edge is \c NULL.
     * \throws no_such_port_exception_exception Thrown when \p port does not exist on the process.
     *
     * \param port The port to connect to.
     * \param edge The edge to connect to the port.
     */
    void connect_output_port(port_t const& port, edge_t edge);

    /**
     * \brief A list of input ports available on the process.
     *
     * \returns The names of all input ports available.
     */
    ports_t input_ports() const;
    /**
     * \brief A list of output ports available on the process.
     *
     * \returns The names of all output ports available.
     */
    ports_t output_ports() const;

    /**
     * \brief Information about an input port on the process.
     *
     * \throws no_such_port_exception_exception Thrown when \p port does not exist on the process.
     *
     * \param port The port to return information about.
     *
     * \returns Information about the input port.
     */
    port_info_t input_port_info(port_t const& port) const;
    /**
     * \brief Information about an output port on the process.
     *
     * \throws no_such_port_exception_exception Thrown when \p port does not exist on the process.
     *
     * \param port The port to return information about.
     *
     * \returns Information about the output port.
     */
    port_info_t output_port_info(port_t const& port) const;

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
    conf_info_t config_info(config::key_t const& key) const;

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
    process_registry::type_t type() const;

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
    /// A flag which indicates that the output cannot be modified.
    static port_flag_t const flag_output_const;
    /// A flag which indicates that the input may be modified.
    static port_flag_t const flag_input_mutable;
    /// A flag which indicates that port is required to be connected.
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
     * \brief Initialization checks for subclasses.
     */
    virtual void _init();

    /**
     * \brief Method where subclass data processing occurs.
     */
    virtual void _step();

    /**
     * \brief Subclass input connection method.
     *
     * \param port The port to connect to.
     * \param edge The edge to connect to the port.
     */
    virtual void _connect_input_port(port_t const& port, edge_ref_t edge);
    /**
     * \brief Subclass output connection method.
     *
     * \param port The port to connect to.
     * \param edge The edge to connect to the port.
     */
    virtual void _connect_output_port(port_t const& port, edge_ref_t edge);

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
    virtual port_info_t _input_port_info(port_t const& port) const;
    /**
     * \brief Subclass output port information.
     *
     * \param port The port to get information about.
     *
     * \returns Information about an output port.
     */
    virtual port_info_t _output_port_info(port_t const& port) const;

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
    virtual conf_info_t _config_info(config::key_t const& key) const;

    /**
     * \brief Marks the process as complete.
     */
    void mark_as_complete();
    /**
     * \brief The \ref stamp that the hearbeat is based off of.
     *
     * \returns The stamp that the heartbeat uses.
     */
    stamp_t heartbeat_stamp() const;

    /**
     * \brief Check if a set of data are colored the same.
     *
     * \param data The data to check.
     *
     * \returns True if each datum has the same coloring, false otherwise.
     */
    static bool same_colored_data(edge_data_t const& data);
    /**
     * \brief Check if a set of data is syncronized.
     *
     * Makes sure that the given data all have equivalent stamps.
     *
     * \param data The data to check.
     *
     * \returns True if the available data have equivalent stamps, false otherwise.
     */
    static bool syncd_data(edge_data_t const& data);
    /**
     * \brief Examine datum packets to determine the highest priority status.
     *
     * \param data The data to look at.
     *
     * \returns The highest priority type present in the data.
     */
    static datum::datum_type_t max_status(edge_data_t const& data);
    /**
     * \brief Pushes data to all given edges.
     *
     * \param edges The edges to push to.
     * \param dat The data to push.
     */
    static void push_to_edges(edge_group_t const& edges, edge_datum_t const& dat);
    /**
     * \brief Grabs a data from an edge.
     *
     * \param edge The edge to grab data from.
     *
     * \returns The next datum from the edge.
     */
    static edge_datum_t grab_from_edge_ref(edge_ref_t const& edge);
  private:
    class priv;
    boost::shared_ptr<priv> d;
};

}

#endif // VISTK_PIPELINE_PROCESS_H
