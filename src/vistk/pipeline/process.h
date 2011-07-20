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
#include "process_registry.h"
#include "types.h"

#include <boost/utility.hpp>

#include <string>
#include <vector>

/**
 * \file process.h
 *
 * \brief Header for \link process processes\endlink.
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
 * \ingroup base_classes
 */
class VISTK_PIPELINE_EXPORT process
  : boost::noncopyable
{
  public:
    /// The type for the name of a process.
    typedef std::string name_t;
    /// The type for the name of a port on a process.
    typedef std::string port_t;
    /// A group of ports.
    typedef std::vector<port_t> ports_t;

    /**
     * \brief Post-connection initialization.
     */
    void init();

    /**
     * \brief Steps through one iteration of the process.
     */
    void step();

    /**
     * \brief Connects an edge to an input port on the process.
     *
     * \param port The port to connect to.
     * \param edge The edge to connect to the port.
     */
    void connect_input_port(port_t const& port, edge_t edge);
    /**
     * \brief Connects an edge to an output port on the process.
     *
     * \throws null_edge_port_connection Thrown when \p edge is \c NULL.
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
     * \brief Request available configuration options for the process.
     *
     * \returns The names of all available configuration keys.
     */
    virtual config::keys_t available_config() const = 0;
    /**
     * \brief Request the default value for a configuration.
     *
     * \throws unknown_configuration_value Thrown when \p key is not a valid configuration key.
     *
     * \param key The name of the configuration value.
     *
     * \returns The default value for \p key.
     */
    virtual config::value_t config_default(config::key_t const& key) const;
    /**
     * \brief Request available configuration options for the process.
     *
     * \throws unknown_configuration_value Thrown when \p key is not a valid configuration key.
     *
     * \param key The name of the configuration value to describe.
     *
     * \returns A description of the value expected for \p key.
     */
    virtual config::description_t config_description(config::key_t const& key) const;

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
    virtual process_registry::type_t type() const = 0;
  protected:
    /**
     * \brief Constructor.
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
    virtual void _connect_input_port(port_t const& port, edge_t edge);
    /**
     * \brief Subclass output connection method.
     *
     * \param port The port to connect to.
     * \param edge The edge to connect to the port.
     */
    virtual void _connect_output_port(port_t const& port, edge_t edge);

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
     * \brief Check if a set of edges carry the same colored data.
     *
     * \param edges The edges to check the color of.
     *
     * \returns True if the available data in each of \p edges have the same coloring, false otherwise.
     */
    static bool same_colored_edges(edges_t const& edges);
    /**
     * \brief Check if a set of edges are syncronized.
     *
     * Makes sure that the given edges all have data that carry equivalent
     * stamps.
     *
     * \param edges The edges to check.
     *
     * \returns True if the available data in each of \p edges have equivalent stamps, false otherwise.
     */
    static bool sync_edges(edges_t const& edges);
  private:
    class priv;
    boost::shared_ptr<priv> d;
};

} // end namespace vistk

#endif // VISTK_PIPELINE_PROCESS_H
