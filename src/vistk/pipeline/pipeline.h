/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_PIPELINE_H
#define VISTK_PIPELINE_PIPELINE_H

#include "pipeline-config.h"

#include "process.h"
#include "types.h"

#include <boost/utility.hpp>

#include <map>
#include <utility>

/**
 * \file pipeline.h
 *
 * \brief Header for \link pipeline pipelines\endlink.
 */

namespace vistk
{

/**
 * \class pipeline pipeline.h <vistk/pipeline/pipeline.h>
 *
 * \brief A collection of interconnected \link process processes\endlink.
 *
 * \ingroup base_classes
 */
class VISTK_PIPELINE_EXPORT pipeline
  : boost::noncopyable
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains configuration for the pipeline.
     */
    pipeline(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~pipeline();

    /**
     * \brief Add a process to the pipeline.
     *
     * \throws null_process_addition Thrown when \p process is \c NULL.
     * \throws duplicate_process_name Thrown when \p process has the same name as another process in the pipeline already.
     *
     * \param process The process to add to the pipeline.
     */
    void add_process(process_t process);
    /**
     * \brief Declares a logical group of processes in the pipeline.
     *
     * \throws duplicate_process_name Thrown when a process or group is already named \p name.
     *
     * \param name The name of the group.
     */
    void add_group(process::name_t const& name);

    /**
     * \brief Connect two ports in the pipeline together with an edge.
     *
     * \throws null_edge_connection Thrown when \p edge is \c NULL.
     * \throws no_such_process Thrown when either \p upstream_process or \p downstream_process do not exist in the pipeline.
     *
     * \param upstream_process The upstream process name.
     * \param upstream_port The upstream process port.
     * \param downstream_process The downstream process name.
     * \param downstream_port The downstream process port.
     * \param edge The edge to connect the ports with.
     */
    void connect(process::name_t const& upstream_process,
                 process::port_t const& upstream_port,
                 process::name_t const& downstream_process,
                 process::port_t const& downstream_port,
                 edge_t edge);

    /**
     * \brief Map a group input port to a process input port.
     *
     * \throws no_such_group Thrown when \p group does not exist in the pipeline.
     * \throws no_such_process Thrown when \p mapped_process does not exist in the pipeline.
     *
     * \param group The group name.
     * \param port The group port.
     * \param mapped_process The mapped process name.
     * \param mapped_port The mapped process port.
     */
    void map_input_port(process::name_t const& group,
                        process::port_t const& port,
                        process::name_t const& mapped_process,
                        process::port_t const& mapped_port);
    /**
     * \brief Map a group output port to a process output port.
     *
     * \throws no_such_group Thrown when \p group does not exist in the pipeline.
     * \throws no_such_process Thrown when \p mapped_process does not exist in the pipeline.
     * \throws group_output_already_mapped Thrown when \p port on \p group has already been mapped.
     *
     * \param group The group name.
     * \param port The group port.
     * \param mapped_process The mapped process name.
     * \param mapped_port The mapped process port.
     */
    void map_output_port(process::name_t const& group,
                         process::port_t const& port,
                         process::name_t const& mapped_process,
                         process::port_t const& mapped_port);

    /**
     * \brief Sets the pipeline up for execution.
     */
    void setup_pipeline();

    /**
     * \brief Runs the pipeline.
     */
    void run();
    /**
     * \brief Resets the pipeline.
     */
    void reset();
    /**
     * \brief Shuts the pipeline down.
     */
    void shutdown();

    /**
     * \brief Get a list of processes in the pipeline.
     *
     * \returns The names of all processes in the pipeline.
     */
    process::names_t process_names() const;
    /**
     * \brief Get a list process by name.
     *
     * \throws no_such_process Thrown when \p name does not exist in the pipeline.
     *
     * \returns The process in the pipeline with the given name.
     */
    process_t process_by_name(process::name_t const& name) const;

    /**
     * \brief Find processes that are feeding data directly into a process.
     *
     * \param name The name of the process to lookup.
     *
     * \returns All processes that feed data into \p name.
     */
    processes_t upstream_for_process(process::name_t const& name) const;
    /**
     * \brief Find processes that are siphoning data directly from a process.
     *
     * \param name The name of the process to lookup.
     *
     * \returns All processes that receive data from \p name.
     */
    processes_t downstream_for_process(process::name_t const& name) const;
    /**
     * \brief Find processes that are siphoning data directly from a port.
     *
     * \param name The name of the process to lookup.
     * \param port The name of the port on the process.
     *
     * \returns All processes that receive data from \p name's \p port.
     */
    processes_t downstream_for_port(process::name_t const& name, process::port_t const& port) const;
    /**
     * \brief Find ports that are siphoning data directly from a port.
     *
     * \param name The name of the process to lookup.
     * \param port The name of the port on the process.
     *
     * \returns All port addresses that receive data from \p name's \p port.
     */
    process::port_addrs_t receivers_for_port(process::name_t const& name, process::port_t const& port) const;

    /**
     * \brief Find edges that are feeding data directly into a process.
     *
     * \param name The name of the process to lookup.
     *
     * \returns All edges that carry data into \p name.
     */
    edges_t input_edges_for_process(process::name_t const& name) const;
    /**
     * \brief Find edges that are siphoning data directly from a process.
     *
     * \param name The name of the process to lookup.
     *
     * \returns All edges that carry data from \p name.
     */
    edges_t output_edges_for_process(process::name_t const& name) const;
    /**
     * \brief Find edges that are siphoning data directly from a port.
     *
     * \param name The name of the process to lookup.
     * \param port The name of the port on the process.
     *
     * \returns All edges that carry data from \p name's \p port.
     */
    edges_t output_edges_for_port(process::name_t const& name, process::port_t const& port) const;

    /**
     * \brief The groups within the pipeline.
     *
     * \returns The list of all group names in the pipeline.
     */
    process::names_t groups() const;
    /**
     * \brief The input port names for a group.
     *
     * \throws no_such_group Thrown when \p name does not exist in the pipeline.
     *
     * \param name The name of the group.
     *
     * \returns The list of input ports available for \p name.
     */
    process::ports_t input_ports_for_group(process::name_t const& name) const;
    /**
     * \brief The output port names for a group.
     *
     * \throws no_such_group Thrown when \p name does not exist in the pipeline.
     *
     * \param name The name of the group.
     *
     * \returns The list of output ports available for \p name.
     */
    process::ports_t output_ports_for_group(process::name_t const& name) const;
    /**
     * \brief Ports that are mapped to the group input port.
     *
     * \throws no_such_group Thrown when \p name does not exist in the pipeline.
     * \throws no_such_group_port Thrown when \p port is not an input port on the group.
     *
     * \param name The name of the group.
     * \param port The name of the port.
     *
     * \returns The list of input ports mapped for \p name on the \p port port.
     */
    process::port_addrs_t mapped_group_input_ports(process::name_t const& name, process::port_t const& port) const;
    /**
     * \brief The port that is mapped to the group output port.
     *
     * \throws no_such_group Thrown when \p name does not exist in the pipeline.
     * \throws no_such_group_port Thrown when \p port is not an output port on the group.
     *
     * \param name The name of the group.
     * \param port The name of the port.
     *
     * \returns The output port mapped for \p name on the \p port port.
     */
    process::port_addr_t mapped_group_output_ports(process::name_t const& name, process::port_t const& port) const;
  private:
    class priv;
    boost::shared_ptr<priv> d;
};

} // end namespace vistk

#endif // VISTK_PIPELINE_PIPELINE_H
