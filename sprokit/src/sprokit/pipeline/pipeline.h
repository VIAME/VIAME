// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PIPELINE_PIPELINE_H
#define SPROKIT_PIPELINE_PIPELINE_H

#include <sprokit/pipeline/sprokit_pipeline_export.h>

#include "process.h"
#include "types.h"

#include <vital/noncopyable.h>

/**
 * \file pipeline.h
 *
 * \brief Header for \link sprokit::pipeline pipelines\endlink.
 */

namespace sprokit {

/**
 * \class pipeline pipeline.h <sprokit/pipeline/pipeline.h>
 *
 * \brief A collection of interconnected \link process processes\endlink.
 *
 * \ingroup base_classes
 */
class SPROKIT_PIPELINE_EXPORT pipeline
  : private kwiver::vital::noncopyable
{
  public:
    /**
     * \brief Constructor.
     *
     * \preconds
     *
     * \precond{\p config}
     *
     * \endpreconds
     *
     * \throws null_pipeline_config_exception Thrown when \p config is \c NULL.
     *
     * \param config Contains configuration for the pipeline.
     */
    pipeline(kwiver::vital::config_block_sptr const& config = kwiver::vital::config_block::empty_config());
    /**
     * \brief Destructor.
     */
    ~pipeline();

    /**
     * \brief Add a process to the pipeline.
     *
     * \preconds
     *
     * \precond{\p process}
     * \precond{Another process is not already named \c process->name()}
     *
     * \endpreconds
     *
     * \throws add_after_setup_exception Thrown if called after the pipeline has been setup.
     * \throws null_process_addition_exception Thrown when \p process is \c NULL.
     * \throws duplicate_process_name_exception Thrown when \p process has the same name as another process in the pipeline already.
     *
     * \postconds
     *
     * \postcond{\p process is owned by \c this}
     *
     * \endpostconds
     *
     * \param process The process to add to the pipeline.
     */
    void add_process(process_t const& process);

    /**
     * \brief Remove a process to the pipeline.
     *
     * \preconds
     *
     * \precond{A process named \c name already exists in the pipeline}
     *
     * \endpreconds
     *
     * \throws remove_after_setup_exception Thrown if called after the pipeline has been setup.
     * \throws no_such_process_exception Thrown when there is no process named \p name in the pipeline.
     *
     * \postconds
     *
     * \postcond{The process named \p name is no longer owned by \c this}
     * \postcond{All connections and mappings to the \p name process are removed}
     *
     * \endpostconds
     *
     * \param name The process to add to the pipeline.
     */
    void remove_process(process::name_t const& name);

    /**
     * \brief Connect two ports in the pipeline together with an edge.
     *
     * \preconds
     *
     * \precond{\p upstream_name is the name of a process in the pipeline}
     * \precond{\p upstream_port is an output port on \p upstream_name}
     * \precond{\p downstream_name is the name of a process in the pipeline}
     * \precond{\p downstream_port is an input port on \p downstream_name}
     * \precond{The types of the ports are compatible\, or at least one is \ref process::type_any}
     * \precond{The flags of the ports are compatible (a \ref
     *          process::flag_output_const output may not be connected to a \ref
     *          process::flag_input_mutable input)}
     *
     * \endpreconds
     *
     * \throws connection_after_setup_exception Thrown if called after the pipeline has been setup.
     * \throws connection_dependent_type_exception Thrown when a dependent type is rejected.
     * \throws connection_dependent_type_cascade_exception Thrown when an indirect dependent type is rejected.
     * \throws connection_type_mismatch_exception Thrown when the types of the ports are incompatible.
     * \throws connection_flag_mismatch_exception Thrown when the flags of the ports are incompatible.
     * \throws no_such_process_exception Thrown when either \p upstream_name or \p downstream_name do not exist in the pipeline.
     *
     * \postconds
     *
     * \postcond{The ports \port{upstream_process.upstream_port} and
     *           \port{downstream_process.downstream_port} are connected}
     *
     * \endpostconds
     *
     * \param upstream_name The upstream process name.
     * \param upstream_port The upstream process port.
     * \param downstream_name The downstream process name.
     * \param downstream_port The downstream process port.
     */
    void connect(process::name_t const& upstream_name,
                 process::port_t const& upstream_port,
                 process::name_t const& downstream_name,
                 process::port_t const& downstream_port);

    /**
     * \brief Disconnect two ports in the pipeline together with an edge.
     *
     * \preconds
     *
     * \precond{A connection from \p upstream_process\ \c .\ \p upstream_port to \p downstream_process\ \c .\ \p downstream_port exists in the pipeline}
     *
     * \endpreconds
     *
     * \throws disconnection_after_setup_exception Thrown if called after the pipeline has been setup.
     * \throws no_such_connection_exception Thrown when the connection does not exist within the pipeline.
     *
     * \postconds
     *
     * \postcond{The ports \port{upstream_process.upstream_port} and
     *           \port{downstream_process.downstream_port} are disconnected}
     *
     * \endpostconds
     *
     * \param upstream_name The upstream process name.
     * \param upstream_port The upstream process port.
     * \param downstream_name The downstream process name.
     * \param downstream_port The downstream process port.
     */
    void disconnect(process::name_t const& upstream_name,
                    process::port_t const& upstream_port,
                    process::name_t const& downstream_name,
                    process::port_t const& downstream_port);

    /**
     * \brief Set the pipeline up for execution.
     *
     * This method ensures that all ports with the flag \ref
     * process::flag_required are connected to an edge. It also ensures that the
     * entire pipeline is fully connected (no set of processes is not connected
     * to some other set of processes within the pipeline somehow).
     *
     * \postconds
     *
     * \postcond{The pipeline is ready to be executed}
     *
     * \endpostconds
     *
     * \throws pipeline_duplicate_setup_exception Thrown when called after a previous successful setup.
     * \throws no_processes_exception Thrown when there are no processes in the pipeline.
     * \throws missing_connection_exception Thrown when there is a required port that is not connected in the pipeline.
     * \throws orphaned_processes_exception Thrown when there is a subgraph which is not connected to another subgraph.
     * \throws untyped_data_dependent_exception Thrown when a data-dependent port type is not set after initialization.
     * \throws connection_dependent_type_exception Thrown when a connection creates a port type problem in the pipeline.
     * \throws connection_dependent_type_cascade_exception Thrown when a data-dependent port type creates a problem in the pipeline.
     * \throws untyped_data_dependent_exception Thrown when there are untyped connections left in the pipeline.
     */
    void setup_pipeline();

    /**
     * \brief Query whether the pipeline has been setup.
     *
     * \returns True if the pipeline has been setup, false otherwise.
     */
    bool is_setup() const;

    /**
     * \brief Query whether the pipeline has been setup successfully.
     *
     * \returns True if the pipeline has been setup successfully, false otherwise.
     */
    bool setup_successful() const;

    /**
     * \brief Resets the pipeline.
     *
     * This method resets a stopped pipeline. The process::reset()
     * method is called on each process and all internal
     * connections/edges are cleared. Then all connections are
     * reestablished.
     *
     * Note that setup_pipeline() must be called after reset to get
     * the pipeline in running condition.
     *
     * \throws reset_running_pipeline_exception Thrown when the
     * pipeline is running.
     */
    void reset();

    /**
     * \brief Reconfigure processes within the pipeline.
     *
     * This method causes the pipeline to reconfigure all processes by
     * calling their reconfigure() method. The supplied config block
     * is used to supply an updated configuration to each process and
     * cluster in the pipeline.
     *
     * This method can be called anytime after the pipeline has been set up.
     *
     * \warning This does not ensure that every process gets reconfigured at the
     * same time; any synchronization is best handled at the cluster level if
     * needed.
     */
    void reconfigure(kwiver::vital::config_block_sptr const& conf) const;

    /**
     * \brief Get a list of processes in the pipeline.
     *
     * \returns The names of all processes in the pipeline.
     */
    process::names_t process_names() const;

    /**
     * \brief Get a process by name.
     *
     * \preconds
     *
     * \precond{A process with the name \p name exists in the pipeline}
     *
     * \endpreconds
     *
     * \throws no_such_process_exception Thrown when \p name does not exist in the pipeline.
     *
     * \param name The name of the process to retrieve.
     *
     * \returns The process in the pipeline with the given name.
     */
    process_t process_by_name(process::name_t const& name) const;

    /**
     * \brief Get the cluster a process is a member of.
     *
     * \preconds
     *
     * \precond{A process with the name \p name exists in the pipeline}
     *
     * \endpreconds
     *
     * \throws no_such_process_exception Thrown when \p name does not exist in the pipeline.
     *
     * \param name The name of the process to retrieve.
     *
     * \returns The process in the pipeline with the given name.
     */
    process::name_t parent_cluster(process::name_t const& name) const;

    /**
     * \brief Get a list of processes in the pipeline.
     *
     * \returns The list of all cluster names in the pipeline.
     */
    process::names_t cluster_names() const;

    /**
     * \brief Get a cluster by name.
     *
     * \preconds
     *
     * \precond{A cluster with the name \p name exists in the pipeline}
     *
     * \endpreconds
     *
     * \throws no_such_process_exception Thrown when \p name does not exist in the pipeline.
     *
     * \param name The name of the cluster to retrieve.
     *
     * \returns The cluster in the pipeline with the given name.
     */
    process_cluster_t cluster_by_name(process::name_t const& name) const;

    /**
     * \brief Find the ports requested to receive data from a port.
     *
     * \param name The name to lookup.
     * \param port The port to lookup.
     *
     * \returns The port address that sends data from \p name's \p port.
     */
    process::port_addrs_t connections_from_addr(process::name_t const& name, process::port_t const& port) const;

    /**
     * \brief Find the port requested to send data to a port.
     *
     * \param name The name to lookup.
     * \param port The port to lookup.
     *
     * \returns All port addresses that receive data from \p name's \p port.
     */
    process::port_addr_t connection_to_addr(process::name_t const& name, process::port_t const& port) const;

    /**
     * \brief Find processes that are feeding data directly into a process.
     *
     * \throws pipeline_not_setup_exception Thrown when the pipeline has not been setup.
     * \throws pipeline_not_ready_exception Thrown when the pipeline has not been setup successfully.
     *
     * \param name The name of the process to lookup.
     *
     * \returns All processes that feed data into \p name.
     */
    processes_t upstream_for_process(process::name_t const& name) const;

    /**
     * \brief Find the process that is sending data directly to a port.
     *
     * \throws pipeline_not_setup_exception Thrown when the pipeline has not been setup.
     * \throws pipeline_not_ready_exception Thrown when the pipeline has not been setup successfully.
     *
     * \param name The name of the process to lookup.
     * \param port The name of the port on the process.
     *
     * \returns The process that sends data to \p name's \p port.
     */
    process_t upstream_for_port(process::name_t const& name, process::port_t const& port) const;

    /**
     * \brief Find processes that are siphoning data directly from a process.
     *
     * \throws pipeline_not_setup_exception Thrown when the pipeline has not been setup.
     * \throws pipeline_not_ready_exception Thrown when the pipeline has not been setup successfully.
     *
     * \param name The name of the process to lookup.
     *
     * \returns All processes that receive data from \p name.
     */
    processes_t downstream_for_process(process::name_t const& name) const;

    /**
     * \brief Find processes that are siphoning data directly from a port.
     *
     * \throws pipeline_not_setup_exception Thrown when the pipeline has not been setup.
     * \throws pipeline_not_ready_exception Thrown when the pipeline has not been setup successfully.
     *
     * \param name The name of the process to lookup.
     * \param port The name of the port on the process.
     *
     * \returns All processes that receive data from \p name's \p port.
     */
    processes_t downstream_for_port(process::name_t const& name, process::port_t const& port) const;

    /**
     * \brief Find the port that is sending data directly to a port.
     *
     * \throws pipeline_not_setup_exception Thrown when the pipeline has not been setup.
     * \throws pipeline_not_ready_exception Thrown when the pipeline has not been setup successfully.
     *
     * \param name The name of the process to lookup.
     * \param port The name of the port on the process.
     *
     * \returns The port address that sends data from \p name's \p port.
     */
    process::port_addr_t sender_for_port(process::name_t const& name, process::port_t const& port) const;

    /**
     * \brief Find ports that are siphoning data directly from a port.
     *
     * \throws pipeline_not_setup_exception Thrown when the pipeline has not been setup.
     * \throws pipeline_not_ready_exception Thrown when the pipeline has not been setup successfully.
     *
     * \param name The name of the process to lookup.
     * \param port The name of the port on the process.
     *
     * \returns All port addresses that receive data from \p name's \p port.
     */
    process::port_addrs_t receivers_for_port(process::name_t const& name, process::port_t const& port) const;

    /**
     * \brief Find the edge that represents a connection.
     *
     * \throws pipeline_not_setup_exception Thrown when the pipeline has not been setup.
     * \throws pipeline_not_ready_exception Thrown when the pipeline has not been setup successfully.
     *
     * \param upstream_name The upstream process name.
     * \param upstream_port The upstream process port.
     * \param downstream_name The downstream process name.
     * \param downstream_port The downstream process port.
     *
     * \returns The edge for the connection, or \c NULL if there is no such connection.
     */
    edge_t edge_for_connection(process::name_t const& upstream_name,
                               process::port_t const& upstream_port,
                               process::name_t const& downstream_name,
                               process::port_t const& downstream_port) const;

    /**
     * \brief Find edges that are feeding data directly into a process.
     *
     * \throws pipeline_not_setup_exception Thrown when the pipeline has not been setup.
     * \throws pipeline_not_ready_exception Thrown when the pipeline has not been setup successfully.
     *
     * \param name The name of the process to lookup.
     *
     * \returns All edges that carry data into \p name.
     */
    edges_t input_edges_for_process(process::name_t const& name) const;

    /**
     * \brief Find the edge that is sending data directly to a port.
     *
     * \throws pipeline_not_setup_exception Thrown when the pipeline has not been setup.
     * \throws pipeline_not_ready_exception Thrown when the pipeline has not been setup successfully.
     *
     * \param name The name of the process to lookup.
     * \param port The name of the port on the process.
     *
     * \returns The edge that sends data to \p name's \p port, or \c NULL.
     */
    edge_t input_edge_for_port(process::name_t const& name, process::port_t const& port) const;

    /**
     * \brief Find edges that are siphoning data directly from a process.
     *
     * \throws pipeline_not_setup_exception Thrown when the pipeline has not been setup.
     * \throws pipeline_not_ready_exception Thrown when the pipeline has not been setup successfully.
     *
     * \param name The name of the process to lookup.
     *
     * \returns All edges that carry data from \p name.
     */
    edges_t output_edges_for_process(process::name_t const& name) const;

    /**
     * \brief Find edges that are siphoning data directly from a port.
     *
     * \throws pipeline_not_setup_exception Thrown when the pipeline has not been setup.
     * \throws pipeline_not_ready_exception Thrown when the pipeline has not been setup successfully.
     *
     * \param name The name of the process to lookup.
     * \param port The name of the port on the process.
     *
     * \returns All edges that carry data from \p name's \p port.
     */
    edges_t output_edges_for_port(process::name_t const& name, process::port_t const& port) const;

    /**
     * \brief Check to see if the pipeline has any python processes.
     *
     * This method returns a list python processes from the
     * pipeline. An empty list is returned if there are no python
     * processes.
     *
     * \returns Return a list of python processes
     */
    processes_t get_python_processes() const;

  private:
    friend class scheduler;
    SPROKIT_PIPELINE_NO_EXPORT void start();
    SPROKIT_PIPELINE_NO_EXPORT void stop();

    class SPROKIT_PIPELINE_NO_EXPORT priv;
    std::unique_ptr<priv> d;
};

}

#endif // SPROKIT_PIPELINE_PIPELINE_H
