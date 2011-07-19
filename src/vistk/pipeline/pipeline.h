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

namespace vistk
{

/**
 * \class pipeline
 *
 * \brief A connection between two process ports which can carry data.
 *
 * \ingroup base_classes
 */
class VISTK_PIPELINE_EXPORT pipeline
  : boost::noncopyable
{
  public:
    /**
     * \brief Destructor.
     */
    virtual ~pipeline();

    /**
     * \brief Add a process to the pipeline.
     *
     * \throws null_process_addition Thrown when \p process is \c NULL.
     * \throws duplicate_process_name Thrown when \p process has the same name as another process in the pipeline already.
     *
     * \param process The process to add to the pipeline.
     */
    virtual void add_process(process_t process);

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
    virtual void connect(process::name_t const& upstream_process,
                         process::port_t const& upstream_port,
                         process::name_t const& downstream_process,
                         process::port_t const& downstream_port,
                         edge_t edge);

    /**
     * \brief Sets the pipeline up for execution.
     */
    void setup_pipeline();

    /**
     * \brief Runs the pipeline.
     */
    virtual void run() = 0;

    /**
     * \brief Shuts the pipeline down.
     */
    virtual void shutdown() = 0;
  protected:
    /**
     * \brief Constructor.
     *
     * \param config Contains configuration for the edge.
     */
    pipeline(config_t const& config);

    /**
     * \brief Subclass pipeline setup method.
     */
    virtual void _setup_pipeline();

    /// Type for a map of processes.
    typedef std::map<process::name_t, process_t> process_map_t;
    /// Type for the address of a port within the pipeline.
    typedef std::pair<process::name_t, process::port_t> port_addr_t;
    /// Type for a connection between two ports.
    typedef std::pair<port_addr_t, port_addr_t> connection_t;
    /// Type for a collection of connections.
    typedef std::vector<connection_t> connections_t;
    /// Type for referencing edges by the connection.
    typedef std::map<size_t, edge_t> edge_map_t;

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
     * \brief Find edges that are feeding data directly into a process.
     *
     * \param name The name of the process to lookup.
     *
     * \returns All edges that carry data into \p name.
     */
    edges_t input_edges_for_process(process::name_t const& name) const;
    /**
     * \brief Find processes that are siphoning data directly from an edge.
     *
     * \param name The name of the process to lookup.
     *
     * \returns All edges that carry data from \p name.
     */
    edges_t output_edges_for_process(process::name_t const& name) const;

    /// All connections made within the pipeline.
    connections_t m_connections;

    /// All processes within the pipeline.
    process_map_t m_process_map;
    /// All edges within the pipeline.
    edge_map_t m_edge_map;
  private:
    class priv;
    boost::shared_ptr<priv> d;
};

} // end namespace vistk

#endif // VISTK_PIPELINE_PIPELINE_H
