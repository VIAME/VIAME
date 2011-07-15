/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "pipeline.h"
#include "pipeline_exception.h"

namespace vistk
{

class pipeline::priv
{
  public:
    priv();
    ~priv();
};

pipeline
::~pipeline()
{
}

void
pipeline
::add_process(process_t process)
{
  if (!process)
  {
    throw null_process_addition();
  }

  process::name_t const name = process->name();

  process_map_t::const_iterator proc_it = m_process_map.find(name);

  if (proc_it != m_process_map.end())
  {
    throw duplicate_process_name(name);
  }

  m_process_map[name] = process;
}

void
pipeline
::connect(process::name_t const& upstream_process,
          process::port_t const& upstream_port,
          process::name_t const& downstream_process,
          process::port_t const& downstream_port,
          edge_t edge)
{
  if (!edge)
  {
    throw null_edge_connection(upstream_process, upstream_port,
                               downstream_process, downstream_port);
  }

  port_addr_t const up_port = port_addr_t(upstream_process, upstream_port);
  port_addr_t const down_port = port_addr_t(downstream_process, downstream_port);
  connection_t const conn = connection_t(up_port, down_port);

  process_map_t::iterator up_it = m_process_map.find(upstream_process);
  process_map_t::iterator down_it = m_process_map.find(downstream_process);

  if (up_it == m_process_map.end())
  {
    throw no_such_process(upstream_process);
  }
  if (down_it == m_process_map.end())
  {
    throw no_such_process(downstream_process);
  }

  up_it->second->connect_output_port(upstream_port, edge);
  down_it->second->connect_input_port(downstream_port, edge);

  m_edge_map[m_connections.size()] = edge;
  m_connections.push_back(conn);
}

void
pipeline
::setup_pipeline()
{
  /// \todo Check for orphan processes
  /// \todo Check for disconnected pipelines

  _setup_pipeline();
}

pipeline
::pipeline(config_t const& /*config*/)
{
  d = boost::shared_ptr<priv>(new priv);
}

processes_t
pipeline
::upstream_for_process(process::name_t const& /*name*/) const
{
  /// \todo Find upstream processes of another process.

  return processes_t();
}

processes_t
pipeline
::downstream_for_process(process::name_t const& /*name*/) const
{
  /// \todo Find downstream processes of another process.

  return processes_t();
}

edges_t
pipeline
::input_edges_for_process(process::name_t const& /*name*/) const
{
  /// \todo Find input edges for a given process.

  return edges_t();
}

edges_t
pipeline
::output_edges_for_process(process::name_t const& /*name*/) const
{
  /// \todo Find output edges for a given process.

  return edges_t();
}

void
pipeline
::_setup_pipeline()
{
}

pipeline::priv
::priv()
{
}

pipeline::priv
::~priv()
{
}

} // end namespace vistk
