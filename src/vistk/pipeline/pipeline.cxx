/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "pipeline.h"
#include "pipeline_exception.h"

#include <boost/foreach.hpp>

#include <set>

/**
 * \file pipeline.cxx
 *
 * \brief Implementation of the base class for \link pipeline pipelines\endlink.
 */

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
  /// \todo Check for disconnected pipelines.
  /// \todo Check for types of connections.
}

process::names_t
pipeline
::process_names() const
{
  process::names_t names;

  BOOST_FOREACH (process_map_t::value_type const& process_index, m_process_map)
  {
    names.push_back(process_index.first);
  }

  return names;
}

process_t
pipeline
::process_by_name(process::name_t const& name) const
{
  process_map_t::const_iterator i = m_process_map.find(name);

  if (i == m_process_map.end())
  {
    throw no_such_process(name);
  }

  return i->second;
}

processes_t
pipeline
::upstream_for_process(process::name_t const& name) const
{
  std::set<process::name_t> process_names;

  BOOST_FOREACH (connection_t const& connection, m_connections)
  {
    if (connection.second.first == name)
    {
      process::name_t const& upstream_name = connection.first.first;

      process_names.insert(upstream_name);
    }
  }

  processes_t processes;

  BOOST_FOREACH (process::name_t const& process_name, process_names)
  {
    process_map_t::const_iterator i = m_process_map.find(process_name);

    processes.push_back(i->second);
  }

  return processes;
}

processes_t
pipeline
::downstream_for_process(process::name_t const& name) const
{
  std::set<process::name_t> process_names;

  BOOST_FOREACH (connection_t const& connection, m_connections)
  {
    if (connection.first.first == name)
    {
      process_names.insert(connection.second.first);
    }
  }

  processes_t processes;

  BOOST_FOREACH (process::name_t const& process_name, process_names)
  {
    process_map_t::const_iterator i = m_process_map.find(process_name);

    processes.push_back(i->second);
  }

  return processes;
}

processes_t
pipeline
::downstream_for_port(process::name_t const& name, process::port_t const& port) const
{
  std::set<process::name_t> process_names;

  BOOST_FOREACH (connection_t const& connection, m_connections)
  {
    if ((connection.first.first == name) &&
        (connection.first.second == port))
    {
      process_names.insert(connection.second.first);
    }
  }

  processes_t processes;

  BOOST_FOREACH (process::name_t const& process_name, process_names)
  {
    process_map_t::const_iterator i = m_process_map.find(process_name);

    processes.push_back(i->second);
  }

  return processes;
}

pipeline::port_addrs_t
pipeline
::receivers_for_port(process::name_t const& name, process::port_t const& port) const
{
  port_addrs_t port_addrs;

  BOOST_FOREACH (connection_t const& connection, m_connections)
  {
    if ((connection.first.first == name) &&
        (connection.first.second == port))
    {
      port_addrs.push_back(connection.second);
    }
  }

  return port_addrs;
}

edges_t
pipeline
::input_edges_for_process(process::name_t const& name) const
{
  edges_t edges;

  BOOST_FOREACH (edge_map_t::value_type const& edge_index, m_edge_map)
  {
    if (m_connections[edge_index.first].second.first == name)
    {
      edges.push_back(edge_index.second);
    }
  }

  return edges;
}

edges_t
pipeline
::output_edges_for_process(process::name_t const& name) const
{
  edges_t edges;

  BOOST_FOREACH (edge_map_t::value_type const& edge_index, m_edge_map)
  {
    if (m_connections[edge_index.first].first.first == name)
    {
      edges.push_back(edge_index.second);
    }
  }

  return edges;
}

edges_t
pipeline
::output_edges_for_port(process::name_t const& name, process::port_t const& port) const
{
  edges_t edges;

  BOOST_FOREACH (edge_map_t::value_type const& edge_index, m_edge_map)
  {
    if ((m_connections[edge_index.first].first.first == name) &&
        (m_connections[edge_index.first].first.second == port))
    {
      edges.push_back(edge_index.second);
    }
  }

  return edges;
}

pipeline
::pipeline(config_t const& /*config*/)
{
  d = boost::shared_ptr<priv>(new priv);
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
