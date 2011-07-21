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

  group_t::const_iterator group_it = m_groups.find(name);

  if (group_it != m_groups.end())
  {
    throw duplicate_process_name(name);
  }

  m_process_map[name] = process;
}

void
pipeline
::add_group(process::name_t const& name)
{
  process_map_t::const_iterator proc_it = m_process_map.find(name);

  if (proc_it != m_process_map.end())
  {
    throw duplicate_process_name(name);
  }

  group_t::const_iterator group_it = m_groups.find(name);

  if (group_it != m_groups.end())
  {
    throw duplicate_process_name(name);
  }

  m_groups[name] = port_mapping_t();
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

  /// \todo Check if up or downstream is a group.

  process::port_addr_t const up_port = process::port_addr_t(upstream_process, upstream_port);
  process::port_addr_t const down_port = process::port_addr_t(downstream_process, downstream_port);
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
::map_input_port(process::name_t const& group,
                 process::port_t const& port,
                 process::name_t const& mapped_process,
                 process::port_t const& mapped_port)
{
  group_t::iterator group_it = m_groups.find(group);

  if (group_it == m_groups.end())
  {
    throw no_such_group(group);
  }

  process_map_t::iterator proc_it = m_process_map.find(mapped_process);

  if (proc_it == m_process_map.end())
  {
    throw no_such_process(mapped_process);
  }

  group_it->second.first[port].push_back(process::port_addr_t(mapped_process, mapped_port));
}

void
pipeline
::map_output_port(process::name_t const& group,
                  process::port_t const& port,
                  process::name_t const& mapped_process,
                  process::port_t const& mapped_port)
{
  group_t::iterator group_it = m_groups.find(group);

  if (group_it == m_groups.end())
  {
    throw no_such_group(group);
  }

  process_map_t::iterator proc_it = m_process_map.find(mapped_process);

  if (proc_it == m_process_map.end())
  {
    throw no_such_process(mapped_process);
  }

  output_port_mapping_t::const_iterator port_it = group_it->second.second.find(port);

  if (port_it != group_it->second.second.end())
  {
    throw group_output_already_mapped(group, port, port_it->second.first, port_it->second.second, mapped_process, mapped_port);
  }

  group_it->second.second[port] = process::port_addr_t(mapped_process, mapped_port);
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

process::port_addrs_t
pipeline
::receivers_for_port(process::name_t const& name, process::port_t const& port) const
{
  process::port_addrs_t port_addrs;

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

process::names_t
pipeline
::groups() const
{
  process::names_t names;

  BOOST_FOREACH (group_t::value_type const& group, m_groups)
  {
    names.push_back(group.first);
  }

  return names;
}

process::ports_t
pipeline
::input_ports_for_group(process::name_t const& name) const
{
  process::ports_t ports;

  group_t::const_iterator group_it = m_groups.find(name);

  if (group_it == m_groups.end())
  {
    throw no_such_group(name);
  }

  BOOST_FOREACH (input_port_mapping_t::value_type const& port_it, group_it->second.first)
  {
    ports.push_back(port_it.first);
  }

  return ports;
}

process::ports_t
pipeline
::output_ports_for_group(process::name_t const& name) const
{
  process::ports_t ports;

  group_t::const_iterator group_it = m_groups.find(name);

  if (group_it == m_groups.end())
  {
    throw no_such_group(name);
  }

  BOOST_FOREACH (output_port_mapping_t::value_type const& port_it, group_it->second.second)
  {
    ports.push_back(port_it.first);
  }

  return ports;
}

process::port_addrs_t
pipeline
::mapped_group_input_ports(process::name_t const& name, process::port_t const& port) const
{
  group_t::const_iterator group_it = m_groups.find(name);

  if (group_it == m_groups.end())
  {
    throw no_such_group(name);
  }

  input_port_mapping_t::const_iterator mapping_it = group_it->second.first.find(port);

  if (mapping_it == group_it->second.first.end())
  {
    throw no_such_group_port(name, port);
  }

  return mapping_it->second;
}

process::port_addr_t
pipeline
::mapped_group_output_ports(process::name_t const& name, process::port_t const& port) const
{
  group_t::const_iterator group_it = m_groups.find(name);

  if (group_it == m_groups.end())
  {
    throw no_such_group(name);
  }

  output_port_mapping_t::const_iterator mapping_it = group_it->second.second.find(port);

  if (mapping_it == group_it->second.second.end())
  {
    throw no_such_group_port(name, port);
  }

  return mapping_it->second;
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
