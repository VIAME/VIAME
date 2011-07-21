/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "pipeline.h"
#include "pipeline_exception.h"

#include "edge.h"

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

    typedef std::map<process::name_t, process_t> process_map_t;
    typedef std::pair<process::port_addr_t, process::port_addr_t> connection_t;
    typedef std::vector<connection_t> connections_t;
    typedef std::map<size_t, edge_t> edge_map_t;

    typedef std::map<process::port_t, process::port_addrs_t> input_port_mapping_t;
    typedef std::map<process::port_t, process::port_addr_t> output_port_mapping_t;
    typedef std::pair<input_port_mapping_t, output_port_mapping_t> port_mapping_t;
    typedef std::map<process::name_t, port_mapping_t> group_t;

    connections_t connections;

    process_map_t process_map;
    edge_map_t edge_map;

    group_t groups;
};

pipeline
::pipeline(config_t const& /*config*/)
{
  d = boost::shared_ptr<priv>(new priv);
}

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

  priv::process_map_t::const_iterator proc_it = d->process_map.find(name);

  if (proc_it != d->process_map.end())
  {
    throw duplicate_process_name(name);
  }

  priv::group_t::const_iterator group_it = d->groups.find(name);

  if (group_it != d->groups.end())
  {
    throw duplicate_process_name(name);
  }

  d->process_map[name] = process;
}

void
pipeline
::add_group(process::name_t const& name)
{
  priv::process_map_t::const_iterator proc_it = d->process_map.find(name);

  if (proc_it != d->process_map.end())
  {
    throw duplicate_process_name(name);
  }

  priv::group_t::const_iterator group_it = d->groups.find(name);

  if (group_it != d->groups.end())
  {
    throw duplicate_process_name(name);
  }

  d->groups[name] = priv::port_mapping_t();
}

void
pipeline
::connect(process::name_t const& upstream_process,
          process::port_t const& upstream_port,
          process::name_t const& downstream_process,
          process::port_t const& downstream_port)
{
  /// \todo Check if up or downstream is a group.

  config_t edge_config = config::empty_config();

  edge_t e = edge_t(new edge(edge_config));

  process::port_addr_t const up_port = process::port_addr_t(upstream_process, upstream_port);
  process::port_addr_t const down_port = process::port_addr_t(downstream_process, downstream_port);
  priv::connection_t const conn = priv::connection_t(up_port, down_port);

  priv::process_map_t::iterator const up_it = d->process_map.find(upstream_process);
  priv::process_map_t::iterator const down_it = d->process_map.find(downstream_process);

  if (up_it == d->process_map.end())
  {
    throw no_such_process(upstream_process);
  }
  if (down_it == d->process_map.end())
  {
    throw no_such_process(downstream_process);
  }

  up_it->second->connect_output_port(upstream_port, e);
  down_it->second->connect_input_port(downstream_port, e);

  d->edge_map[d->connections.size()] = e;
  d->connections.push_back(conn);
}

void
pipeline
::map_input_port(process::name_t const& group,
                 process::port_t const& port,
                 process::name_t const& mapped_process,
                 process::port_t const& mapped_port)
{
  priv::group_t::iterator const group_it = d->groups.find(group);

  if (group_it == d->groups.end())
  {
    throw no_such_group(group);
  }

  priv::process_map_t::iterator const proc_it = d->process_map.find(mapped_process);

  if (proc_it == d->process_map.end())
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
  priv::group_t::iterator const group_it = d->groups.find(group);

  if (group_it == d->groups.end())
  {
    throw no_such_group(group);
  }

  priv::process_map_t::iterator const proc_it = d->process_map.find(mapped_process);

  if (proc_it == d->process_map.end())
  {
    throw no_such_process(mapped_process);
  }

  priv::output_port_mapping_t::const_iterator const port_it = group_it->second.second.find(port);

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

  BOOST_FOREACH (priv::process_map_t::value_type const& process_index, d->process_map)
  {
    names.push_back(process_index.first);
  }

  return names;
}

process_t
pipeline
::process_by_name(process::name_t const& name) const
{
  priv::process_map_t::const_iterator i = d->process_map.find(name);

  if (i == d->process_map.end())
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

  BOOST_FOREACH (priv::connection_t const& connection, d->connections)
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
    priv::process_map_t::const_iterator i = d->process_map.find(process_name);

    processes.push_back(i->second);
  }

  return processes;
}

processes_t
pipeline
::downstream_for_process(process::name_t const& name) const
{
  std::set<process::name_t> process_names;

  BOOST_FOREACH (priv::connection_t const& connection, d->connections)
  {
    if (connection.first.first == name)
    {
      process_names.insert(connection.second.first);
    }
  }

  processes_t processes;

  BOOST_FOREACH (process::name_t const& process_name, process_names)
  {
    priv::process_map_t::const_iterator i = d->process_map.find(process_name);

    processes.push_back(i->second);
  }

  return processes;
}

processes_t
pipeline
::downstream_for_port(process::name_t const& name, process::port_t const& port) const
{
  std::set<process::name_t> process_names;

  BOOST_FOREACH (priv::connection_t const& connection, d->connections)
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
    priv::process_map_t::const_iterator i = d->process_map.find(process_name);

    processes.push_back(i->second);
  }

  return processes;
}

process::port_addrs_t
pipeline
::receivers_for_port(process::name_t const& name, process::port_t const& port) const
{
  process::port_addrs_t port_addrs;

  BOOST_FOREACH (priv::connection_t const& connection, d->connections)
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

  BOOST_FOREACH (priv::edge_map_t::value_type const& edge_index, d->edge_map)
  {
    if (d->connections[edge_index.first].second.first == name)
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

  BOOST_FOREACH (priv::edge_map_t::value_type const& edge_index, d->edge_map)
  {
    if (d->connections[edge_index.first].first.first == name)
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

  BOOST_FOREACH (priv::edge_map_t::value_type const& edge_index, d->edge_map)
  {
    if ((d->connections[edge_index.first].first.first == name) &&
        (d->connections[edge_index.first].first.second == port))
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

  BOOST_FOREACH (priv::group_t::value_type const& group, d->groups)
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

  priv::group_t::const_iterator const group_it = d->groups.find(name);

  if (group_it == d->groups.end())
  {
    throw no_such_group(name);
  }

  BOOST_FOREACH (priv::input_port_mapping_t::value_type const& port_it, group_it->second.first)
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

  priv::group_t::const_iterator const group_it = d->groups.find(name);

  if (group_it == d->groups.end())
  {
    throw no_such_group(name);
  }

  BOOST_FOREACH (priv::output_port_mapping_t::value_type const& port_it, group_it->second.second)
  {
    ports.push_back(port_it.first);
  }

  return ports;
}

process::port_addrs_t
pipeline
::mapped_group_input_ports(process::name_t const& name, process::port_t const& port) const
{
  priv::group_t::const_iterator const group_it = d->groups.find(name);

  if (group_it == d->groups.end())
  {
    throw no_such_group(name);
  }

  priv::input_port_mapping_t::const_iterator const mapping_it = group_it->second.first.find(port);

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
  priv::group_t::const_iterator const group_it = d->groups.find(name);

  if (group_it == d->groups.end())
  {
    throw no_such_group(name);
  }

  priv::output_port_mapping_t::const_iterator const mapping_it = group_it->second.second.find(port);

  if (mapping_it == group_it->second.second.end())
  {
    throw no_such_group_port(name, port);
  }

  return mapping_it->second;
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
