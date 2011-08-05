/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "pipeline.h"
#include "pipeline_exception.h"

#include "edge.h"
#include "process_exception.h"

#include <boost/foreach.hpp>

#include <queue>
#include <set>

/**
 * \file pipeline.cxx
 *
 * \brief Implementation of the base class for \link vistk::pipeline pipelines\endlink.
 */

namespace vistk
{

class pipeline::priv
{
  public:
    priv();
    ~priv();

    void check_duplicate_name(process::name_t const& name);

    typedef std::map<process::name_t, process_t> process_map_t;
    typedef std::pair<process::port_addr_t, process::port_addr_t> connection_t;
    typedef std::vector<connection_t> connections_t;
    typedef std::map<size_t, edge_t> edge_map_t;

    typedef boost::tuple<process::port_flags_t, process::port_addrs_t> input_mapping_info_t;
    typedef boost::tuple<process::port_flags_t, process::port_addr_t> output_mapping_info_t;
    typedef std::map<process::port_t, input_mapping_info_t> input_port_mapping_t;
    typedef std::map<process::port_t, output_mapping_info_t> output_port_mapping_t;
    typedef std::pair<input_port_mapping_t, output_port_mapping_t> port_mapping_t;
    typedef std::map<process::name_t, port_mapping_t> group_t;

    typedef std::map<process::name_t, process::ports_t> connected_mappings_t;

    connections_t connections;

    process_map_t process_map;
    edge_map_t edge_map;

    group_t groups;

    connected_mappings_t used_input_mappings;
    connected_mappings_t used_output_mappings;
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
    throw null_process_addition_exception();
  }

  process::name_t const name = process->name();

  d->check_duplicate_name(name);

  d->process_map[name] = process;
}

void
pipeline
::add_group(process::name_t const& name)
{
  d->check_duplicate_name(name);

  d->groups[name] = priv::port_mapping_t();
}

void
pipeline
::connect(process::name_t const& upstream_process,
          process::port_t const& upstream_port,
          process::name_t const& downstream_process,
          process::port_t const& downstream_port)
{
  priv::group_t::const_iterator const up_group_it = d->groups.find(upstream_process);

  if (up_group_it != d->groups.end())
  {
    priv::output_port_mapping_t const& mapping = up_group_it->second.second;

    priv::output_port_mapping_t::const_iterator const mapping_it = mapping.find(upstream_port);

    if (mapping_it != mapping.end())
    {
      process::port_addr_t const& mapped_port_addr = mapping_it->second.get<1>();

      connect(mapped_port_addr.first, mapped_port_addr.second,
              downstream_process, downstream_port);

      d->used_output_mappings[upstream_process].push_back(upstream_port);

      return;
    }
  }

  priv::group_t::const_iterator const down_group_it = d->groups.find(upstream_process);

  if (down_group_it != d->groups.end())
  {
    priv::input_port_mapping_t const& mapping = down_group_it->second.first;

    priv::input_port_mapping_t::const_iterator const mapping_it = mapping.find(downstream_port);

    if (mapping_it != mapping.end())
    {
      process::port_addrs_t const& mapped_port_addrs = mapping_it->second.get<1>();

      BOOST_FOREACH (process::port_addr_t const& port_addr, mapped_port_addrs)
      {
        connect(upstream_process, upstream_port,
                port_addr.first, port_addr.second);
      }

      d->used_input_mappings[downstream_process].push_back(downstream_port);

      return;
    }
  }

  process::port_addr_t const up_port = process::port_addr_t(upstream_process, upstream_port);
  process::port_addr_t const down_port = process::port_addr_t(downstream_process, downstream_port);
  priv::connection_t const conn = priv::connection_t(up_port, down_port);

  priv::process_map_t::iterator const up_it = d->process_map.find(upstream_process);
  priv::process_map_t::iterator const down_it = d->process_map.find(downstream_process);

  if (up_it == d->process_map.end())
  {
    throw no_such_process_exception(upstream_process);
  }
  if (down_it == d->process_map.end())
  {
    throw no_such_process_exception(downstream_process);
  }

  process_t const& up_proc = up_it->second;
  process_t const& down_proc = down_it->second;

  process::port_type_t const up_type = up_proc->output_port_type(upstream_port);
  process::port_type_t const down_type = down_proc->input_port_type(downstream_port);

  process::port_type_name_t const up_type_name = up_type.get<0>();
  process::port_type_name_t const down_type_name = down_type.get<0>();

  if ((up_type_name != process::type_any) &&
      (down_type_name != process::type_any) &&
      (up_type_name != down_type_name))
  {
    throw connection_type_mismatch_exception(upstream_process, upstream_port, up_type_name,
                                             downstream_process, downstream_port, down_type_name);
  }

  process::port_flags_t const up_flags = up_type.get<1>();
  process::port_flags_t const down_flags = down_type.get<1>();

  process::port_flags_t::const_iterator i;

  i = up_flags.find(process::flag_output_const);

  bool const is_const = (i != up_flags.end());

  i = down_flags.find(process::flag_input_mutable);

  bool const requires_mutable = (i != down_flags.end());

  if (is_const && requires_mutable)
  {
    throw connection_flag_mismatch_exception(upstream_process, upstream_port,
                                             downstream_process, downstream_port);
  }

  config_t edge_config = config::empty_config();

  edge_t e = edge_t(new edge(edge_config));

  up_proc->connect_output_port(upstream_port, e);
  down_proc->connect_input_port(downstream_port, e);

  d->edge_map[d->connections.size()] = e;
  d->connections.push_back(conn);
}

void
pipeline
::map_input_port(process::name_t const& group,
                 process::port_t const& port,
                 process::name_t const& mapped_process,
                 process::port_t const& mapped_port,
                 process::port_flags_t const& flags)
{
  priv::group_t::iterator const group_it = d->groups.find(group);

  if (group_it == d->groups.end())
  {
    throw no_such_group_exception(group);
  }

  priv::process_map_t::iterator const proc_it = d->process_map.find(mapped_process);

  if (proc_it == d->process_map.end())
  {
    throw no_such_process_exception(mapped_process);
  }

  priv::input_port_mapping_t& mapping = group_it->second.first;

  priv::input_mapping_info_t& mapping_info = mapping[port];

  process::port_addr_t const mapped_port_addr = process::port_addr_t(mapped_process, mapped_port);

  mapping_info.get<0>().insert(flags.begin(), flags.end());
  mapping_info.get<1>().push_back(mapped_port_addr);
}

void
pipeline
::map_output_port(process::name_t const& group,
                  process::port_t const& port,
                  process::name_t const& mapped_process,
                  process::port_t const& mapped_port,
                  process::port_flags_t const& flags)
{
  priv::group_t::iterator const group_it = d->groups.find(group);

  if (group_it == d->groups.end())
  {
    throw no_such_group_exception(group);
  }

  priv::process_map_t::iterator const proc_it = d->process_map.find(mapped_process);

  if (proc_it == d->process_map.end())
  {
    throw no_such_process_exception(mapped_process);
  }

  priv::output_port_mapping_t& mapping = group_it->second.second;

  priv::output_port_mapping_t::const_iterator const port_it = mapping.find(port);

  if (port_it != mapping.end())
  {
    process::port_addr_t const prev_port_addr = port_it->second.get<1>();

    throw group_output_already_mapped_exception(group, port, prev_port_addr.first, prev_port_addr.second, mapped_process, mapped_port);
  }

  process::port_addr_t const mapped_port_addr = process::port_addr_t(mapped_process, mapped_port);
  priv::output_mapping_info_t const mapping_info = priv::output_mapping_info_t(flags, mapped_port_addr);

  mapping[port] = mapping_info;
}

void
pipeline
::setup_pipeline()
{
  typedef std::set<process::name_t> name_set_t;
  typedef std::queue<process::name_t> name_queue_t;

  if (!d->process_map.size())
  {
    throw no_processes_exception();
  }

  name_set_t procs;

  {
    name_queue_t to_visit;

    // Traverse the pipeline starting with a process.
    to_visit.push(d->process_map.begin()->first);

    // While we have processes to visit yet.
    while (!to_visit.empty())
    {
      process::name_t const cur_proc = to_visit.front();

      to_visit.pop();

      // Ignore the process if we've already visited it.
      name_set_t::const_iterator const i = procs.find(cur_proc);
      if (i != procs.end())
      {
        continue;
      }

      procs.insert(cur_proc);

      processes_t connected_procs;

      // Find all processes upstream of the current process.
      processes_t const upstream_procs = upstream_for_process(cur_proc);
      connected_procs.insert(connected_procs.end(), upstream_procs.begin(), upstream_procs.end());

      // Find all processes downstream of the current process.
      processes_t const downstream_procs = downstream_for_process(cur_proc);
      connected_procs.insert(connected_procs.end(), downstream_procs.begin(), downstream_procs.end());

      // Mark all connected processes for visitation.
      BOOST_FOREACH (process_t const& proc, connected_procs)
      {
        to_visit.push(proc->name());
      }
    }
  }

  if (d->groups.size())
  {
    process::names_t const group_names = groups();

    BOOST_FOREACH(process::name_t const& cur_group, group_names)
    {
      process::port_addrs_t connected_ports;

      // Get all processes input ports on the group map to.
      process::ports_t const input_ports = input_ports_for_group(cur_group);
      BOOST_FOREACH (process::port_t const& port, input_ports)
      {
        // Check for required flags.
        process::port_flags_t const mapped_port_flags = mapped_group_input_port_flags(cur_group, port);

        process::port_flags_t::const_iterator const i = mapped_port_flags.find(process::flag_required);
        if (i != mapped_port_flags.end())
        {
          priv::connected_mappings_t& connections = d->used_input_mappings;

          priv::connected_mappings_t::const_iterator const c = connections.find(cur_group);

          if (c == connections.end())
          {
            static std::string const reason = "The input mapping has the required flag.";

            throw missing_connection_exception(cur_group, port, reason);
          }
        }

        // Mark mapped ports as connected.
        process::port_addrs_t const mapped_ports = mapped_group_input_ports(cur_group, port);

        connected_ports.insert(connected_ports.end(), mapped_ports.begin(), mapped_ports.end());
      }

      // Get all processes output ports on the group map to.
      process::ports_t const output_ports = output_ports_for_group(cur_group);
      BOOST_FOREACH (process::port_t const& port, output_ports)
      {
        // Check for required flags.
        process::port_flags_t const mapped_port_flags = mapped_group_output_port_flags(cur_group, port);

        process::port_flags_t::const_iterator const i = mapped_port_flags.find(process::flag_required);
        if (i != mapped_port_flags.end())
        {
          priv::connected_mappings_t& connections = d->used_input_mappings;

          priv::connected_mappings_t::const_iterator const c = connections.find(cur_group);

          if (c == connections.end())
          {
            static std::string const reason = "The output mapping has the required flag.";

            throw missing_connection_exception(cur_group, port, reason);
          }
        }

        // Mark mapped ports as connected.
        process::port_addr_t const mapped_port = mapped_group_output_port(cur_group, port);

        connected_ports.push_back(mapped_port);
      }

      // Mark these processes as connected.
      BOOST_FOREACH (process::port_addr_t const& port_addr, connected_ports)
      {
        procs.insert(port_addr.first);
      }
    }
  }

  if (procs.size() != d->process_map.size())
  {
    throw orphaned_processes_exception();
  }
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
    throw no_such_process_exception(name);
  }

  return i->second;
}

processes_t
pipeline
::upstream_for_process(process::name_t const& name) const
{
  std::set<process::name_t> names;

  BOOST_FOREACH (priv::connection_t const& connection, d->connections)
  {
    if (connection.second.first == name)
    {
      process::name_t const& upstream_name = connection.first.first;

      names.insert(upstream_name);
    }
  }

  processes_t processes;

  BOOST_FOREACH (process::name_t const& process_name, names)
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
  std::set<process::name_t> names;

  BOOST_FOREACH (priv::connection_t const& connection, d->connections)
  {
    if (connection.first.first == name)
    {
      process::name_t const& downstream_name = connection.second.first;

      names.insert(downstream_name);
    }
  }

  processes_t processes;

  BOOST_FOREACH (process::name_t const& process_name, names)
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
  std::set<process::name_t> names;

  BOOST_FOREACH (priv::connection_t const& connection, d->connections)
  {
    if ((connection.first.first == name) &&
        (connection.first.second == port))
    {
      process::name_t const& downstream_name = connection.second.first;

      names.insert(downstream_name);
    }
  }

  processes_t processes;

  BOOST_FOREACH (process::name_t const& process_name, names)
  {
    priv::process_map_t::const_iterator i = d->process_map.find(process_name);

    processes.push_back(i->second);
  }

  return processes;
}

process::port_addr_t
pipeline
::sender_for_port(process::name_t const& name, process::port_t const& port) const
{
  process::port_addrs_t port_addrs;

  BOOST_FOREACH (priv::connection_t const& connection, d->connections)
  {
    if ((connection.second.first == name) &&
        (connection.second.second == port))
    {
      process::port_addr_t const& upstream_addr = connection.first;

      return upstream_addr;
    }
  }

  return process::port_addr_t();
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
      process::port_addr_t const& downstream_addr = connection.second;

      port_addrs.push_back(downstream_addr);
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
    process::name_t const& name = group.first;

    names.push_back(name);
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
    throw no_such_group_exception(name);
  }

  priv::input_port_mapping_t const& mapping = group_it->second.first;

  BOOST_FOREACH (priv::input_port_mapping_t::value_type const& port_it, mapping)
  {
    process::port_t const& port = port_it.first;

    ports.push_back(port);
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
    throw no_such_group_exception(name);
  }

  priv::output_port_mapping_t const& mapping = group_it->second.second;

  BOOST_FOREACH (priv::output_port_mapping_t::value_type const& port_it, mapping)
  {
    process::port_t const& port = port_it.first;

    ports.push_back(port);
  }

  return ports;
}

process::port_flags_t
pipeline
::mapped_group_input_port_flags(process::name_t const& name, process::port_t const& port) const
{
  priv::group_t::const_iterator const group_it = d->groups.find(name);

  if (group_it == d->groups.end())
  {
    throw no_such_group_exception(name);
  }

  priv::input_port_mapping_t const& mapping = group_it->second.first;

  priv::input_port_mapping_t::const_iterator const mapping_it = mapping.find(port);

  if (mapping_it == mapping.end())
  {
    throw no_such_group_port_exception(name, port);
  }

  return mapping_it->second.get<0>();
}

process::port_flags_t
pipeline
::mapped_group_output_port_flags(process::name_t const& name, process::port_t const& port) const
{
  priv::group_t::const_iterator const group_it = d->groups.find(name);

  if (group_it == d->groups.end())
  {
    throw no_such_group_exception(name);
  }

  priv::output_port_mapping_t const& mapping = group_it->second.second;

  priv::output_port_mapping_t::const_iterator const mapping_it = mapping.find(port);

  if (mapping_it == mapping.end())
  {
    throw no_such_group_port_exception(name, port);
  }

  return mapping_it->second.get<0>();
}

process::port_addrs_t
pipeline
::mapped_group_input_ports(process::name_t const& name, process::port_t const& port) const
{
  priv::group_t::const_iterator const group_it = d->groups.find(name);

  if (group_it == d->groups.end())
  {
    throw no_such_group_exception(name);
  }

  priv::input_port_mapping_t const& mapping = group_it->second.first;

  priv::input_port_mapping_t::const_iterator const mapping_it = mapping.find(port);

  if (mapping_it == mapping.end())
  {
    throw no_such_group_port_exception(name, port);
  }

  return mapping_it->second.get<1>();
}

process::port_addr_t
pipeline
::mapped_group_output_port(process::name_t const& name, process::port_t const& port) const
{
  priv::group_t::const_iterator const group_it = d->groups.find(name);

  if (group_it == d->groups.end())
  {
    throw no_such_group_exception(name);
  }

  priv::output_port_mapping_t const& mapping = group_it->second.second;

  priv::output_port_mapping_t::const_iterator const mapping_it = mapping.find(port);

  if (mapping_it == mapping.end())
  {
    throw no_such_group_port_exception(name, port);
  }

  return mapping_it->second.get<1>();
}

pipeline::priv
::priv()
{
}

pipeline::priv
::~priv()
{
}

void
pipeline::priv
::check_duplicate_name(process::name_t const& name)
{
  process_map_t::const_iterator proc_it = process_map.find(name);
  group_t::const_iterator group_it = groups.find(name);

  if ((proc_it != process_map.end()) ||
      (group_it != groups.end()))
  {
    throw duplicate_process_name_exception(name);
  }
}

}
