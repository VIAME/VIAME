/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "pipeline.h"
#include "pipeline_exception.h"

#include "edge.h"
#include "process_exception.h"

#include <boost/graph/directed_graph.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/foreach.hpp>
#include <boost/make_shared.hpp>

#include <map>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include <cstddef>

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
    priv(pipeline* pipe);
    ~priv();

    void check_duplicate_name(process::name_t const& name);
    void propogate(process::name_t const& root);

    typedef std::map<process::name_t, process_t> process_map_t;
    typedef std::pair<process::port_addr_t, process::port_addr_t> connection_t;
    typedef std::vector<connection_t> connections_t;
    typedef std::map<size_t, edge_t> edge_map_t;

    typedef boost::tuple<process::port_flags_t, process::port_addrs_t> input_mapping_info_t;
    typedef boost::tuple<process::port_flags_t, process::port_addr_t> output_mapping_info_t;
    typedef std::map<process::port_t, input_mapping_info_t> input_port_mapping_t;
    typedef std::map<process::port_t, output_mapping_info_t> output_port_mapping_t;
    typedef std::pair<input_port_mapping_t, output_port_mapping_t> port_mapping_t;
    typedef std::map<process::name_t, port_mapping_t> group_map_t;

    typedef std::map<process::name_t, process::ports_t> connected_mappings_t;

    // Steps for checking a connection.
    bool check_connection_flags(process::port_flags_t const& up_flags, process::port_flags_t const& down_flags) const;

    // Steps for setting up the pipeline.
    void check_for_processes() const;
    void make_connections();
    void check_for_required_ports() const;
    void initialize_processes();
    void check_for_untyped_ports() const;

    pipeline* const q;

    connections_t connections;

    process_map_t process_map;
    edge_map_t edge_map;

    group_map_t groups;

    connected_mappings_t used_input_mappings;
    connected_mappings_t used_output_mappings;

    process::port_addrs_t data_dep_ports;
    connections_t untyped_connections;

    bool setup;
    bool setup_in_progress;
    bool setup_successful;

    class propogation_exception
      : public pipeline_exception
    {
      public:
        propogation_exception(process::name_t const& upstream_name,
                              process::port_t const& upstream_port,
                              process::name_t const& downstream_name,
                              process::port_t const& downstream_port,
                              process::port_type_t const& type,
                              bool push_upstream) throw();
        ~propogation_exception() throw();

        process::name_t const m_upstream_name;
        process::port_t const m_upstream_port;
        process::name_t const m_downstream_name;
        process::port_t const m_downstream_port;
        process::port_type_t const m_type;
        bool const m_push_upstream;
    };
  private:
    process::names_t sorted_names() const;
};

pipeline
::pipeline(config_t const& config)
  : d(new priv(this))
{
  if (!config)
  {
    throw null_pipeline_config_exception();
  }
}

pipeline
::~pipeline()
{
}

void
pipeline
::add_process(process_t const& process)
{
  if (!process)
  {
    throw null_process_addition_exception();
  }

  if (d->setup)
  {
    throw add_after_setup_exception(process->name(), true);
  }

  process::name_t const name = process->name();

  d->check_duplicate_name(name);

  d->process_map[name] = process;
}

void
pipeline
::add_group(process::name_t const& name)
{
  if (d->setup)
  {
    throw add_after_setup_exception(name, false);
  }

  d->check_duplicate_name(name);

  d->groups[name] = priv::port_mapping_t();
}

void
pipeline
::connect(process::name_t const& upstream_name,
          process::port_t const& upstream_port,
          process::name_t const& downstream_name,
          process::port_t const& downstream_port)
{
  if (d->setup && !d->setup_in_progress)
  {
    throw connection_after_setup_exception(upstream_name, upstream_port,
                                           downstream_name, downstream_port);
  }

  priv::group_map_t::const_iterator const up_group_it = d->groups.find(upstream_name);

  if (up_group_it != d->groups.end())
  {
    priv::port_mapping_t const& port_mapping = up_group_it->second;
    priv::output_port_mapping_t const& mapping = port_mapping.second;
    priv::output_port_mapping_t::const_iterator const mapping_it = mapping.find(upstream_port);

    if (mapping_it != mapping.end())
    {
      process::port_addr_t const& mapped_port_addr = mapping_it->second.get<1>();
      process::name_t const& proc_name = mapped_port_addr.first;
      process::port_t const& port_name = mapped_port_addr.second;

      connect(proc_name, port_name,
              downstream_name, downstream_port);

      d->used_output_mappings[upstream_name].push_back(upstream_port);

      return;
    }
  }

  priv::group_map_t::const_iterator const down_group_it = d->groups.find(downstream_name);

  if (down_group_it != d->groups.end())
  {
    priv::port_mapping_t const& port_mapping = down_group_it->second;
    priv::input_port_mapping_t const& mapping = port_mapping.first;
    priv::input_port_mapping_t::const_iterator const mapping_it = mapping.find(downstream_port);

    if (mapping_it != mapping.end())
    {
      process::port_addrs_t const& mapped_port_addrs = mapping_it->second.get<1>();

      BOOST_FOREACH (process::port_addr_t const& mapped_port_addr, mapped_port_addrs)
      {
        process::name_t const& proc_name = mapped_port_addr.first;
        process::port_t const& port_name = mapped_port_addr.second;

        connect(upstream_name, upstream_port,
                proc_name, port_name);
      }

      d->used_input_mappings[downstream_name].push_back(downstream_port);

      return;
    }
  }

  process::port_addr_t const up_port = process::port_addr_t(upstream_name, upstream_port);
  process::port_addr_t const down_port = process::port_addr_t(downstream_name, downstream_port);
  priv::connection_t const conn = priv::connection_t(up_port, down_port);

  process_t const up_proc = process_by_name(upstream_name);
  process_t const down_proc = process_by_name(downstream_name);

  process::port_info_t const up_info = up_proc->output_port_info(upstream_port);
  process::port_info_t const down_info = down_proc->input_port_info(downstream_port);

  process::port_type_t const& up_type = up_info->type;
  process::port_type_t const& down_type = down_info->type;

  bool const up_data_dep = (up_type == process::type_data_dependent);

  if (up_data_dep)
  {
    d->data_dep_ports.push_back(up_port);
  }

  bool const up_flow_dep = boost::starts_with(up_type, process::type_flow_dependent);
  bool const down_flow_dep = boost::starts_with(down_type, process::type_flow_dependent);

  if ((up_data_dep || up_flow_dep) && down_flow_dep)
  {
    d->untyped_connections.push_back(conn);
  }
  else if (up_data_dep || up_flow_dep)
  {
    if (!up_proc->set_output_port_type(upstream_port, down_type))
    {
      throw connection_dependent_type_exception(upstream_name, upstream_port,
                                                downstream_name, downstream_port,
                                                down_type, true);
    }

    try
    {
      d->propogate(upstream_name);
    }
    catch (priv::propogation_exception& e)
    {
      throw connection_dependent_type_cascade_exception(upstream_name, upstream_port, down_type,
                                                        e.m_upstream_name, e.m_upstream_port,
                                                        e.m_downstream_name, e.m_downstream_port,
                                                        e.m_type, e.m_push_upstream);
    }

    // Retry the connection.
    connect(upstream_name, upstream_port,
            downstream_name, downstream_port);

    return;
  }
  else if (down_flow_dep)
  {
    if (!down_proc->set_input_port_type(downstream_port, up_type))
    {
      throw connection_dependent_type_exception(upstream_name, upstream_port,
                                                downstream_name, downstream_port,
                                                up_type, false);
    }

    try
    {
      d->propogate(downstream_name);
    }
    catch (priv::propogation_exception& e)
    {
      throw connection_dependent_type_cascade_exception(downstream_name, downstream_port, up_type,
                                                        e.m_upstream_name, e.m_upstream_port,
                                                        e.m_downstream_name, e.m_downstream_port,
                                                        e.m_type, e.m_push_upstream);
    }

    // Retry the connection.
    connect(upstream_name, upstream_port,
            downstream_name, downstream_port);

    return;
  }
  else if ((up_type != process::type_any) &&
           (down_type != process::type_any) &&
           (up_type != down_type))
  {
    throw connection_type_mismatch_exception(upstream_name, upstream_port, up_type,
                                             downstream_name, downstream_port, down_type);
  }

  process::port_flags_t const& up_flags = up_info->flags;
  process::port_flags_t const& down_flags = down_info->flags;

  if (!d->check_connection_flags(up_flags, down_flags))
  {
    throw connection_flag_mismatch_exception(upstream_name, upstream_port,
                                             downstream_name, downstream_port);
  }

  d->connections.push_back(conn);
}

void
pipeline
::map_input_port(process::name_t const& group,
                 process::port_t const& port,
                 process::name_t const& mapped_name,
                 process::port_t const& mapped_port,
                 process::port_flags_t const& flags)
{
  if (d->setup)
  {
    throw connection_after_setup_exception(group, port,
                                           mapped_name, mapped_port);
  }

  priv::group_map_t::iterator const group_it = d->groups.find(group);

  if (group_it == d->groups.end())
  {
    throw no_such_group_exception(group);
  }

  priv::port_mapping_t& port_mapping = group_it->second;
  priv::input_port_mapping_t& mapping = port_mapping.first;

  priv::input_mapping_info_t& mapping_info = mapping[port];

  process::port_addr_t const mapped_port_addr = process::port_addr_t(mapped_name, mapped_port);

  mapping_info.get<0>().insert(flags.begin(), flags.end());
  mapping_info.get<1>().push_back(mapped_port_addr);
}

void
pipeline
::map_output_port(process::name_t const& group,
                  process::port_t const& port,
                  process::name_t const& mapped_name,
                  process::port_t const& mapped_port,
                  process::port_flags_t const& flags)
{
  if (d->setup)
  {
    throw connection_after_setup_exception(mapped_name, mapped_port,
                                           group, port);
  }

  priv::group_map_t::iterator const group_it = d->groups.find(group);

  if (group_it == d->groups.end())
  {
    throw no_such_group_exception(group);
  }

  priv::port_mapping_t& port_mapping = group_it->second;
  priv::output_port_mapping_t& mapping = port_mapping.second;

  priv::output_port_mapping_t::const_iterator const port_it = mapping.find(port);

  if (port_it != mapping.end())
  {
    priv::output_mapping_info_t const& output_info = port_it->second;
    process::port_addr_t const& prev_port_addr = output_info.get<1>();
    process::name_t const& prev_name = prev_port_addr.first;
    process::port_t const& prev_port = prev_port_addr.second;

    throw group_output_already_mapped_exception(group, port, prev_name, prev_port, mapped_name, mapped_port);
  }

  process::port_addr_t const mapped_port_addr = process::port_addr_t(mapped_name, mapped_port);
  priv::output_mapping_info_t const mapping_info = priv::output_mapping_info_t(flags, mapped_port_addr);

  mapping[port] = mapping_info;
}

void
pipeline
::setup_pipeline()
{
  if (d->setup)
  {
    throw pipeline_duplicate_setup_exception();
  }

  d->check_for_processes();

  // There's no turning back after this (processes are modified and may not be
  // able to be added/removed without compromising the checks after this point).
  d->setup = true;
  d->setup_in_progress = true;

  d->make_connections();
  d->check_for_required_ports();
  d->initialize_processes();
  d->check_for_untyped_ports();

  d->setup_in_progress = false;
  d->setup_successful = true;
}

bool
pipeline
::is_setup() const
{
  return d->setup;
}

bool
pipeline
::setup_successful() const
{
  return d->setup_successful;
}

process::names_t
pipeline
::process_names() const
{
  process::names_t names;

  BOOST_FOREACH (priv::process_map_t::value_type const& process_index, d->process_map)
  {
    process::name_t const& name = process_index.first;

    names.push_back(name);
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
    process::port_addr_t const& upstream_addr = connection.first;
    process::port_addr_t const& downstream_addr = connection.second;

    process::name_t const& upstream_name = upstream_addr.first;
    process::name_t const& downstream_name = downstream_addr.first;

    if (downstream_name == name)
    {
      names.insert(upstream_name);
    }
  }

  processes_t processes;

  BOOST_FOREACH (process::name_t const& process_name, names)
  {
    priv::process_map_t::const_iterator i = d->process_map.find(process_name);
    process_t const& process = i->second;

    processes.push_back(process);
  }

  return processes;
}

process_t
pipeline
::upstream_for_port(process::name_t const& name, process::port_t const& port) const
{
  BOOST_FOREACH (priv::connection_t const& connection, d->connections)
  {
    process::port_addr_t const& upstream_addr = connection.first;
    process::port_addr_t const& downstream_addr = connection.second;

    process::name_t const& upstream_name = upstream_addr.first;
    process::name_t const& downstream_name = downstream_addr.first;
    process::port_t const& downstream_port = downstream_addr.second;

    if ((downstream_name == name) &&
        (downstream_port == port))
    {
      priv::process_map_t::const_iterator const i = d->process_map.find(upstream_name);

      return i->second;
    }
  }

  return process_t();
}

processes_t
pipeline
::downstream_for_process(process::name_t const& name) const
{
  std::set<process::name_t> names;

  BOOST_FOREACH (priv::connection_t const& connection, d->connections)
  {
    process::port_addr_t const& upstream_addr = connection.first;
    process::port_addr_t const& downstream_addr = connection.second;

    process::name_t const& upstream_name = upstream_addr.first;
    process::name_t const& downstream_name = downstream_addr.first;

    if (upstream_name == name)
    {
      names.insert(downstream_name);
    }
  }

  processes_t processes;

  BOOST_FOREACH (process::name_t const& process_name, names)
  {
    priv::process_map_t::const_iterator const i = d->process_map.find(process_name);
    process_t const& process = i->second;

    processes.push_back(process);
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
    process::port_addr_t const& upstream_addr = connection.first;
    process::port_addr_t const& downstream_addr = connection.second;

    process::name_t const& upstream_name = upstream_addr.first;
    process::port_t const& upstream_port = upstream_addr.second;
    process::name_t const& downstream_name = downstream_addr.first;

    if ((upstream_name == name) &&
        (upstream_port == port))
    {
      names.insert(downstream_name);
    }
  }

  processes_t processes;

  BOOST_FOREACH (process::name_t const& process_name, names)
  {
    priv::process_map_t::const_iterator const i = d->process_map.find(process_name);
    process_t const& process = i->second;

    processes.push_back(process);
  }

  return processes;
}

process::port_addr_t
pipeline
::sender_for_port(process::name_t const& name, process::port_t const& port) const
{
  BOOST_FOREACH (priv::connection_t const& connection, d->connections)
  {
    process::port_addr_t const& upstream_addr = connection.first;
    process::port_addr_t const& downstream_addr = connection.second;

    process::name_t const& downstream_name = downstream_addr.first;
    process::port_t const& downstream_port = downstream_addr.second;

    if ((downstream_name == name) &&
        (downstream_port == port))
    {
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
    process::port_addr_t const& upstream_addr = connection.first;
    process::port_addr_t const& downstream_addr = connection.second;

    process::name_t const& upstream_name = upstream_addr.first;
    process::port_t const& upstream_port = upstream_addr.second;

    if ((upstream_name == name) &&
        (upstream_port == port))
    {
      port_addrs.push_back(downstream_addr);
    }
  }

  return port_addrs;
}

edge_t
pipeline
::edge_for_connection(process::name_t const& upstream_name,
                      process::port_t const& upstream_port,
                      process::name_t const& downstream_name,
                      process::port_t const& downstream_port) const
{
  for (size_t i = 0; i < d->connections.size(); ++i)
  {
    priv::connection_t const& connection = d->connections[i];

    process::port_addr_t const& up = connection.first;
    process::port_addr_t const& down = connection.second;

    process::name_t const& up_name = up.first;
    process::port_t const& up_port = up.second;

    process::name_t const& down_name = down.first;
    process::port_t const& down_port = down.second;

    if ((up_name == upstream_name) &&
        (up_port == upstream_port) &&
        (down_name == downstream_name) &&
        (down_port == downstream_port))
    {
      return d->edge_map[i];
    }
  }

  return edge_t();
}

edges_t
pipeline
::input_edges_for_process(process::name_t const& name) const
{
  edges_t edges;

  BOOST_FOREACH (priv::edge_map_t::value_type const& edge_index, d->edge_map)
  {
    size_t const& i = edge_index.first;
    edge_t const& edge = edge_index.second;

    priv::connection_t const connection = d->connections[i];

    process::port_addr_t const& downstream_addr = connection.second;

    process::name_t const& downstream_name = downstream_addr.first;

    if (downstream_name == name)
    {
      edges.push_back(edge);
    }
  }

  return edges;
}

edge_t
pipeline
::input_edge_for_port(process::name_t const& name, process::port_t const& port) const
{
  BOOST_FOREACH (priv::edge_map_t::value_type const& edge_index, d->edge_map)
  {
    size_t const& i = edge_index.first;
    edge_t const& edge = edge_index.second;

    priv::connection_t const connection = d->connections[i];

    process::port_addr_t const& downstream_addr = connection.second;

    process::name_t const& downstream_name = downstream_addr.first;
    process::port_t const& downstream_port = downstream_addr.second;

    if ((downstream_name == name) &&
        (downstream_port == port))
    {
      return edge;
    }
  }

  return edge_t();
}

edges_t
pipeline
::output_edges_for_process(process::name_t const& name) const
{
  edges_t edges;

  BOOST_FOREACH (priv::edge_map_t::value_type const& edge_index, d->edge_map)
  {
    size_t const& i = edge_index.first;
    edge_t const& edge = edge_index.second;

    priv::connection_t const connection = d->connections[i];

    process::port_addr_t const& upstream_addr = connection.first;

    process::name_t const& upstream_name = upstream_addr.first;

    if (upstream_name == name)
    {
      edges.push_back(edge);
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
    size_t const& i = edge_index.first;
    edge_t const& edge = edge_index.second;

    priv::connection_t const connection = d->connections[i];

    process::port_addr_t const& upstream_addr = connection.first;

    process::name_t const& upstream_name = upstream_addr.first;
    process::port_t const& upstream_port = upstream_addr.second;

    if ((upstream_name == name) &&
        (upstream_port == port))
    {
      edges.push_back(edge);
    }
  }

  return edges;
}

process::names_t
pipeline
::groups() const
{
  process::names_t names;

  BOOST_FOREACH (priv::group_map_t::value_type const& group, d->groups)
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

  priv::group_map_t::const_iterator const group_it = d->groups.find(name);

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

  priv::group_map_t::const_iterator const group_it = d->groups.find(name);

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
  priv::group_map_t::const_iterator const group_it = d->groups.find(name);

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
  priv::group_map_t::const_iterator const group_it = d->groups.find(name);

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
  priv::group_map_t::const_iterator const group_it = d->groups.find(name);

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
  priv::group_map_t::const_iterator const group_it = d->groups.find(name);

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
::priv(pipeline* pipe)
  : q(pipe)
  , setup(false)
  , setup_in_progress(false)
  , setup_successful(false)
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
  process_map_t::const_iterator const proc_it = process_map.find(name);
  group_map_t::const_iterator const group_it = groups.find(name);

  if ((proc_it != process_map.end()) ||
      (group_it != groups.end()))
  {
    throw duplicate_process_name_exception(name);
  }
}

bool
pipeline::priv
::check_connection_flags(process::port_flags_t const& up_flags, process::port_flags_t const& down_flags) const
{
  process::port_flags_t::const_iterator i;

  i = up_flags.find(process::flag_output_const);

  bool const is_const = (i != up_flags.end());

  i = down_flags.find(process::flag_input_mutable);

  bool const requires_mutable = (i != down_flags.end());

  if (is_const && requires_mutable)
  {
    return false;
  }

  return true;
}

void
pipeline::priv
::propogate(process::name_t const& root)
{
  std::queue<process::name_t> kyu;

  kyu.push(root);

  while (!kyu.empty())
  {
    process::name_t const name = kyu.front();
    kyu.pop();

    process_t const proc = q->process_by_name(name);

    connections_t unresolved_connections;

    BOOST_FOREACH (connection_t const& connection, untyped_connections)
    {
      process::port_addr_t const& upstream_addr = connection.first;
      process::port_addr_t const& downstream_addr = connection.second;
      process::name_t const& upstream_name = upstream_addr.first;
      process::port_t const& upstream_port = upstream_addr.second;
      process::name_t const& downstream_name = downstream_addr.first;
      process::port_t const& downstream_port = downstream_addr.second;

      if (downstream_name == name)
      {
        // Push up.
        process::port_info_t const info = proc->input_port_info(downstream_port);
        process::port_type_t const& type = info->type;

        bool const data_dep = (type == process::type_data_dependent);
        bool const flow_dep = boost::starts_with(type, process::type_flow_dependent);

        if (!data_dep && !flow_dep)
        {
          process_t const up_proc = q->process_by_name(upstream_name);

          if (up_proc->set_output_port_type(upstream_port, type))
          {
            kyu.push(upstream_name);

            if (data_dep)
            {
              process::port_addrs_t::iterator const i = std::find(data_dep_ports.begin(), data_dep_ports.end(), upstream_addr);

              if (i == data_dep_ports.end())
              {
                static std::string const reason = "Data dependency port tracking failed.";

                throw std::logic_error(reason);
              }

              data_dep_ports.erase(i);
            }
          }
          else
          {
            throw propogation_exception(upstream_name, upstream_port,
                                        downstream_name, downstream_port,
                                        type, true);
          }
        }
      }
      else if (upstream_name == name)
      {
        // Push down.
        process::port_info_t const info = proc->output_port_info(upstream_port);
        process::port_type_t const& type = info->type;

        bool const flow_dep = boost::starts_with(type, process::type_flow_dependent);

        if (!flow_dep)
        {
          process_t const down_proc = q->process_by_name(downstream_name);

          if (down_proc->set_input_port_type(downstream_port, type))
          {
            kyu.push(downstream_name);
          }
          else
          {
            throw propogation_exception(upstream_name, upstream_port,
                                        downstream_name, downstream_port,
                                        type, false);
          }
        }
      }
      else
      {
        // Remember that the push didn't happen.
        unresolved_connections.push_back(connection);
      }
    }

    // Overwrite untyped connections.
    untyped_connections = unresolved_connections;
  }
}

void
pipeline::priv
::check_for_processes() const
{
  if (!process_map.size())
  {
    throw no_processes_exception();
  }
}

void
pipeline::priv
::make_connections()
{
  size_t const len = connections.size();

  for (size_t i = 0; i < len; ++i)
  {
    connection_t const& connection = connections[i];

    process::port_addr_t const& upstream_addr = connection.first;
    process::port_addr_t const& downstream_addr = connection.second;

    process::name_t const& upstream_name = upstream_addr.first;
    process::port_t const& upstream_port = upstream_addr.second;
    process::name_t const& downstream_name = downstream_addr.first;
    process::port_t const& downstream_port = downstream_addr.second;

    process_t const up_proc = q->process_by_name(upstream_name);
    process_t const down_proc = q->process_by_name(downstream_name);

    process::port_info_t const down_info = down_proc->input_port_info(downstream_port);
    process::port_flags_t const& down_flags = down_info->flags;

    config_t edge_config = config::empty_config();

    // Configure the edge.
    {
      process::port_flags_t::const_iterator const it = down_flags.find(process::flag_input_nodep);

      bool const has_nodep = (it != down_flags.end());

      edge_config->set_value(edge::config_dependency, (has_nodep ? "false" : "true"));
    }

    edge_t e = boost::make_shared<edge>(edge_config);

    edge_map[i] = e;

    up_proc->connect_output_port(upstream_port, e);
    down_proc->connect_input_port(downstream_port, e);

    e->set_upstream_process(up_proc);
    e->set_downstream_process(down_proc);
  }
}

void
pipeline::priv
::check_for_required_ports() const
{
  typedef std::set<process::name_t> name_set_t;
  typedef std::queue<process::name_t> name_queue_t;

  name_set_t procs;

  {
    name_queue_t to_visit;

    // Traverse the pipeline starting with a process.
    to_visit.push(process_map.begin()->first);

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

      // Check for required ports.
      {
        process_t const process = q->process_by_name(cur_proc);

        // Check for required input ports.
        process::ports_t const input_ports = process->input_ports();
        BOOST_FOREACH (process::port_t const& port, input_ports)
        {
          // Check for required flags.
          process::port_flags_t const port_flags = process->input_port_info(port)->flags;

          process::port_flags_t::const_iterator const f = port_flags.find(process::flag_required);
          if (f != port_flags.end())
          {
            if (!q->input_edge_for_port(cur_proc, port))
            {
              static std::string const reason = "The input port has the required flag";

              throw missing_connection_exception(cur_proc, port, reason);
            }
          }
        }

        // Check for required output ports.
        process::ports_t const output_ports = process->output_ports();
        BOOST_FOREACH (process::port_t const& port, output_ports)
        {
          // Check for required flags.
          process::port_flags_t const port_flags = process->output_port_info(port)->flags;

          process::port_flags_t::const_iterator const f = port_flags.find(process::flag_required);
          if (f != port_flags.end())
          {
            if (q->output_edges_for_port(cur_proc, port).empty())
            {
              static std::string const reason = "The output port has the required flag";

              throw missing_connection_exception(cur_proc, port, reason);
            }
          }
        }
      }

      processes_t connected_procs;

      // Find all processes upstream of the current process.
      processes_t const upstream_procs = q->upstream_for_process(cur_proc);
      connected_procs.insert(connected_procs.end(), upstream_procs.begin(), upstream_procs.end());

      // Find all processes downstream of the current process.
      processes_t const downstream_procs = q->downstream_for_process(cur_proc);
      connected_procs.insert(connected_procs.end(), downstream_procs.begin(), downstream_procs.end());

      // Mark all connected processes for visitation.
      BOOST_FOREACH (process_t const& proc, connected_procs)
      {
        to_visit.push(proc->name());
      }
    }
  }

  if (groups.size())
  {
    process::names_t const group_names = q->groups();

    BOOST_FOREACH(process::name_t const& cur_group, group_names)
    {
      process::port_addrs_t connected_ports;

      // Get all processes input ports on the group map to.
      process::ports_t const input_ports = q->input_ports_for_group(cur_group);
      BOOST_FOREACH (process::port_t const& port, input_ports)
      {
        // Check for required flags.
        process::port_flags_t const mapped_port_flags = q->mapped_group_input_port_flags(cur_group, port);

        process::port_flags_t::const_iterator const i = mapped_port_flags.find(process::flag_required);
        if (i != mapped_port_flags.end())
        {
          connected_mappings_t const& conns = used_input_mappings;

          connected_mappings_t::const_iterator const c = conns.find(cur_group);

          if (c == conns.end())
          {
            static std::string const reason = "The input mapping has the required flag";

            throw missing_connection_exception(cur_group, port, reason);
          }
        }

        // Mark mapped ports as connected.
        process::port_addrs_t const mapped_ports = q->mapped_group_input_ports(cur_group, port);

        connected_ports.insert(connected_ports.end(), mapped_ports.begin(), mapped_ports.end());
      }

      // Get all processes output ports on the group map to.
      process::ports_t const output_ports = q->output_ports_for_group(cur_group);
      BOOST_FOREACH (process::port_t const& port, output_ports)
      {
        // Check for required flags.
        process::port_flags_t const mapped_port_flags = q->mapped_group_output_port_flags(cur_group, port);

        process::port_flags_t::const_iterator const i = mapped_port_flags.find(process::flag_required);
        if (i != mapped_port_flags.end())
        {
          connected_mappings_t const& conns = used_input_mappings;

          connected_mappings_t::const_iterator const c = conns.find(cur_group);

          if (c == conns.end())
          {
            static std::string const reason = "The output mapping has the required flag";

            throw missing_connection_exception(cur_group, port, reason);
          }
        }

        // Mark mapped ports as connected.
        process::port_addr_t const mapped_port = q->mapped_group_output_port(cur_group, port);

        connected_ports.push_back(mapped_port);
      }

      // Mark these processes as connected.
      BOOST_FOREACH (process::port_addr_t const& port_addr, connected_ports)
      {
        process::name_t const& name = port_addr.first;

        procs.insert(name);
      }
    }
  }

  if (procs.size() != process_map.size())
  {
    throw orphaned_processes_exception();
  }
}

void
pipeline::priv
::initialize_processes()
{
  process::names_t const names = sorted_names();

  // Initialize processes.
  BOOST_FOREACH (process::name_t const& name, names)
  {
    process_t const proc = q->process_by_name(name);

    proc->init();

    bool resolved_types = false;

    BOOST_FOREACH (process::port_addr_t const& data_dep_port, data_dep_ports)
    {
      process::name_t const& data_name = data_dep_port.first;
      process::port_t const& data_port = data_dep_port.second;

      if (name == data_name)
      {
        process::port_info_t const info = proc->output_port_info(data_port);

        if (info->type == process::type_data_dependent)
        {
          throw untyped_data_dependent_exception(data_name, data_port);
        }

        resolved_types = true;
      }
    }

    if (resolved_types)
    {
      try
      {
        propogate(name);
      }
      catch (propogation_exception& e)
      {
        throw connection_dependent_type_cascade_exception(name, "<data-dependent ports>", "<data-dependent types>",
                                                          e.m_upstream_name, e.m_upstream_port,
                                                          e.m_downstream_name, e.m_downstream_port,
                                                          e.m_type, e.m_push_upstream);
      }
    }
  }
}

void
pipeline::priv
::check_for_untyped_ports() const
{
  if (untyped_connections.size())
  {
    throw untyped_connection_exception();
  }
}

process::names_t
pipeline::priv
::sorted_names() const
{
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, process::name_t> pipeline_graph_t;
  typedef boost::graph_traits<pipeline_graph_t>::vertex_descriptor vertex_t;
  typedef std::deque<vertex_t> vertices_t;
  typedef std::map<process::name_t, vertex_t> vertex_map_t;

  pipeline_graph_t graph;

  // Create the graph.
  {
    vertex_map_t vertex_map;

    process::names_t const names = q->process_names();

    BOOST_FOREACH (process::name_t const& name, names)
    {
      vertex_t s = boost::add_vertex(graph);
      graph[s] = name;
      vertex_map[name] = s;
    }

    BOOST_FOREACH (process::name_t const& name, names)
    {
      process_t const proc = q->process_by_name(name);
      process::ports_t const iports = proc->input_ports();

      vertex_t const t = vertex_map[name];

      BOOST_FOREACH (process::port_t const& port, iports)
      {
        process::port_addr_t const sender = q->sender_for_port(name, port);
        process::name_t const& sender_name = sender.first;
        process::port_t const& sender_port = sender.second;
        edge_t const edge = q->edge_for_connection(sender_name, sender_port,
                                                   name, port);

        if (edge && edge->makes_dependency())
        {
          vertex_t const s = vertex_map[sender_name];

          boost::add_edge(s, t, graph);
        }
      }
    }
  }

  vertices_t vertices;

  try
  {
    boost::topological_sort(graph, std::front_inserter(vertices));
  }
  catch (boost::not_a_dag&)
  {
    throw not_a_dag_exception();
  }

  process::names_t names;

  BOOST_FOREACH (vertex_t const& vertex, vertices)
  {
    names.push_back(graph[vertex]);
  }

  return names;
}

pipeline::priv::propogation_exception
::propogation_exception(process::name_t const& upstream_name,
                        process::port_t const& upstream_port,
                        process::name_t const& downstream_name,
                        process::port_t const& downstream_port,
                        process::port_type_t const& type,
                        bool push_upstream) throw()
  : m_upstream_name(upstream_name)
  , m_upstream_port(upstream_port)
  , m_downstream_name(downstream_name)
  , m_downstream_port(downstream_port)
  , m_type(type)
  , m_push_upstream(push_upstream)
{
  m_what = "<internal>";
}

pipeline::priv::propogation_exception
::~propogation_exception() throw()
{
}

}
