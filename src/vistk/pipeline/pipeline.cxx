/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "pipeline.h"
#include "pipeline_exception.h"

#include "edge.h"
#include "process_exception.h"

#include <boost/algorithm/string/predicate.hpp>
#include <boost/graph/directed_graph.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/math/common_factor_rt.hpp>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/function.hpp>
#include <boost/make_shared.hpp>

#include <functional>
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
    priv(pipeline* pipe, config_t conf);
    ~priv();

    void check_duplicate_name(process::name_t const& name);
    void remove_from_pipeline(process::name_t const& name);
    void remove_group_input_port(process::name_t const& name, process::port_t const& port);
    void remove_group_output_port(process::name_t const& name, process::port_t const& port);
    void propagate(process::name_t const& root);

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

    typedef enum
    {
      group_upstream,
      group_downstream
    } group_connection_type_t;
    typedef std::pair<connection_t, group_connection_type_t> group_connection_t;
    typedef std::vector<group_connection_t> group_connections_t;

    typedef enum
    {
      push_upstream,
      push_downstream
    } direction_t;
    typedef std::pair<connection_t, direction_t> type_pinning_t;
    typedef std::vector<type_pinning_t> type_pinnings_t;

    typedef enum
    {
      type_deferred,
      type_mismatch,
      type_compatible
    } port_type_status;

    // Steps for checking a connection.
    port_type_status check_connection_types(connection_t const& connection, process::port_type_t const& up_type, process::port_type_t const& down_type);
    bool check_connection_flags(process::port_flags_t const& up_flags, process::port_flags_t const& down_flags) const;

    // Steps for setting up the pipeline.
    void check_for_processes() const;
    void map_group_connections();
    void configure_processes();
    void check_for_data_dep_ports() const;
    void propagate_pinned_types();
    void check_for_untyped_ports() const;
    void make_connections();
    void check_for_required_ports() const;
    void check_for_dag() const;
    void initialize_processes();
    void check_port_frequencies() const;

    void ensure_setup() const;

    pipeline* const q;
    config_t const config;

    connections_t planned_connections;
    connections_t connections;

    process_map_t process_map;
    edge_map_t edge_map;

    group_map_t groups;

    connected_mappings_t used_input_mappings;
    connected_mappings_t used_output_mappings;

    connections_t data_dep_connections;
    group_connections_t group_connections;
    connections_t untyped_connections;
    type_pinnings_t type_pinnings;

    bool setup;
    bool setup_in_progress;
    bool setup_successful;
    bool running;

    static bool is_upstream_for(process::port_addr_t const& addr, connection_t const& connection);
    static bool is_downstream_for(process::port_addr_t const& addr, connection_t const& connection);
    static bool is_group_upstream_for(process::port_addr_t const& addr, group_connection_t const& gconnection);
    static bool is_group_downstream_for(process::port_addr_t const& addr, group_connection_t const& gconnection);
    static bool is_addr_on(process::name_t const& name, process::port_addr_t const& addr);
    static bool is_connection_with(process::name_t const& name, connection_t const& connection);
    static bool is_group_connection_with(process::name_t const& name, group_connection_t const& gconnection);
    static bool is_group_connection_for(connection_t const& connection, group_connection_t const& gconnection);

    static process::port_t const port_sep;
    static config::key_t const config_edge;
    static config::key_t const config_edge_type;
    static config::key_t const config_edge_conn;

    class propagation_exception
      : public pipeline_exception
    {
      public:
        propagation_exception(process::name_t const& upstream_name,
                              process::port_t const& upstream_port,
                              process::name_t const& downstream_name,
                              process::port_t const& downstream_port,
                              process::port_type_t const& type,
                              bool push_upstream) throw();
        ~propagation_exception() throw();

        process::name_t const m_upstream_name;
        process::port_t const m_upstream_port;
        process::name_t const m_downstream_name;
        process::port_t const m_downstream_port;
        process::port_type_t const m_type;
        bool const m_push_upstream;
    };
};

process::port_t const pipeline::priv::port_sep = config::key_t(".");
config::key_t const pipeline::priv::config_edge = config::key_t("_edge");
config::key_t const pipeline::priv::config_edge_type = config::key_t("_edge_by_type");
config::key_t const pipeline::priv::config_edge_conn = config::key_t("_edge_by_conn");

pipeline
::pipeline(config_t const& config)
{
  d.reset(new priv(this, config));

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
::remove_process(process::name_t const& name)
{
  if (d->setup)
  {
    throw remove_after_setup_exception(name, true);
  }

  priv::process_map_t::const_iterator const i = d->process_map.find(name);

  if (i == d->process_map.end())
  {
    throw no_such_process_exception(name);
  }

  d->process_map.erase(name);

  d->remove_from_pipeline(name);
}

void
pipeline
::remove_group(process::name_t const& name)
{
  if (d->setup)
  {
    throw remove_after_setup_exception(name, false);
  }

  priv::group_map_t::const_iterator const i = d->groups.find(name);

  if (i == d->groups.end())
  {
    throw no_such_process_exception(name);
  }

  d->groups.erase(name);

  d->remove_from_pipeline(name);
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

  process::port_addr_t const up_addr = process::port_addr_t(upstream_name, upstream_port);
  process::port_addr_t const down_addr = process::port_addr_t(downstream_name, downstream_port);
  priv::connection_t const connection = priv::connection_t(up_addr, down_addr);

  if (!d->setup_in_progress)
  {
    d->planned_connections.push_back(connection);
  }

  priv::group_map_t::const_iterator const up_group_it = d->groups.find(upstream_name);
  priv::group_map_t::const_iterator const down_group_it = d->groups.find(downstream_name);

  bool const upstream_is_group = (up_group_it != d->groups.end());
  bool const downstream_is_group = (down_group_it != d->groups.end());

  if (upstream_is_group || downstream_is_group)
  {
    if (upstream_is_group)
    {
      d->group_connections.push_back(priv::group_connection_t(connection, priv::group_upstream));
    }
    else if (downstream_is_group)
    {
      d->group_connections.push_back(priv::group_connection_t(connection, priv::group_downstream));
    }

    return;
  }

  process_t const up_proc = process_by_name(upstream_name);
  process_t const down_proc = process_by_name(downstream_name);

  process::port_info_t const up_info = up_proc->output_port_info(upstream_port);
  process::port_info_t const down_info = down_proc->input_port_info(downstream_port);

  process::port_flags_t const& up_flags = up_info->flags;
  process::port_flags_t const& down_flags = down_info->flags;

  if (!d->check_connection_flags(up_flags, down_flags))
  {
    throw connection_flag_mismatch_exception(upstream_name, upstream_port,
                                             downstream_name, downstream_port);
  }

  process::port_type_t const& up_type = up_info->type;
  process::port_type_t const& down_type = down_info->type;

  switch (d->check_connection_types(connection, up_type, down_type))
  {
    case priv::type_deferred:
      return;
    case priv::type_mismatch:
      throw connection_type_mismatch_exception(upstream_name, upstream_port, up_type,
                                               downstream_name, downstream_port, down_type);
    case priv::type_compatible:
    default:
      break;
  }

  d->connections.push_back(connection);
}

void
pipeline
::disconnect(process::name_t const& upstream_name,
             process::port_t const& upstream_port,
             process::name_t const& downstream_name,
             process::port_t const& downstream_port)
{
  if (d->setup)
  {
    throw disconnection_after_setup_exception(upstream_name, upstream_port,
                                              downstream_name, downstream_port);
  }

  process::port_addr_t const upstream_addr = process::port_addr_t(upstream_name, upstream_port);
  process::port_addr_t const downstream_addr = process::port_addr_t(downstream_name, downstream_port);
  priv::connection_t const conn = priv::connection_t(upstream_addr, downstream_addr);

  boost::function<bool (priv::connection_t const&)> const eq = boost::bind(std::equal_to<priv::connection_t>(), conn, _1);
  boost::function<bool (priv::group_connection_t const&)> const group_eq = boost::bind(&priv::is_group_connection_for, conn, _1);

#define FORGET_CONNECTION(T, f, conns)                                   \
  do                                                                     \
  {                                                                      \
    T::iterator const i = std::remove_if(conns.begin(), conns.end(), f); \
    conns.erase(i, conns.end());                                         \
  } while (false)

  FORGET_CONNECTION(priv::connections_t, eq, d->planned_connections);
  FORGET_CONNECTION(priv::connections_t, eq, d->connections);
  FORGET_CONNECTION(priv::connections_t, eq, d->data_dep_connections);
  FORGET_CONNECTION(priv::connections_t, eq, d->untyped_connections);
  FORGET_CONNECTION(priv::group_connections_t, group_eq, d->group_connections);

#undef FORGET_CONNECTION
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
::unmap_input_port(process::name_t const& group,
                   process::port_t const& port,
                   process::name_t const& mapped_name,
                   process::port_t const& mapped_port)
{
  if (d->setup)
  {
    throw disconnection_after_setup_exception(group, port,
                                              mapped_name, mapped_port);
  }

  priv::group_map_t::iterator const group_it = d->groups.find(group);

  if (group_it == d->groups.end())
  {
    throw no_such_group_exception(group);
  }

  priv::port_mapping_t& port_mapping = group_it->second;
  priv::input_port_mapping_t& imapping = port_mapping.first;

  priv::input_port_mapping_t::iterator const map_i = imapping.find(port);

  if (map_i == imapping.end())
  {
    throw no_such_group_port_exception(group, port);
  }

  priv::input_mapping_info_t& mapping_info = map_i->second;
  process::port_addrs_t& mappings = mapping_info.get<1>();

  process::port_addr_t const mapped_addr = process::port_addr_t(mapped_name, mapped_port);

  process::port_addrs_t::iterator const i = std::remove(mappings.begin(), mappings.end(),
                                                        mapped_addr);
  mappings.erase(i, mappings.end());

  if (!mappings.size())
  {
    imapping.erase(port);

    d->remove_group_input_port(group, port);
  }
}

void
pipeline
::unmap_output_port(process::name_t const& group,
                    process::port_t const& port,
                    process::name_t const& mapped_name,
                    process::port_t const& mapped_port)
{
  if (d->setup)
  {
    throw disconnection_after_setup_exception(mapped_name, mapped_port,
                                              group, port);
  }

  priv::group_map_t::iterator const group_it = d->groups.find(group);

  if (group_it == d->groups.end())
  {
    throw no_such_group_exception(group);
  }

  priv::port_mapping_t& port_mapping = group_it->second;
  priv::output_port_mapping_t& omapping = port_mapping.second;

  priv::output_port_mapping_t::const_iterator const map_i = omapping.find(port);

  if (map_i == omapping.end())
  {
    throw no_such_group_port_exception(group, port);
  }

  priv::output_mapping_info_t const& mapping_info = map_i->second;
  process::port_addr_t const& mapping = mapping_info.get<1>();

  process::port_addr_t const mapped_addr = process::port_addr_t(mapped_name, mapped_port);

  if (mapping == mapped_addr)
  {
    omapping.erase(port);

    d->remove_group_output_port(group, port);
  }
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
  d->setup_successful = false;

  try
  {
    d->map_group_connections();
    d->configure_processes();
    d->check_for_data_dep_ports();
    d->propagate_pinned_types();
    d->check_for_untyped_ports();
    d->make_connections();
    d->check_for_required_ports();
    d->check_for_dag();
    d->initialize_processes();
    d->check_port_frequencies();
  }
  catch (...)
  {
    d->setup_in_progress = false;
    throw;
  }

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

void
pipeline
::reset()
{
  if (d->running)
  {
    throw reset_running_pipeline_exception();
  }

  d->setup = false;
  d->setup_successful = false;

  priv::process_map_t const names = d->process_map;

  // Reset all the processes.
  BOOST_FOREACH (priv::process_map_t::value_type& process_entry, d->process_map)
  {
    process_t const& process = process_entry.second;

    process->reset();
  }

  // Clear internal bookkeeping.
  d->connections.clear();
  d->edge_map.clear();
  d->used_input_mappings.clear();
  d->used_output_mappings.clear();
  d->data_dep_connections.clear();
  d->group_connections.clear();
  d->untyped_connections.clear();
  d->type_pinnings.clear();

  d->setup_in_progress = true;

  // Replay connections.
  BOOST_FOREACH (priv::connection_t const& connection, d->planned_connections)
  {
    process::port_addr_t const& upstream_addr = connection.first;
    process::port_addr_t const& downstream_addr = connection.second;

    process::name_t const& upstream_name = upstream_addr.first;
    process::port_t const& upstream_port = upstream_addr.second;
    process::name_t const& downstream_name = downstream_addr.first;
    process::port_t const& downstream_port = downstream_addr.second;

    connect(upstream_name, upstream_port,
            downstream_name, downstream_port);
  }

  d->setup_in_progress = false;
}

void
pipeline
::start()
{
  d->ensure_setup();

  d->running = true;
}

void
pipeline
::stop()
{
  if (!d->running)
  {
    throw pipeline_not_running_exception();
  }

  d->running = false;
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

process::port_addrs_t
pipeline
::connections_from_addr(process::name_t const& name, process::port_t const& port) const
{
  process::port_addrs_t addrs;

  BOOST_FOREACH (priv::connection_t const& connection, d->planned_connections)
  {
    process::port_addr_t const& upstream_addr = connection.first;
    process::port_addr_t const& downstream_addr = connection.second;

    process::name_t const& upstream_name = upstream_addr.first;
    process::port_t const& upstream_port = upstream_addr.second;

    if ((upstream_name == name) && (upstream_port == port))
    {
      addrs.push_back(downstream_addr);
    }
  }

  return addrs;
}

process::port_addr_t
pipeline
::connection_to_addr(process::name_t const& name, process::port_t const& port) const
{
  BOOST_FOREACH (priv::connection_t const& connection, d->planned_connections)
  {
    process::port_addr_t const& upstream_addr = connection.first;
    process::port_addr_t const& downstream_addr = connection.second;

    process::name_t const& downstream_name = downstream_addr.first;
    process::port_t const& downstream_port = downstream_addr.second;

    if ((downstream_name == name) && (downstream_port == port))
    {
      return upstream_addr;
    }
  }

  return process::port_addr_t();
}

processes_t
pipeline
::upstream_for_process(process::name_t const& name) const
{
  d->ensure_setup();

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
  d->ensure_setup();

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
  d->ensure_setup();

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
  d->ensure_setup();

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
  d->ensure_setup();

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
  d->ensure_setup();

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
  d->ensure_setup();

  for (size_t i = 0; i < d->connections.size(); ++i)
  {
    priv::connection_t const& connection = d->connections[i];

    process::port_addr_t const& upstream_addr = connection.first;
    process::port_addr_t const& downstream_addr = connection.second;

    process::name_t const& up_name = upstream_addr.first;
    process::port_t const& up_port = upstream_addr.second;
    process::name_t const& down_name = downstream_addr.first;
    process::port_t const& down_port = downstream_addr.second;

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
  d->ensure_setup();

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
  d->ensure_setup();

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
  d->ensure_setup();

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
  d->ensure_setup();

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
::priv(pipeline* pipe, config_t conf)
  : q(pipe)
  , config(conf)
  , setup(false)
  , setup_in_progress(false)
  , setup_successful(false)
  , running(false)
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

void
pipeline::priv
::remove_from_pipeline(process::name_t const& name)
{
  boost::function<bool (connection_t const&)> const is = boost::bind(&is_connection_with, name, _1);
  boost::function<bool (group_connection_t const&)> const group_is = boost::bind(&is_group_connection_with, name, _1);

#define FORGET_CONNECTIONS(T, f, conns)                                  \
  do                                                                     \
  {                                                                      \
    T::iterator const i = std::remove_if(conns.begin(), conns.end(), f); \
    conns.erase(i, conns.end());                                         \
  } while (false)

  FORGET_CONNECTIONS(connections_t, is, planned_connections);
  FORGET_CONNECTIONS(connections_t, is, connections);
  FORGET_CONNECTIONS(connections_t, is, data_dep_connections);
  FORGET_CONNECTIONS(connections_t, is, untyped_connections);
  FORGET_CONNECTIONS(group_connections_t, group_is, group_connections);

#undef FORGET_CONNECTIONS

  BOOST_FOREACH (group_map_t::value_type& group, groups)
  {
    process::name_t const& group_name = group.first;
    port_mapping_t& port_mapping = group.second;

    input_port_mapping_t& input_mappings = port_mapping.first;
    input_port_mapping_t::iterator in = input_mappings.begin();
    input_port_mapping_t::iterator const in_end = input_mappings.end();

    while (in != in_end)
    {
      input_port_mapping_t::value_type& input_mapping = *in;

      process::port_t const& port = input_mapping.first;
      input_mapping_info_t& info = input_mapping.second;

      process::port_addrs_t& mappings = info.get<1>();

      process::port_addrs_t::iterator const i = std::remove_if(mappings.begin(), mappings.end(),
                                                               boost::bind(is_addr_on, name, _1));
      mappings.erase(i, mappings.end());

      if (!mappings.size())
      {
        process::port_t const port_save = port;

        input_mappings.erase(in++);

        remove_group_input_port(group_name, port_save);
      }
      else
      {
        ++in;
      }
    }

    output_port_mapping_t& output_mappings = port_mapping.second;
    output_port_mapping_t::iterator out = output_mappings.begin();
    output_port_mapping_t::iterator const out_end = output_mappings.end();

    while (out != out_end)
    {
      output_port_mapping_t::value_type& output_mapping = *out;

      process::port_t const& port = output_mapping.first;
      output_mapping_info_t& info = output_mapping.second;

      process::port_addr_t& mapping = info.get<1>();

      if (!is_addr_on(name, mapping))
      {
        process::port_t const port_save = port;

        output_mappings.erase(out++);

        remove_group_input_port(group_name, port_save);
      }
      else
      {
        ++out;
      }
    }
  }
}

void
pipeline::priv
::remove_group_input_port(process::name_t const& name, process::port_t const& port)
{
  process::port_addr_t const addr = process::port_addr_t(name, port);

  boost::function<bool (connection_t const&)> const down = boost::bind(&is_downstream_for, addr, _1);
  boost::function<bool (group_connection_t const&)> const group_down = boost::bind(&is_group_downstream_for, addr, _1);

#define FORGET_PORT(T, f, conns)                                         \
  do                                                                     \
  {                                                                      \
    T::iterator const i = std::remove_if(conns.begin(), conns.end(), f); \
    conns.erase(i, conns.end());                                         \
  } while (false)

    FORGET_PORT(priv::connections_t, down, planned_connections);
    FORGET_PORT(priv::connections_t, down, connections);
    FORGET_PORT(priv::connections_t, down, data_dep_connections);
    FORGET_PORT(priv::connections_t, down, untyped_connections);
    FORGET_PORT(priv::group_connections_t, group_down, group_connections);

#undef FORGET_PORT
}

void
pipeline::priv
::remove_group_output_port(process::name_t const& name, process::port_t const& port)
{
  process::port_addr_t const addr = process::port_addr_t(name, port);

  boost::function<bool (connection_t const&)> const up = boost::bind(&is_upstream_for, addr, _1);
  boost::function<bool (group_connection_t const&)> const group_up = boost::bind(&is_group_upstream_for, addr, _1);

#define FORGET_PORT(T, f, conns)                                         \
  do                                                                     \
  {                                                                      \
    T::iterator const i = std::remove_if(conns.begin(), conns.end(), f); \
    conns.erase(i, conns.end());                                         \
  } while (false)

    FORGET_PORT(priv::connections_t, up, planned_connections);
    FORGET_PORT(priv::connections_t, up, connections);
    FORGET_PORT(priv::connections_t, up, data_dep_connections);
    FORGET_PORT(priv::connections_t, up, untyped_connections);
    FORGET_PORT(priv::group_connections_t, group_up, group_connections);

#undef FORGET_PORT
}

pipeline::priv::port_type_status
pipeline::priv
::check_connection_types(connection_t const& connection, process::port_type_t const& up_type, process::port_type_t const& down_type)
{
  bool const up_data_dep = (up_type == process::type_data_dependent);

  if (up_data_dep)
  {
    data_dep_connections.push_back(connection);

    return type_deferred;
  }

  bool const up_flow_dep = boost::starts_with(up_type, process::type_flow_dependent);
  bool const down_flow_dep = boost::starts_with(down_type, process::type_flow_dependent);

  if (up_flow_dep || down_flow_dep)
  {
    if (up_flow_dep && down_flow_dep)
    {
      untyped_connections.push_back(connection);
    }
    else if (up_flow_dep)
    {
      type_pinnings.push_back(priv::type_pinning_t(connection, priv::push_upstream));
    }
    else if (down_flow_dep)
    {
      type_pinnings.push_back(priv::type_pinning_t(connection, priv::push_downstream));
    }

    return type_deferred;
  }
  else if ((up_type != process::type_any) &&
           (down_type != process::type_any) &&
           (up_type != down_type))
  {
    return type_mismatch;
  }

  return type_compatible;
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
::propagate(process::name_t const& root)
{
  std::queue<process::name_t> kyu;

  kyu.push(root);

  while (!kyu.empty())
  {
    process::name_t const name = kyu.front();
    kyu.pop();

    process_t const proc = q->process_by_name(name);

    connections_t const conns = untyped_connections;
    untyped_connections.clear();

    BOOST_FOREACH (connection_t const& connection, conns)
    {
      process::port_addr_t const& upstream_addr = connection.first;
      process::port_addr_t const& downstream_addr = connection.second;
      process::name_t const& upstream_name = upstream_addr.first;
      process::port_t const& upstream_port = upstream_addr.second;
      process::name_t const& downstream_name = downstream_addr.first;
      process::port_t const& downstream_port = downstream_addr.second;

      bool resolved = false;

      if (downstream_name == name)
      {
        // Push up.
        process::port_info_t const info = proc->input_port_info(downstream_port);
        process::port_type_t const& type = info->type;

        bool const flow_dep = boost::starts_with(type, process::type_flow_dependent);

        if (!flow_dep)
        {
          process_t const up_proc = q->process_by_name(upstream_name);

          if (!up_proc->set_output_port_type(upstream_port, type))
          {
            throw propagation_exception(upstream_name, upstream_port,
                                        downstream_name, downstream_port,
                                        type, true);
          }

          resolved = true;

          q->connect(upstream_name, upstream_port,
                     downstream_name, downstream_port);

          kyu.push(upstream_name);
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

          if (!down_proc->set_input_port_type(downstream_port, type))
          {
            throw propagation_exception(upstream_name, upstream_port,
                                        downstream_name, downstream_port,
                                        type, false);
          }

          resolved = true;

          q->connect(upstream_name, upstream_port,
                     downstream_name, downstream_port);

          kyu.push(downstream_name);
        }
      }

      if (!resolved)
      {
        // Remember that the resolution didn't happen.
        untyped_connections.push_back(connection);
      }
    }
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
::map_group_connections()
{
  group_connections_t const gconnections = group_connections;

  // Forget the connections we'll be mapping.
  group_connections.clear();

  BOOST_FOREACH (group_connection_t const& gconnection, gconnections)
  {
    connection_t const& connection = gconnection.first;
    group_connection_type_t const& type = gconnection.second;

    process::port_addr_t const& upstream_addr = connection.first;
    process::port_addr_t const& downstream_addr = connection.second;

    process::name_t const& upstream_name = upstream_addr.first;
    process::port_t const& upstream_port = upstream_addr.second;
    process::name_t const& downstream_name = downstream_addr.first;
    process::port_t const& downstream_port = downstream_addr.second;

    switch (type)
    {
      case group_upstream:
        {
          process::name_t const& group_name = upstream_name;
          process::port_t const& group_port = upstream_port;

          group_map_t::const_iterator const group_it = groups.find(group_name);

          if (group_it == groups.end())
          {
            throw no_such_group_exception(group_name);
          }

          port_mapping_t const& port_mapping = group_it->second;
          output_port_mapping_t const& mapping = port_mapping.second;
          output_port_mapping_t::const_iterator const mapping_it = mapping.find(group_port);

          if (mapping_it == mapping.end())
          {
            throw no_such_group_port_exception(group_name, group_port);
          }

          output_mapping_info_t const& info = mapping_it->second;
          process::port_addr_t const& mapped_port_addr = info.get<1>();

          process::name_t const& mapped_name = mapped_port_addr.first;
          process::port_t const& mapped_port = mapped_port_addr.second;

          q->connect(mapped_name, mapped_port,
                     downstream_name, downstream_port);

          used_output_mappings[group_name].push_back(group_port);
        }

        break;
      case group_downstream:
        {
          process::name_t const& group_name = downstream_name;
          process::port_t const& group_port = downstream_port;

          group_map_t::const_iterator const group_it = groups.find(group_name);

          if (group_it == groups.end())
          {
            throw no_such_group_exception(group_name);
          }

          port_mapping_t const& port_mapping = group_it->second;
          input_port_mapping_t const& mapping = port_mapping.first;
          input_port_mapping_t::const_iterator const mapping_it = mapping.find(group_port);

          if (mapping_it == mapping.end())
          {
            throw no_such_group_port_exception(group_name, group_port);
          }

          input_mapping_info_t const& info = mapping_it->second;
          process::port_addrs_t const& mapped_port_addrs = info.get<1>();

          BOOST_FOREACH (process::port_addr_t const& mapped_port_addr, mapped_port_addrs)
          {
            process::name_t const& mapped_name = mapped_port_addr.first;
            process::port_t const& mapped_port = mapped_port_addr.second;

            q->connect(upstream_name, upstream_port,
                       mapped_name, mapped_port);
          }

          used_input_mappings[group_name].push_back(group_port);
        }

        break;
      default:
        break;
    }
  }

  // Group ports could be mapped to other group ports. We need to call again
  // until every group port has been resolved to a process.
  if (group_connections.size())
  {
    map_group_connections();
  }
}

void
pipeline::priv
::configure_processes()
{
  process::names_t const names = q->process_names();

  // Configure processes.
  BOOST_FOREACH (process::name_t const& name, names)
  {
    process_t const proc = q->process_by_name(name);
    connections_t unresolved_connections;

    proc->configure();

    bool resolved_types = false;

    BOOST_FOREACH (connection_t const& data_dep_connection, data_dep_connections)
    {
      process::port_addr_t const& data_addr = data_dep_connection.first;
      process::port_addr_t const& downstream_addr = data_dep_connection.second;

      process::name_t const& data_name = data_addr.first;
      process::port_t const& data_port = data_addr.second;
      process::name_t const& downstream_name = downstream_addr.first;
      process::port_t const& downstream_port = downstream_addr.second;

      if (name == data_name)
      {
        process::port_info_t const info = proc->output_port_info(data_port);

        if (info->type == process::type_data_dependent)
        {
          throw untyped_data_dependent_exception(data_name, data_port);
        }

        resolved_types = true;

        q->connect(data_name, data_port,
                   downstream_name, downstream_port);
      }
      else
      {
        unresolved_connections.push_back(data_dep_connection);
      }
    }

    if (resolved_types)
    {
      data_dep_connections = unresolved_connections;
    }
  }
}

void
pipeline::priv
::check_for_data_dep_ports() const
{
  if (data_dep_connections.size())
  {
    static std::string const reason = "Data dependency port tracking failed.";

    throw std::logic_error(reason);
  }
}

void
pipeline::priv
::propagate_pinned_types()
{
  type_pinnings_t const pinnings = type_pinnings;
  type_pinnings.clear();

  BOOST_FOREACH (type_pinning_t const& pinning, pinnings)
  {
    connection_t const& connection = pinning.first;
    direction_t const& direction = pinning.second;

    process::port_addr_t const& upstream_addr = connection.first;
    process::port_addr_t const& downstream_addr = connection.second;

    process::name_t const& upstream_name = upstream_addr.first;
    process::port_t const& upstream_port = upstream_addr.second;
    process::name_t const& downstream_name = downstream_addr.first;
    process::port_t const& downstream_port = downstream_addr.second;

    process_t const up_proc = q->process_by_name(upstream_name);
    process_t const down_proc = q->process_by_name(downstream_name);

    process::port_info_t const up_info = up_proc->output_port_info(upstream_port);
    process::port_info_t const down_info = down_proc->input_port_info(downstream_port);

    process::port_type_t const& up_type = up_info->type;
    process::port_type_t const& down_type = down_info->type;

    process::name_t name;
    process::port_t port;
    process::port_type_t type;

    switch (direction)
    {
      case push_upstream:
        if (!up_proc->set_output_port_type(upstream_port, down_type))
        {
          throw connection_dependent_type_exception(upstream_name, upstream_port,
                                                    downstream_name, downstream_port,
                                                    down_type, true);
        }

        name = upstream_name;
        port = upstream_port;
        type = down_type;

        break;
      case push_downstream:
        if (!down_proc->set_input_port_type(downstream_port, up_type))
        {
          throw connection_dependent_type_exception(upstream_name, upstream_port,
                                                    downstream_name, downstream_port,
                                                    up_type, false);
        }

        name = downstream_name;
        port = downstream_port;
        type = up_type;

        break;
      default:
        continue;
    }

    try
    {
      propagate(name);
    }
    catch (propagation_exception const& e)
    {
      throw connection_dependent_type_cascade_exception(name, port, type,
                                                        e.m_upstream_name, e.m_upstream_port,
                                                        e.m_downstream_name, e.m_downstream_port,
                                                        e.m_type, e.m_push_upstream);
    }

    // Retry the connection.
    q->connect(upstream_name, upstream_port,
               downstream_name, downstream_port);
  }

  if (type_pinnings.size())
  {
    propagate_pinned_types();
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

    config_t edge_config = config->subblock(priv::config_edge);

    // Configure the edge based on its type.
    {
      process::port_type_t const& down_type = down_info->type;
      config_t const type_config = config->subblock(priv::config_edge_type);
      config_t const edge_type_config = type_config->subblock(down_type);

      edge_config->merge_config(edge_type_config);
    }

    // Configure the edge based on what it is mapped to.
    {
      /// \todo Remember mappings.
    }

    // Configure the edge based on the connected ports.
    {
      config_t const conn_config = config->subblock(priv::config_edge_conn);
      config_t const up_config = conn_config->subblock(upstream_name + priv::port_sep + upstream_port);
      config_t const down_config = conn_config->subblock(downstream_name + priv::port_sep + downstream_port);

      edge_config->merge_config(up_config);
      edge_config->merge_config(down_config);
    }

    // Configure the edge.
    {
      process::port_flags_t::const_iterator const it = down_flags.find(process::flag_input_nodep);

      bool const has_nodep = (it != down_flags.end());

      edge_config->set_value(edge::config_dependency, (has_nodep ? "false" : "true"));
      edge_config->mark_read_only(edge::config_dependency);
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
::check_for_dag() const
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
      vertex_t const s = boost::add_vertex(graph);
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

        if (sender == process::port_addr_t())
        {
          continue;
        }

        process::name_t const& sender_name = sender.first;

        process::port_info_t const info = proc->input_port_info(port);
        process::port_flags_t const& flags = info->flags;

        process::port_flags_t::const_iterator const i = flags.find(process::flag_input_nodep);

        if (i != flags.end())
        {
          continue;
        }

        vertex_t const s = vertex_map[sender_name];

        boost::add_edge(s, t, graph);
      }
    }
  }

  vertices_t vertices;

  try
  {
    boost::topological_sort(graph, std::front_inserter(vertices));
  }
  catch (boost::not_a_dag const&)
  {
    throw not_a_dag_exception();
  }
}

void
pipeline::priv
::initialize_processes()
{
  process::names_t const names = q->process_names();

  // Initialize processes.
  BOOST_FOREACH (process::name_t const& name, names)
  {
    process_t const proc = q->process_by_name(name);

    proc->init();
  }
}

void
pipeline::priv
::check_port_frequencies() const
{
  static process::port_frequency_t const base_freq = process::port_frequency_t(1, 1);

  process::names_t const names = q->process_names();

  typedef std::map<process::name_t, process::port_frequency_t> process_frequency_map_t;

  process_frequency_map_t freq_map;

  std::queue<connection_t> unchecked_connections;

  BOOST_FOREACH (connection_t const& connection, connections)
  {
    unchecked_connections.push(connection);
  }

  while (!unchecked_connections.empty())
  {
    connection_t const connection = unchecked_connections.front();
    unchecked_connections.pop();

    process::port_addr_t const& upstream_addr = connection.first;
    process::port_addr_t const& downstream_addr = connection.second;

    process::name_t const& upstream_name = upstream_addr.first;
    process::port_t const& upstream_port = upstream_addr.second;
    process::name_t const& downstream_name = downstream_addr.first;
    process::port_t const& downstream_port = downstream_addr.second;

    process_t const up_proc = q->process_by_name(upstream_name);
    process::port_info_t const up_info = up_proc->output_port_info(upstream_port);
    process::port_frequency_t const& up_port_freq = up_info->frequency;

    process_t const down_proc = q->process_by_name(downstream_name);
    process::port_info_t const down_info = down_proc->input_port_info(downstream_port);
    process::port_frequency_t const& down_port_freq = down_info->frequency;

    if (!up_port_freq || !down_port_freq)
    {
      /// \todo Issue a warning that the edge frequency cannot be validated.

      continue;
    }

    process_frequency_map_t::const_iterator const i_up = freq_map.find(upstream_name);
    process_frequency_map_t::const_iterator const i_down = freq_map.find(downstream_name);
    process_frequency_map_t::const_iterator const i_end = freq_map.end();

    bool have_upstream = false;
    bool have_downstream = false;

    if ((i_up == i_end) &&
        (i_down == i_end))
    {
      if (!freq_map.size())
      {
        // Seed the frequency map at 1-to-1 based on the upstream process.
        freq_map[upstream_name] = base_freq;
        have_upstream = true;
      }
    }

    if (i_up != i_end)
    {
      have_upstream = true;
    }
    if (i_down != i_end)
    {
      have_downstream = true;
    }

    // Validate the connection.
    if (have_upstream && have_downstream)
    {
      process::port_frequency_t const up_proc_freq = freq_map[upstream_name];

      process::port_frequency_t const edge_freq = up_proc_freq * up_port_freq;
      process::port_frequency_t const expect_freq = edge_freq / down_port_freq;

      process::port_frequency_t const down_proc_freq = freq_map[downstream_name];

      if (down_proc_freq != expect_freq)
      {
        throw frequency_mismatch_exception(upstream_name, upstream_port, up_proc_freq,
                                           downstream_name, downstream_port, down_proc_freq);
      }
    }
    // Propagate the frequency downstream.
    else if (have_upstream)
    {
      process::port_frequency_t const up_proc_freq = freq_map[upstream_name];

      process::port_frequency_t const edge_freq = up_proc_freq * up_port_freq;
      process::port_frequency_t const expect_freq = edge_freq / down_port_freq;

      freq_map[downstream_name] = expect_freq;
    }
    // Propagate the frequency upstream.
    else if (have_downstream)
    {
      process::port_frequency_t const down_proc_freq = freq_map[downstream_name];

      process::port_frequency_t const edge_freq = down_proc_freq * down_port_freq;
      process::port_frequency_t const expect_freq = edge_freq / up_port_freq;

      freq_map[upstream_name] = expect_freq;
    }
    // Not part of the already-checked parts.
    else
    {
      unchecked_connections.push(connection);
    }
  }

  process::frequency_component_t freq_gcd = process::frequency_component_t(1);

  BOOST_FOREACH (process_frequency_map_t::value_type const& proc_freq, freq_map)
  {
    process::port_frequency_t const& freq = proc_freq.second;
    process::frequency_component_t const denom = freq.denominator();

    freq_gcd = boost::math::lcm(freq_gcd, denom);
  }

  BOOST_FOREACH (process_frequency_map_t::value_type const& proc_freq, freq_map)
  {
    process::name_t const& name = proc_freq.first;
    process::port_frequency_t const& freq = proc_freq.second;

    process::port_frequency_t const core_freq = freq_gcd * freq;

    process_t const proc = q->process_by_name(name);

    proc->set_core_frequency(core_freq);
  }
}

void
pipeline::priv
::ensure_setup() const
{
  if (!setup)
  {
    throw pipeline_not_setup_exception();
  }

  if (!setup_in_progress && !setup_successful)
  {
    throw pipeline_not_ready_exception();
  }
}

bool
pipeline::priv
::is_upstream_for(process::port_addr_t const& addr, connection_t const& connection)
{
  process::port_addr_t const up_addr = connection.first;

  return (addr == up_addr);
}

bool
pipeline::priv
::is_downstream_for(process::port_addr_t const& addr, connection_t const& connection)
{
  process::port_addr_t const down_addr = connection.second;

  return (addr == down_addr);
}

bool
pipeline::priv
::is_group_upstream_for(process::port_addr_t const& addr, group_connection_t const& gconnection)
{
  connection_t const connection = gconnection.first;

  return is_upstream_for(addr, connection);
}

bool
pipeline::priv
::is_group_downstream_for(process::port_addr_t const& addr, group_connection_t const& gconnection)
{
  connection_t const connection = gconnection.first;

  return is_downstream_for(addr, connection);
}

bool
pipeline::priv
::is_addr_on(process::name_t const& name, process::port_addr_t const& addr)
{
  process::name_t const& proc_name = addr.first;

  return (name == proc_name);
}

bool
pipeline::priv
::is_connection_with(process::name_t const& name, connection_t const& connection)
{
  process::port_addr_t const& upstream_addr = connection.first;
  process::port_addr_t const& downstream_addr = connection.second;

  return (is_addr_on(name, upstream_addr) || is_addr_on(name, downstream_addr));
}

bool
pipeline::priv
::is_group_connection_with(process::name_t const& name, group_connection_t const& gconnection)
{
  connection_t const& connection = gconnection.first;

  return is_connection_with(name, connection);
}

bool
pipeline::priv
::is_group_connection_for(connection_t const& connection, group_connection_t const& gconnection)
{
  connection_t const& group_connection = gconnection.first;

  return (connection == group_connection);
}

pipeline::priv::propagation_exception
::propagation_exception(process::name_t const& upstream_name,
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

pipeline::priv::propagation_exception
::~propagation_exception() throw()
{
}

}
