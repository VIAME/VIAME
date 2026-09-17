// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "pipeline.h"
#include "pipeline_exception.h"

#include "edge.h"
#include "process_exception.h"

#include <viame/algorithm_framework/logger/logger.h>
#include <viame/algorithm_framework/config/config_block.h>
#include <viame/algorithm_framework/config/config_block_formatter.h>
#include <viame/algorithm_framework/util/string.h>

#include <viame/algorithm_framework/util/directed_graph.h>

#include <functional>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <stack>
#include <string>
#include <vector>
#include <sstream>
#include <memory>
#include <cstddef>

/**
 * \file pipeline.cxx
 *
 * \brief Implementation of the base class for \link viame::pipeline::pipeline pipelines\endlink.
 */

namespace viame::pipeline {

namespace {

/**
 * @brief Greatest common denominator
 *
 * gcd algorithm ref:
 * Knuth, Donald. "Art of Computer Programming", Volume 2, Third edition, pp. 337
 *
 * @param a First number
 * @param b Second number
 *
 * @return GCD
 */
template <typename T>
T
gcd( T a, T b )
{
  while (b != 0)
  {
    T r = a % b;
    a = b;
    b = r;
  } // end while

  return a;
}

// ----------------------------------------------------------------------------
/**
 * @brief Find least common multiple of two values
 *
 * @param a First value
 * @param b Second value
 *
 * @return Least common multiple.
 */
template <typename T>
T
lcm( T a, T b )
{
  T temp = gcd( a, b );

  return temp ? ( a / temp * b ) : 0;
}

} // end namespace

// ============================================================================
class pipeline::priv
{
  public:
    priv(pipeline* pipe, viame::config_block_sptr conf);
    ~priv();

    void check_duplicate_name(process::name_t const& name);
    void remove_from_pipeline(process::name_t const& name);
    void propagate(process::name_t const& root);

    typedef std::map<process::name_t, process_t> process_map_t;
    typedef std::stack<process::name_t> parent_stack_t;
    typedef std::map<process::name_t, process::name_t> process_parent_map_t;
    typedef std::map<size_t, edge_t> edge_map_t;

    typedef enum
    {
      push_upstream,
      push_downstream
    } direction_t;
    typedef std::pair<process::connection_t, direction_t> type_pinning_t;
    typedef std::vector<type_pinning_t> type_pinnings_t;

    typedef enum
    {
      type_deferred,
      type_mismatch,
      type_compatible
    } port_type_status;

    typedef std::map<process::port_addr_t, bool> shared_port_map_t;

    // Steps for checking a connection.
    port_type_status check_connection_types(process::connection_t const& connection,
                                            process::port_type_t const& up_type,
                                            process::port_type_t const& down_type);

    bool check_connection_flags(process::connection_t const& connection,
                                process::port_flags_t const& up_flags,
                                process::port_flags_t const& down_flags);

    // Steps for setting up the pipeline.
    void check_for_processes() const;
    void configure_processes();
    void check_for_data_dep_ports() const;
    void propagate_pinned_types();
    void check_for_untyped_ports() const;
    void make_connections();
    void check_for_required_ports() const;
    void check_for_dag() const;
    void initialize_processes();

    void ensure_setup() const;

    pipeline* const q;
    viame::config_block_sptr const config;

    process::connections_t planned_connections;
    process::connections_t connections;

    process_map_t process_map;
    edge_map_t edge_map;

    process_parent_map_t process_parent_map;
    parent_stack_t parent_stack;

    process::connections_t data_dep_connections;
    process::connections_t untyped_connections;
    type_pinnings_t type_pinnings;

    shared_port_map_t connected_shared_ports;

    bool setup;
    bool setup_in_progress;
    bool setup_successful;
    bool running;

    viame::logger_handle_t m_logger;

    static bool is_upstream_for(process::port_addr_t const& addr, process::connection_t const& connection);
    static bool is_downstream_for(process::port_addr_t const& addr, process::connection_t const& connection);
    static bool is_addr_on(process::name_t const& name, process::port_addr_t const& addr);
    static bool is_connection_with(process::name_t const& name, process::connection_t const& connection);

    class propagation_exception
      : public pipeline_exception
    {
      public:
        propagation_exception(process::name_t const& upstream_name,
                              process::port_t const& upstream_port,
                              process::name_t const& downstream_name,
                              process::port_t const& downstream_port,
                              process::port_type_t const& type,
                              bool push_upstream) noexcept;
        ~propagation_exception() noexcept;

        process::name_t const m_upstream_name;
        process::port_t const m_upstream_port;
        process::name_t const m_downstream_name;
        process::port_t const m_downstream_port;
        process::port_type_t const m_type;
        bool const m_push_upstream;
    };

  private:
  static viame::config_block_key_t const config_non_blocking;
    static viame::config_block_key_t const config_edge;
    static viame::config_block_key_t const config_edge_type;
    static viame::config_block_key_t const config_edge_conn;
    static viame::config_block_key_t const upstream_subblock;
    static viame::config_block_key_t const downstream_subblock;
};

// Process property
viame::config_block_key_t const pipeline::priv::config_non_blocking = viame::config_block_key_t("_non_blocking");

// Pipeline properties
viame::config_block_key_t const pipeline::priv::config_edge         = viame::config_block_key_t("_edge");
viame::config_block_key_t const pipeline::priv::config_edge_type    = viame::config_block_key_t("_edge_by_type");
viame::config_block_key_t const pipeline::priv::config_edge_conn    = viame::config_block_key_t("_edge_by_conn");
viame::config_block_key_t const pipeline::priv::upstream_subblock   = viame::config_block_key_t("up");
viame::config_block_key_t const pipeline::priv::downstream_subblock = viame::config_block_key_t("down");

// ------------------------------------------------------------------
pipeline
::pipeline(viame::config_block_sptr const& config)
  : d()
{
  if (!config)
  {
    VITAL_THROW( null_pipeline_config_exception );
  }

  d.reset(new priv(this, config));
}

pipeline
::~pipeline()
{
}

// ------------------------------------------------------------------
void
pipeline
::add_process(process_t const& process)
{
  if (!process)
  {
    VITAL_THROW( null_process_addition_exception );
  }

  if (d->setup)
  {
    VITAL_THROW( add_after_setup_exception,
                 process->name());
  }

  process::name_t const name = process->name();

  d->check_duplicate_name(name);

  process::name_t parent;

  if (!d->parent_stack.empty())
  {
    parent = d->parent_stack.top();
  }

  d->process_parent_map[name] = parent;

  d->process_map[name] = process;
}

// ------------------------------------------------------------------
void
pipeline
::remove_process(process::name_t const& name)
{
  if (d->setup)
  {
    VITAL_THROW( remove_after_setup_exception,
                 name);
  }



  if (!d->process_map.count(name))
  {
    VITAL_THROW( no_such_process_exception,
                 name);
  }

  d->process_map.erase(name);

  d->remove_from_pipeline(name);
}

// ------------------------------------------------------------------
void
pipeline
::connect(process::name_t const& upstream_name,
          process::port_t const& upstream_port,
          process::name_t const& downstream_name,
          process::port_t const& downstream_port)
{
  if (d->setup && !d->setup_in_progress)
  {
    VITAL_THROW( connection_after_setup_exception,
                 upstream_name, upstream_port,
                 downstream_name, downstream_port);
  }

  process::port_addr_t const up_addr = process::port_addr_t(upstream_name, upstream_port);
  process::port_addr_t const down_addr = process::port_addr_t(downstream_name, downstream_port);
  process::connection_t const connection = process::connection_t(up_addr, down_addr);

  if (!d->setup_in_progress)
  {
    d->planned_connections.push_back(connection);
  }

  process_t const up_proc = process_by_name(upstream_name);
  process_t const down_proc = process_by_name(downstream_name);

  process::port_info_t const up_info = up_proc->output_port_info(upstream_port);
  process::port_info_t const down_info = down_proc->input_port_info(downstream_port);

  process::port_flags_t const& up_flags = up_info->flags;
  process::port_flags_t const& down_flags = down_info->flags;

  if (!d->check_connection_flags(connection, up_flags, down_flags))
  {
    VITAL_THROW( connection_flag_mismatch_exception,
                 upstream_name, upstream_port,
                 downstream_name, downstream_port);
  }

  process::port_type_t const& up_type = up_info->type;
  process::port_type_t const& down_type = down_info->type;

  switch (d->check_connection_types(connection, up_type, down_type))
  {
  case priv::type_deferred:
    return;

  case priv::type_mismatch:
    VITAL_THROW( connection_type_mismatch_exception,
                 upstream_name, upstream_port, up_type,
                 downstream_name, downstream_port, down_type);
  case priv::type_compatible:
  default:
    break;
  }

  d->connections.push_back(connection);
}

// ------------------------------------------------------------------
void
pipeline
::disconnect(process::name_t const& upstream_name,
             process::port_t const& upstream_port,
             process::name_t const& downstream_name,
             process::port_t const& downstream_port)
{
  if (d->setup)
  {
    VITAL_THROW( disconnection_after_setup_exception,
                 upstream_name, upstream_port,
                 downstream_name, downstream_port);
  }

  process::port_addr_t const upstream_addr = process::port_addr_t(upstream_name, upstream_port);
  process::port_addr_t const downstream_addr = process::port_addr_t(downstream_name, downstream_port);
  process::connection_t const conn = process::connection_t(upstream_addr, downstream_addr);

  std::function<bool (process::connection_t const&)> const eq = std::bind(std::equal_to<process::connection_t>(),
                                                                          conn, std::placeholders::_1);

#define FORGET_CONNECTION(T, f, conns)                                   \
  do                                                                     \
  {                                                                      \
    T::iterator const i = std::remove_if(conns.begin(), conns.end(), f); \
    conns.erase(i, conns.end());                                         \
  } while (false)

  FORGET_CONNECTION(process::connections_t, eq, d->planned_connections);
  FORGET_CONNECTION(process::connections_t, eq, d->connections);
  FORGET_CONNECTION(process::connections_t, eq, d->data_dep_connections);
  FORGET_CONNECTION(process::connections_t, eq, d->untyped_connections);

#undef FORGET_CONNECTION
}

// ------------------------------------------------------------------
void
pipeline
::setup_pipeline()
{
  if (d->setup)
  {
    VITAL_THROW( pipeline_duplicate_setup_exception );
  }

  d->check_for_processes();

  // There's no turning back after this (processes are modified and may not be
  // able to be added/removed without compromising the checks after this point).
  d->setup = true;
  d->setup_in_progress = true;
  d->setup_successful = false;

  try
  {
    d->configure_processes();
    d->check_for_data_dep_ports();
    d->propagate_pinned_types();
    d->check_for_untyped_ports();
    d->make_connections();
    d->check_for_required_ports();
    d->check_for_dag();
    d->initialize_processes();
  }
  catch (...)
  {
    d->setup_in_progress = false;
    throw;
  }

  d->setup_in_progress = false;
  d->setup_successful = true;
}

// ------------------------------------------------------------------
bool
pipeline
::is_setup() const
{
  return d->setup;
}

// ------------------------------------------------------------------
bool
pipeline
::setup_successful() const
{
  return d->setup_successful;
}

// ------------------------------------------------------------------
void
pipeline
::reset()
{
  if (d->running)
  {
    VITAL_THROW( reset_running_pipeline_exception );
  }

  d->setup = false;
  d->setup_successful = false;

  priv::process_map_t const names = d->process_map;

  // Reset all the processes.
  for (priv::process_map_t::value_type& process_entry : d->process_map)
  {
    process_t const& process = process_entry.second;

    process->reset();
  }

  // Clear internal bookkeeping.
  d->connections.clear();
  d->edge_map.clear();
  d->data_dep_connections.clear();
  d->untyped_connections.clear();
  d->type_pinnings.clear();
  d->connected_shared_ports.clear();

  d->setup_in_progress = true;

  // Replay connections.
  for (process::connection_t const& connection : d->planned_connections)
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

// ------------------------------------------------------------------
void
pipeline
::reconfigure(viame::config_block_sptr const& conf) const
{
  if (!d->setup)
  {
    VITAL_THROW( reconfigure_before_setup_exception );
  }

  for (priv::process_map_t::value_type const& proc_entry : d->process_map)
  {
    process::name_t const& name = proc_entry.first;
    process_t const& proc = proc_entry.second;
    viame::config_block_sptr const proc_conf = conf->subblock_view(name);

    proc->reconfigure(proc_conf);
  }
}

// ------------------------------------------------------------------
process::names_t
pipeline
::process_names() const
{
  process::names_t names;

  for (priv::process_map_t::value_type const& process_index : d->process_map)
  {
    process::name_t const& name = process_index.first;

    names.push_back(name);
  }

  return names;
}

// ------------------------------------------------------------------
process_t
pipeline
::process_by_name(process::name_t const& name) const
{
  priv::process_map_t::const_iterator i = d->process_map.find(name);

  if (i == d->process_map.end())
  {
    VITAL_THROW( no_such_process_exception,name);
  }

  return i->second;
}

// ------------------------------------------------------------------
process::port_addrs_t
pipeline
::connections_from_addr(process::name_t const& name, process::port_t const& port) const
{
  process::port_addrs_t addrs;

  for (process::connection_t const& connection : d->planned_connections)
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

// ------------------------------------------------------------------
process::port_addr_t
pipeline
::connection_to_addr(process::name_t const& name, process::port_t const& port) const
{
  for (process::connection_t const& connection : d->planned_connections)
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

// ------------------------------------------------------------------
processes_t
pipeline
::upstream_for_process(process::name_t const& name) const
{
  d->ensure_setup();

  std::set<process::name_t> names;

  for (process::connection_t const& connection : d->connections)
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

  for (process::name_t const& process_name : names)
  {
    priv::process_map_t::const_iterator const i = d->process_map.find(process_name);
    process_t const& process = i->second;

    processes.push_back(process);
  }

  return processes;
}

// ------------------------------------------------------------------
process_t
pipeline
::upstream_for_port(process::name_t const& name, process::port_t const& port) const
{
  d->ensure_setup();

  for (process::connection_t const& connection : d->connections)
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

// ------------------------------------------------------------------
processes_t
pipeline
::downstream_for_process(process::name_t const& name) const
{
  d->ensure_setup();

  std::set<process::name_t> names;

  for (process::connection_t const& connection : d->connections)
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

  for (process::name_t const& process_name : names)
  {
    priv::process_map_t::const_iterator const i = d->process_map.find(process_name);
    process_t const& process = i->second;

    processes.push_back(process);
  }

  return processes;
}

// ------------------------------------------------------------------
processes_t
pipeline
::downstream_for_port(process::name_t const& name, process::port_t const& port) const
{
  d->ensure_setup();

  std::set<process::name_t> names;

  for (process::connection_t const& connection : d->connections)
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

  for (process::name_t const& process_name : names)
  {
    priv::process_map_t::const_iterator const i = d->process_map.find(process_name);
    process_t const& process = i->second;

    processes.push_back(process);
  }

  return processes;
}

// ------------------------------------------------------------------
process::port_addr_t
pipeline
::sender_for_port(process::name_t const& name, process::port_t const& port) const
{
  d->ensure_setup();

  for (process::connection_t const& connection : d->connections)
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

// ------------------------------------------------------------------
process::port_addrs_t
pipeline
::receivers_for_port(process::name_t const& name, process::port_t const& port) const
{
  d->ensure_setup();

  process::port_addrs_t port_addrs;

  for (process::connection_t const& connection : d->connections)
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

// ------------------------------------------------------------------
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
    process::connection_t const& connection = d->connections[i];

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

// ------------------------------------------------------------------
edges_t
pipeline
::input_edges_for_process(process::name_t const& name) const
{
  d->ensure_setup();

  edges_t edges;

  for (priv::edge_map_t::value_type const& edge_index : d->edge_map)
  {
    size_t const& i = edge_index.first;
    edge_t const& edge = edge_index.second;

    process::connection_t const connection = d->connections[i];

    process::port_addr_t const& downstream_addr = connection.second;

    process::name_t const& downstream_name = downstream_addr.first;

    if (downstream_name == name)
    {
      edges.push_back(edge);
    }
  }

  return edges;
}

// ------------------------------------------------------------------
edge_t
pipeline
::input_edge_for_port(process::name_t const& name, process::port_t const& port) const
{
  d->ensure_setup();

  for (priv::edge_map_t::value_type const& edge_index : d->edge_map)
  {
    size_t const& i = edge_index.first;
    edge_t const& edge = edge_index.second;

    process::connection_t const connection = d->connections[i];

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

// ------------------------------------------------------------------
edges_t
pipeline
::output_edges_for_process(process::name_t const& name) const
{
  d->ensure_setup();

  edges_t edges;

  for (priv::edge_map_t::value_type const& edge_index : d->edge_map)
  {
    size_t const& i = edge_index.first;
    edge_t const& edge = edge_index.second;

    process::connection_t const connection = d->connections[i];

    process::port_addr_t const& upstream_addr = connection.first;

    process::name_t const& upstream_name = upstream_addr.first;

    if (upstream_name == name)
    {
      edges.push_back(edge);
    }
  }

  return edges;
}

// ------------------------------------------------------------------
edges_t
pipeline
::output_edges_for_port(process::name_t const& name, process::port_t const& port) const
{
  d->ensure_setup();

  edges_t edges;

  for (priv::edge_map_t::value_type const& edge_index : d->edge_map)
  {
    size_t const& i = edge_index.first;
    edge_t const& edge = edge_index.second;

    process::connection_t const connection = d->connections[i];

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

// ------------------------------------------------------------------
void
pipeline
::start()
{
  d->ensure_setup();

  d->running = true;
}

// ------------------------------------------------------------------
void
pipeline
::stop()
{
  if (!d->running)
  {
    static std::string const reason = "Start/stop pipeline state tracking failed";

    throw std::logic_error(reason);
  }

  d->running = false;
}

// ------------------------------------------------------------------
processes_t
pipeline
::get_python_processes() const
{
  // Run through each process, checking to see if any are python
  processes_t python_processes; // Start with empty list
  for (priv::process_map_t::value_type const& process_index : d->process_map)
  {
    process_t proc = process_index.second;
    auto properties = proc->properties();
    if ( properties.find("_python") != properties.end() )
    {
      python_processes.push_back( proc );
    }
  }

  return python_processes;
}

// ------------------------------------------------------------------
pipeline::priv
::priv(pipeline* pipe, viame::config_block_sptr conf)
  : q(pipe)
  , config(conf)
  , planned_connections()
  , connections()
  , process_map()
  , edge_map()
  , data_dep_connections()
  , untyped_connections()
  , type_pinnings()
  , setup(false)
  , setup_in_progress(false)
  , setup_successful(false)
  , running(false)
  , m_logger( viame::get_logger( "viame.pipeline" ) )
{
  if ( IS_DEBUG_ENABLED( m_logger ) )
  {
    std::stringstream msg;
    viame::config_block_formatter fmt( config );
    fmt.print(msg);
    LOG_DEBUG( m_logger, "pipeline config:\n" << msg.str() );
  }
}

pipeline::priv
::~priv()
{
}

// ------------------------------------------------------------------
void
pipeline::priv
::check_duplicate_name(process::name_t const& name)
{
  if (process_map.count(name))
  {
    VITAL_THROW( duplicate_process_name_exception,
                 name);
  }
}

// ------------------------------------------------------------------
void
pipeline::priv
::remove_from_pipeline(process::name_t const& name)
{
  std::function<bool (process::connection_t const&)> const is = std::bind(&is_connection_with, name,
                                                                          std::placeholders::_1);

#define FORGET_CONNECTIONS(T, f, conns)                                  \
  do                                                                     \
  {                                                                      \
    T::iterator const i = std::remove_if(conns.begin(), conns.end(), f); \
    conns.erase(i, conns.end());                                         \
  } while (false)

  FORGET_CONNECTIONS(process::connections_t, is, planned_connections);
  FORGET_CONNECTIONS(process::connections_t, is, connections);
  FORGET_CONNECTIONS(process::connections_t, is, data_dep_connections);
  FORGET_CONNECTIONS(process::connections_t, is, untyped_connections);

#undef FORGET_CONNECTIONS
}

// ------------------------------------------------------------------
pipeline::priv::port_type_status
pipeline::priv
::check_connection_types(process::connection_t const& connection,
                         process::port_type_t const& up_type,
                         process::port_type_t const& down_type)
{
  bool const up_data_dep = (up_type == process::type_data_dependent);

  if (up_data_dep)
  {
    data_dep_connections.push_back(connection);

    return type_deferred;
  }

  bool const up_flow_dep = viame::starts_with(up_type, process::type_flow_dependent);
  bool const down_flow_dep = viame::starts_with(down_type, process::type_flow_dependent);

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

// ------------------------------------------------------------------
/// Check connection flags to see if they are consistent.
/**
 * This method checks the flags for a proposed connection to see if
 * they are consistent and the connection should be allowed.
 */
bool
pipeline::priv
::check_connection_flags(process::connection_t const& connection,
                         process::port_flags_t const& up_flags,
                         process::port_flags_t const& down_flags)
{
  // Test the port flags to see what has been specified at port creation time.
  bool const is_const = (0 != up_flags.count(process::flag_output_const));
  bool const is_shared = (0 != up_flags.count(process::flag_output_shared));
  bool const is_mutable = (0 != down_flags.count(process::flag_input_mutable));

  // If "up" is const and "down" is mutable, then connection is not allowed
  if (is_const && is_mutable)
  {
    return false;
  }

  if (is_shared)
  {
    process::port_addr_t const& up_addr = connection.first;

    shared_port_map_t::const_iterator const i = connected_shared_ports.find(up_addr);

    if (i == connected_shared_ports.end())
    {
      // Nothing is connected yet.
      // Since the port is shared, then mark as mutable.
      connected_shared_ports[up_addr] = is_mutable;
    }
    else
    {
      const bool has_mutable = i->second;

      // Only one input can listen to a shared port if any are mutable.
      if (is_mutable || has_mutable)
      {
        return false;
      }
    }
  }

  // All other combinations are allowable
  return true;
}

// ------------------------------------------------------------------
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

    process::connections_t const conns = untyped_connections;
    untyped_connections.clear();

    for (process::connection_t const& connection : conns)
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

        bool const flow_dep = viame::starts_with(type, process::type_flow_dependent);

        if (!flow_dep)
        {
          process_t const up_proc = q->process_by_name(upstream_name);

          if (!up_proc->set_output_port_type(upstream_port, type))
          {
            VITAL_THROW( propagation_exception,
                         upstream_name, upstream_port,
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

        bool const flow_dep = viame::starts_with(type, process::type_flow_dependent);

        if (!flow_dep)
        {
          process_t const down_proc = q->process_by_name(downstream_name);

          if (!down_proc->set_input_port_type(downstream_port, type))
          {
            VITAL_THROW( propagation_exception,
                         upstream_name, upstream_port,
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

// ------------------------------------------------------------------
void
pipeline::priv
::check_for_processes() const
{
  if (process_map.empty())
  {
    VITAL_THROW( no_processes_exception );
  }
}

// ------------------------------------------------------------------
void
pipeline::priv
::configure_processes()
{
  // Configure processes.
  for (process_map_t::value_type const& proc_data : process_map)
  {
    process::name_t const& name = proc_data.first;
    process_t const& proc = proc_data.second;
    process::connections_t unresolved_connections;

    // Configure the process.
    proc->configure();

    bool resolved_types = false;

    // Resolve any data dependent connections.
    for (process::connection_t const& data_dep_connection : data_dep_connections)
    {
      process::port_addr_t const& data_addr = data_dep_connection.first;
      process::port_addr_t const& downstream_addr = data_dep_connection.second;

      process::name_t const& data_name = data_addr.first;
      process::port_t const& data_port = data_addr.second;
      process::name_t const& downstream_name = downstream_addr.first;
      process::port_t const& downstream_port = downstream_addr.second;

      // if this is a connection from this process...
      if (name == data_name)
      {
        process::port_info_t const info = proc->output_port_info(data_port);

        // The process should have resolved all data dependent ports
        // by now. It is an error if there are still some around.
        if (info->type == process::type_data_dependent)
        {
          VITAL_THROW( untyped_data_dependent_exception,
                       data_name, data_port);
        }

        resolved_types = true;

        q->connect(data_name, data_port,
                   downstream_name, downstream_port);
      }
      else
      {
        unresolved_connections.push_back(data_dep_connection);
      }
    } // end for

    if (resolved_types)
    {
      data_dep_connections = unresolved_connections;
    }
  } // end for

}

// ------------------------------------------------------------------
void
pipeline::priv
::check_for_data_dep_ports() const
{
  if (!data_dep_connections.empty())
  {
    static std::string const reason = "Data dependency port tracking failed";

    throw std::logic_error(reason);
  }
}

// ------------------------------------------------------------------
void
pipeline::priv
::propagate_pinned_types()
{
  type_pinnings_t const pinnings = type_pinnings;
  type_pinnings.clear();

  for (type_pinning_t const& pinning : pinnings)
  {
    process::connection_t const& connection = pinning.first;
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
          VITAL_THROW( connection_dependent_type_exception,
                       upstream_name, upstream_port,
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
          VITAL_THROW( connection_dependent_type_exception,
                       upstream_name, upstream_port,
                       downstream_name, downstream_port,
                       up_type, false);
        }

        name = downstream_name;
        port = downstream_port;
        type = up_type;

        break;

      default:
        continue;
    } // end switch

    try
    {
      propagate(name);
    }
    catch (propagation_exception const& e)
    {
      VITAL_THROW( connection_dependent_type_cascade_exception,
                   name, port, type,
                   e.m_upstream_name, e.m_upstream_port,
                   e.m_downstream_name, e.m_downstream_port,
                   e.m_type, e.m_push_upstream);
    }

    // Retry the connection.
    q->connect(upstream_name, upstream_port,
               downstream_name, downstream_port);
  }

  if (!type_pinnings.empty())
  {
    propagate_pinned_types();
  }
}

// ------------------------------------------------------------------
void
pipeline::priv
::check_for_untyped_ports() const
{
  if (!untyped_connections.empty())
  {
    VITAL_THROW( untyped_connection_exception );
  }
}

// ------------------------------------------------------------------
void
pipeline::priv
::make_connections()
{
  size_t const len = connections.size();

  for (size_t i = 0; i < len; ++i)
  {
    process::connection_t const& connection = connections[i];

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

    // Extract the "_edge:" subblock from the supplied config.
    // This supplies the default or most general config values.
    // The edge type config will be merged in to override defaults for this edge.
    // Then the connection based config will be merged to override.
    viame::config_block_sptr edge_config = config->subblock(priv::config_edge);

    // Configure the edge based on its type. (_edge_by_type)
    {
      process::port_type_t const& down_type = down_info->type;  // data type on edge
      viame::config_block_sptr const type_config = config->subblock(priv::config_edge_type);
      viame::config_block_sptr const edge_type_config = type_config->subblock(down_type);

      edge_config->merge_config(edge_type_config);

      if ( IS_TRACE_ENABLED( m_logger ) )
      {
        std::stringstream msg;
        msg << "-- Edge type config for type \"" << down_type << "\" :\n";
        viame::config_block_formatter fmt( edge_type_config );
        fmt.print( msg );
        LOG_TRACE( m_logger, msg.str() );
      }
    }

    // Configure the edge based on the connected ports. (_edge_by_conn)
    {
      viame::config_block_sptr const conn_config = config->subblock(priv::config_edge_conn);
      viame::config_block_sptr const up_config =
        conn_config->subblock(upstream_name + viame::config_block::block_sep() +
          upstream_subblock + viame::config_block::block_sep() + upstream_port);

      viame::config_block_sptr const down_config =
        conn_config->subblock(downstream_name + viame::config_block::block_sep() +
          downstream_subblock + viame::config_block::block_sep() + downstream_port);

      edge_config->merge_config(up_config);
      edge_config->merge_config(down_config);

      if ( IS_TRACE_ENABLED( m_logger ) )
      {
        std::stringstream msg;
        msg << "-- Up_config for \""
            << upstream_name + viame::config_block::block_sep()
             + upstream_subblock + viame::config_block::block_sep() + upstream_port
            << "\" :\n";
        viame::config_block_formatter up_fmt( up_config );
        up_fmt.print(msg);
        msg << "\n-- Down_config for \""
            << downstream_name + viame::config_block::block_sep()
             + downstream_subblock + viame::config_block::block_sep()
             + downstream_port
            << "\" :\n";
        viame::config_block_formatter down_fmt( up_config );
        down_fmt.print(msg);
        LOG_TRACE( m_logger, msg.str() );
      }

    }

    // Configure the edge.
    {
      // Check to see if this port has _nodep flag set (for backward edge.)
      // Pass that value to the edge through the config.
      bool const has_nodep = (0 != down_flags.count(process::flag_input_nodep));

      edge_config->set_value(edge::config_dependency, (has_nodep ? "false" : "true"));
      edge_config->mark_read_only(edge::config_dependency);
    }

    // Process non_blocking processes
    {
      // Check the config of the down stream process to see if the
      // process is marked as non_blocking. If it is non_blocking,
      // then set the input edge property to be non-blocking and force
      // the capacity.
      //
      // Since we are looking at the process property for
      // non_blocking, all input edges for this process will be
      // configured the same.
      const auto proc_config = down_proc->get_config();
      if ( proc_config->has_value( config_non_blocking ) )
      {
        const size_t capacity = proc_config->get_value<size_t>( config_non_blocking );
        edge_config->set_value( edge::config_capacity, capacity );
        edge_config->set_value( edge::config_blocking, false );
      }
    }

    if ( IS_DEBUG_ENABLED( m_logger ) )
    {
      std::stringstream msg;
      viame::config_block_formatter fmt( edge_config );
      fmt.print(msg);

      LOG_TRACE( m_logger,
                 "Edge config for "  << upstream_name << "." <<
                 upstream_port  << " -> " << downstream_name << "." <<
                 downstream_port << "\n" << msg.str() );
    }

    // Create a new edge
    edge_t const e = std::make_shared<edge>(edge_config);

    edge_map[i] = e;

    up_proc->connect_output_port(upstream_port, e);
    down_proc->connect_input_port(downstream_port, e);

    e->set_upstream_process(up_proc);
    e->set_downstream_process(down_proc);
  }
}

// ------------------------------------------------------------------
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
      if (procs.count(cur_proc))
      {
        continue;
      }

      procs.insert(cur_proc);

      // Check for required input ports.
      {
        process_t const process = q->process_by_name(cur_proc);

        // Check for required input ports.
        process::ports_t const input_ports = process->input_ports();
        for (process::port_t const& port : input_ports)
        {
          // Check for required flags.
          process::port_flags_t const port_flags = process->input_port_info(port)->flags;

          if (port_flags.count(process::flag_required))
          {
            // Port is marked as required and therefore, must be connected.
            if (!q->input_edge_for_port(cur_proc, port))
            {
              static std::string const reason = "The input port has the required flag";

              VITAL_THROW( missing_connection_exception,
                           cur_proc, port, reason);
            }
          }
        }

        // Check for required output ports.
        process::ports_t const output_ports = process->output_ports();
        for (process::port_t const& port : output_ports)
        {
          // Check for required flags.
          process::port_flags_t const port_flags = process->output_port_info(port)->flags;

          if (port_flags.count(process::flag_required))
          {
            if (q->output_edges_for_port(cur_proc, port).empty())
            {
              static std::string const reason = "The output port has the required flag";

              VITAL_THROW( missing_connection_exception,
                           cur_proc, port, reason);
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
      for (process_t const& proc : connected_procs)
      {
        to_visit.push(proc->name());
      }
    }
  }

  if (procs.size() != process_map.size())
  {
    VITAL_THROW( orphaned_processes_exception );
  }
}

// ------------------------------------------------------------------
void
pipeline::priv
::check_for_dag() const
{
  typedef viame::directed_graph<process::name_t> pipeline_graph_t;
  typedef pipeline_graph_t::vertex_descriptor vertex_t;
  typedef std::deque<vertex_t> vertices_t;
  typedef std::map<process::name_t, vertex_t> vertex_map_t;

  pipeline_graph_t graph;

  // Create the graph.
  {
    vertex_map_t vertex_map;

    process::names_t const names = q->process_names();

    for (process::name_t const& name : names)
    {
      vertex_t const s = graph.add_vertex();
      graph[s] = name;
      vertex_map[name] = s;
    }

    for (process::name_t const& name : names)
    {
      process_t const proc = q->process_by_name(name);
      process::ports_t const iports = proc->input_ports();

      vertex_t const t = vertex_map[name];

      for (process::port_t const& port : iports)
      {
        process::port_addr_t const sender = q->sender_for_port(name, port);

        if (sender == process::port_addr_t())
        {
          continue;
        }

        process::name_t const& sender_name = sender.first;

        process::port_info_t const info = proc->input_port_info(port);
        process::port_flags_t const& flags = info->flags;

        if (flags.count(process::flag_input_nodep))
        {
          continue;
        }

        vertex_t const s = vertex_map[sender_name];

        graph.add_edge(s, t);
      }
    }
  }

  vertices_t vertices;

  try
  {
    viame::topological_sort(graph, std::front_inserter(vertices));
  }
  catch (viame::not_a_dag_exception const&)
  {
    VITAL_THROW( not_a_dag_exception );
  }
}

// ------------------------------------------------------------------
void
pipeline::priv
::initialize_processes()
{
  process::names_t const names = q->process_names();

  // Initialize processes.
  for (process::name_t const& name : names)
  {
    process_t const proc = q->process_by_name(name);

    proc->init();
  }
}

// ------------------------------------------------------------------
void
pipeline::priv
::ensure_setup() const
{
  if (!setup)
  {
    VITAL_THROW( pipeline_not_setup_exception );
  }

  if (!setup_in_progress && !setup_successful)
  {
    VITAL_THROW( pipeline_not_ready_exception );
  }
}

// ------------------------------------------------------------------
bool
pipeline::priv
::is_upstream_for(process::port_addr_t const& addr, process::connection_t const& connection)
{
  process::port_addr_t const up_addr = connection.first;

  return (addr == up_addr);
}

// ------------------------------------------------------------------
bool
pipeline::priv
::is_downstream_for(process::port_addr_t const& addr, process::connection_t const& connection)
{
  process::port_addr_t const down_addr = connection.second;

  return (addr == down_addr);
}

// ------------------------------------------------------------------
bool
pipeline::priv
::is_addr_on(process::name_t const& name, process::port_addr_t const& addr)
{
  process::name_t const& proc_name = addr.first;

  return (name == proc_name);
}

// ------------------------------------------------------------------
bool
pipeline::priv
::is_connection_with(process::name_t const& name, process::connection_t const& connection)
{
  process::port_addr_t const& upstream_addr = connection.first;
  process::port_addr_t const& downstream_addr = connection.second;

  return (is_addr_on(name, upstream_addr) || is_addr_on(name, downstream_addr));
}

// ------------------------------------------------------------------
pipeline::priv::propagation_exception
::propagation_exception(process::name_t const& upstream_name,
                        process::port_t const& upstream_port,
                        process::name_t const& downstream_name,
                        process::port_t const& downstream_port,
                        process::port_type_t const& type,
                        bool push_upstream) noexcept
  : m_upstream_name(upstream_name)
  , m_upstream_port(upstream_port)
  , m_downstream_name(downstream_name)
  , m_downstream_port(downstream_port)
  , m_type(type)
  , m_push_upstream(push_upstream)
{
  m_what = "<internal>";
}

// ------------------------------------------------------------------
pipeline::priv::propagation_exception
::~propagation_exception() noexcept
{
}

} // namespace viame::pipeline
