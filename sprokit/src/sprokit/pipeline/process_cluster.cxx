// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "process_cluster.h"

#include "pipeline_exception.h"
#include "process_cluster_exception.h"
#include "process_exception.h"
#include "process_factory.h"

#include <vital/logger/logger.h>

#include <algorithm>
#include <map>
#include <utility>
#include <vector>

/**
 * \file process_cluster.cxx
 *
 * \brief Implementation for \link sprokit::process_cluster process cluster\endlink.
 */

namespace sprokit {

process::property_t const process_cluster::property_cluster = process::property_t("_cluster");

// ==================================================================
class process_cluster::priv
{
  public:
    priv();
    ~priv();

    typedef std::pair<kwiver::vital::config_block_key_t, kwiver::vital::config_block_key_t> config_mapping_t;
    typedef std::vector<config_mapping_t> config_mappings_t;
    typedef std::map<process::name_t, config_mappings_t> config_map_t;
    typedef std::map<process::name_t, process_t> process_map_t;

    bool has_name(name_t const& name) const;
    void ensure_name(name_t const& name) const;

    config_map_t config_map;
    process_map_t processes;
    connections_t input_mappings;
    connections_t output_mappings;
    connections_t internal_connections;
    kwiver::vital::logger_handle_t m_logger;
};

// ==================================================================
processes_t
process_cluster
::processes() const
{
  processes_t procs;

  for (priv::process_map_t::value_type const& process_entry : d->processes)
  {
    process_t const& proc = process_entry.second;

    procs.push_back(proc);
  }

  return procs;
}

// ------------------------------------------------------------------
process::connections_t
process_cluster
::input_mappings() const
{
  return d->input_mappings;
}

// ------------------------------------------------------------------
process::connections_t
process_cluster
::output_mappings() const
{
  return d->output_mappings;
}

// ------------------------------------------------------------------
process::connections_t
process_cluster
::internal_connections() const
{
  return d->internal_connections;
}

// ------------------------------------------------------------------
process_cluster
::process_cluster(kwiver::vital::config_block_sptr const& config)
  : process(config)
  , d(new priv)
{
}

// ------------------------------------------------------------------
process_cluster
::~process_cluster()
{
}

static process::name_t convert_name(process::name_t const& cluster_name, process::name_t const& process_name);

// ------------------------------------------------------------------
void
process_cluster
::map_config(kwiver::vital::config_block_key_t const& key,
             name_t const& name_,
             kwiver::vital::config_block_key_t const& mapped_key)
{
  if (d->has_name(name_))
  {
    VITAL_THROW( mapping_after_process_exception,
                 name(), key,
                 name_, mapped_key);
  }

  priv::config_mapping_t const mapping = priv::config_mapping_t(key, mapped_key);

  d->config_map[name_].push_back(mapping);
}

// ------------------------------------------------------------------
void
process_cluster
::add_process( name_t const& name_, // local process name
               type_t const& type_, // process type
               kwiver::vital::config_block_sptr const& conf )
{
  if ( d->processes.count( name_ ) )
  {
    VITAL_THROW( duplicate_process_name_exception,
                 name_ );
  }

  typedef std::set< kwiver::vital::config_block_key_t > key_set_t;

  kwiver::vital::config_block_keys_t const cur_keys = conf->available_values();
  key_set_t ro_keys;

  kwiver::vital::config_block_sptr const new_conf = kwiver::vital::config_block::empty_config();

  // Loop over all config keys provided and make a list of those that are READ_ONLY
  for ( kwiver::vital::config_block_key_t const& key : cur_keys )
  {
    kwiver::vital::config_block_value_t const value =
      conf->get_value< kwiver::vital::config_block_value_t >( key );

    new_conf->set_value( key, value );

    if ( conf->is_read_only( key ) )
    {
      ro_keys.insert( key );
    }
  }

  // Now map these configs for the named process
  priv::config_mappings_t const mappings = d->config_map[ name_ ];

  for ( priv::config_mapping_t const& mapping : mappings )
  {
    kwiver::vital::config_block_key_t const& key = mapping.first;
    kwiver::vital::config_block_key_t const& mapped_key = mapping.second;

    kwiver::vital::config_block_value_t const value =
      config_value< kwiver::vital::config_block_value_t >( key );

    if ( ro_keys.count( mapped_key ) )
    {
      kwiver::vital::config_block_value_t const new_value =
        new_conf->get_value< kwiver::vital::config_block_value_t >( mapped_key );

      VITAL_THROW( mapping_to_read_only_value_exception,
                   name(), key, value, name_, mapped_key, new_value );
    }

    if ( new_conf->has_value( mapped_key ) )
    {
      LOG_WARN( d->m_logger,
                "Config item \"" << mapped_key <<
      "\" already has a value. Value will be replaced." );
    }

    // Create an entry for the config value to be keyed by the mapped key
    new_conf->set_value( mapped_key, value );

    // Make sure that the parameter is not reconfigured away by anything other
    // than this cluster.
    new_conf->mark_read_only( mapped_key );
  }

  // Make sure all RO entries are marked
  for ( kwiver::vital::config_block_key_t const& key : ro_keys )
  {
    new_conf->mark_read_only( key );
  }

  // convert the supplied process name into the cluster based name and
  // create using that name.
  name_t const real_name = convert_name( name(), name_ );

  process_t const proc = create_process( type_, real_name, new_conf );

  // Note we are filing the process under the supplied name
  d->processes[ name_ ] = proc;
}

// ------------------------------------------------------------------
void
process_cluster
::map_input(port_t const& port, name_t const& name_, port_t const& mapped_port)
{
  d->ensure_name(name_);

  process_t const& proc = d->processes[name_];

  if (!proc->input_port_info(mapped_port))
  {
    VITAL_THROW( no_such_port_exception,
                 name_, mapped_port);

    return;
  }

  name_t const real_name = convert_name(name(), name_);

  for (connection_t const& input_mapping : d->input_mappings)
  {
    port_addr_t const& process_addr = input_mapping.second;
    name_t const& process_name = process_addr.first;
    port_t const& process_port = process_addr.second;

    if ((process_name == real_name) &&
        (process_port == mapped_port))
    {
      VITAL_THROW( port_reconnect_exception,
                   process_name, mapped_port);
    }
  }

  port_addr_t const cluster_addr = port_addr_t(name(), port);
  port_addr_t const mapped_addr = port_addr_t(real_name, mapped_port);

  connection_t const connection = connection_t(cluster_addr, mapped_addr);

  d->input_mappings.push_back(connection);
}

// ------------------------------------------------------------------
void
process_cluster
::map_output(port_t const& port, name_t const& name_, port_t const& mapped_port)
{
  d->ensure_name(name_);

  /// \todo Make sure that only one process is mapped to a port.

  process_t const& proc = d->processes[name_];

  if (!proc->output_port_info(mapped_port))
  {
    VITAL_THROW( no_such_port_exception,
                 name_, mapped_port);

    return;
  }

  for (connection_t const& output_mapping : d->output_mappings)
  {
    port_addr_t const& cluster_addr = output_mapping.second;
    port_t const& cluster_port = cluster_addr.second;

    if (cluster_port == port)
    {
      VITAL_THROW( port_reconnect_exception,
                   name(), port);
    }
  }

  name_t const real_name = convert_name(name(), name_);

  port_addr_t const cluster_addr = port_addr_t(name(), port);
  port_addr_t const mapped_addr = port_addr_t(real_name, mapped_port);

  connection_t const connection = connection_t(mapped_addr, cluster_addr);

  d->output_mappings.push_back(connection);
}

// ------------------------------------------------------------------
void
process_cluster
::connect(name_t const& upstream_name, port_t const& upstream_port,
          name_t const& downstream_name, port_t const& downstream_port)
{
  d->ensure_name(upstream_name);
  d->ensure_name(downstream_name);

  process_t const& up_proc = d->processes[upstream_name];

  if (!up_proc->output_port_info(upstream_port))
  {
    VITAL_THROW( no_such_port_exception,
                 upstream_name, upstream_port);
  }

  process_t const& down_proc = d->processes[downstream_name];

  if (!down_proc->input_port_info(downstream_port))
  {
    VITAL_THROW( no_such_port_exception,
                 downstream_name, downstream_port);
  }

  name_t const up_real_name = convert_name(name(), upstream_name);
  name_t const down_real_name = convert_name(name(), downstream_name);

  port_addr_t const up_addr = port_addr_t(up_real_name, upstream_port);
  port_addr_t const down_addr = port_addr_t(down_real_name, downstream_port);

  connection_t const connection = connection_t(up_addr, down_addr);

  d->internal_connections.push_back(connection);
}

// ============================================================================
// Stub process implementations.
void
process_cluster
::_configure()
{
}

// ------------------------------------------------------------------
void
process_cluster
::_init()
{
}

// ------------------------------------------------------------------
void
process_cluster
::_reset()
{
}

// ------------------------------------------------------------------
void
process_cluster
::_finalize()
{
}

// ------------------------------------------------------------------
void
process_cluster
::_step()
{
  VITAL_THROW( process_exception );
}

// ------------------------------------------------------------------
void
process_cluster
::_reconfigure( kwiver::vital::config_block_sptr const& conf )
{
  kwiver::vital::config_block_keys_t const tunable_keys = available_tunable_config();

  for ( priv::config_map_t::value_type const& config_mapping : d->config_map )
  {
    name_t const& name_ = config_mapping.first;
    priv::config_mappings_t const& mappings = config_mapping.second;

    kwiver::vital::config_block_sptr const provide_conf =
      kwiver::vital::config_block::empty_config();

    for ( priv::config_mapping_t const& mapping : mappings )
    {
      kwiver::vital::config_block_key_t const& key = mapping.first;

      if ( !std::count( tunable_keys.begin(), tunable_keys.end(), key ) )
      {
        continue;
      }

      kwiver::vital::config_block_key_t const& mapped_key = mapping.second;

      kwiver::vital::config_block_value_t const& value =
        config_value< kwiver::vital::config_block_value_t >( key );

      provide_conf->set_value( mapped_key, value );
    }

    process_t const proc = d->processes[ name_ ];

    // Grab the new subblock for the process.
    kwiver::vital::config_block_sptr const proc_conf = conf->subblock( name_ );

    // Reconfigure the given process normally.
    proc->reconfigure( proc_conf );
    // Overwrite any provided configuration values which may be read-only.
    proc->reconfigure_with_provides( provide_conf );
  }

  process::_reconfigure( conf );
}

// ------------------------------------------------------------------
process::properties_t
process_cluster
::_properties() const
{
  properties_t base_properties = process::_properties();

  base_properties.insert(property_cluster);

  return base_properties;
}

// ==================================================================
process_cluster::priv
::priv()
  : config_map()
  , processes()
  , input_mappings()
  , output_mappings()
  , internal_connections()
  , m_logger( kwiver::vital::get_logger( "sprokit.process_cluster" ) )
{
}

// ------------------------------------------------------------------
process_cluster::priv
::~priv()
{
}

// ------------------------------------------------------------------
bool
process_cluster::priv
::has_name(name_t const& name) const
{
  return (0 != processes.count(name));
}

// ------------------------------------------------------------------
void
process_cluster::priv
::ensure_name(name_t const& name) const
{
  if (!has_name(name))
  {
    VITAL_THROW( no_such_process_exception,
                 name);
  }
}

// ------------------------------------------------------------------
/*
 * This function creates a new name from the cluster name and process name.
 * The result is <cluster>/<proc>
 */
process::name_t
convert_name(process::name_t const& cluster_name,
             process::name_t const& process_name)
{
  static process::name_t const sep = process::name_t("/");

  process::name_t const full_name = cluster_name + sep + process_name;

  return full_name;
}

} // end namespace
