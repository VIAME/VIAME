// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * @file   cluster_creator.cxx
 * @brief  Implementation for cluster_creator class.
 */

#include "cluster_creator.h"

#include "loaded_cluster.h"
#include "provided_by_cluster.h"

#include <vital/util/tokenize.h>

#include <algorithm>
#include <memory>

namespace sprokit {

cluster_creator
::cluster_creator( cluster_bakery const& bakery )
  : m_bakery( bakery )
  , m_logger( kwiver::vital::get_logger( "sprokit.create_pipeline" ) )

{
  bakery_base::config_decls_t default_configs = m_bakery.m_configs;

  m_default_config = bakery_base::extract_configuration_from_decls( default_configs );
}

cluster_creator
::~cluster_creator()
{
}

// ------------------------------------------------------------------
/*
 * This method creates a cluster. It is called when a cluster is
 * instantiated when the pipeline is being built.
 */
process_t
cluster_creator
::operator()( kwiver::vital::config_block_sptr const& config ) const
{
  bakery_base::config_decls_t all_configs = m_bakery.m_configs;

  process::type_t const& type = m_bakery.m_type;

  process::names_t proc_names;

  for( bakery_base::process_decl_t const & proc_decl : m_bakery.m_processes )
  {
    process::name_t const& proc_name = proc_decl.first;

    proc_names.push_back( proc_name );
  }

  provided_by_cluster const mapping_filter( type, proc_names );

  bakery_base::config_decls_t mapped_decls;

  // Copy out configuration settings which are mapped by the cluster.
  std::copy_if( all_configs.begin(), all_configs.end(), std::back_inserter( mapped_decls ), mapping_filter );

  // Append the given configuration to the declarations from the parsed blocks.
  kwiver::vital::config_block_keys_t const& keys = config->available_values();
  for( kwiver::vital::config_block_key_t const & key : keys )
  {
    kwiver::vital::config_block_value_t const value = config->get_value< kwiver::vital::config_block_value_t > ( key );
    bool const is_read_only = config->is_read_only( key );

    kwiver::vital::source_location loc;
    config->get_location( key, loc );
    bakery_base::config_info_t const info = bakery_base::config_info_t( value,
                                                                        is_read_only,
                                                                        false, // relative path
                                                                        loc );

    kwiver::vital::config_block_key_t const full_key =
            kwiver::vital::config_block_key_t( type ) +
             kwiver::vital::config_block::block_sep() + key;
    bakery_base::config_decl_t const decl = bakery_base::config_decl_t( full_key, info );

    all_configs.push_back( decl );
  } // end foreach

  kwiver::vital::config_block_sptr const full_config = bakery_base::extract_configuration_from_decls( all_configs );

  typedef std::shared_ptr< loaded_cluster > loaded_cluster_t;

  // Pull out the main config block to the top-level.
  kwiver::vital::config_block_sptr const cluster_config = full_config->subblock_view( type );
  full_config->merge_config( cluster_config );

  // Create object that will hold the cluster's processes and connections
  loaded_cluster_t const cluster = std::make_shared< loaded_cluster > ( full_config );

  cluster_bakery::opt_cluster_component_info_t const& opt_info = m_bakery.m_cluster;

  if ( ! opt_info )
  {
    static std::string const reason = "Failed to catch missing cluster block earlier";
    throw std::logic_error( reason );
  }

  cluster_bakery::cluster_component_info_t const& info = *opt_info;

  kwiver::vital::config_block_sptr const main_config = m_default_config->subblock_view( type );

  // Declare configuration values.
  for( cluster_config_t const & conf : info.m_configs )
  {
    config_value_t const& config_value = conf.config_value;
    kwiver::vital::config_block_keys_t const& key_path = config_value.key_path;
    kwiver::vital::config_block_key_t const& key = bakery_base::flatten_keys( key_path );
    kwiver::vital::config_block_value_t const& value = main_config->get_value< kwiver::vital::config_block_value_t > ( key );
    kwiver::vital::config_block_description_t const& description = conf.description;
    bool tunable = false;

    if ( ! config_value.flags.empty() )
    {
      config_flags_t const& flags = config_value.flags;
      tunable = ( 0 != std::count( flags.begin(), flags.end(), bakery_base::flag_tunable ) );
    }

    cluster->declare_configuration_key(
      key,
      value,
      description,
      tunable );
  } // end for

  // Add config mappings.
  for( bakery_base::config_decl_t const & decl : mapped_decls )
  {
    kwiver::vital::config_block_key_t const& key = decl.first;
    bakery_base::config_info_t const& mapping_info = decl.second;

    kwiver::vital::config_block_value_t const value = mapping_info.value;

    kwiver::vital::config_block_keys_t mapped_key_path;
    kwiver::vital::config_block_keys_t source_key_path;

    /// \bug Does not work if (kwiver::vital::config_block::block_sep.size() != 1).
    kwiver::vital::tokenize( key, mapped_key_path,
                             kwiver::vital::config_block::block_sep(),
                             kwiver::vital::TokenizeTrimEmpty );
    /// \bug Does not work if (kwiver::vital::config_block::block_sep.size() != 1).
    kwiver::vital::tokenize( value, source_key_path,
                             kwiver::vital::config_block::block_sep(),
                             kwiver::vital::TokenizeTrimEmpty );

    if ( mapped_key_path.size() < 2 )
    {
      LOG_WARN( m_logger, "Mapped key path is less than two elements" );
      continue;
    }

    if ( source_key_path.size() < 2 )
    {
      LOG_WARN( m_logger, "Source key path is less than two elements" );
      continue;
    }

    kwiver::vital::config_block_key_t const mapped_name = mapped_key_path[0];
    mapped_key_path.erase( mapped_key_path.begin() );

    kwiver::vital::config_block_key_t const mapped_key = bakery_base::flatten_keys( mapped_key_path );

    source_key_path.erase( source_key_path.begin() );
    kwiver::vital::config_block_key_t const source_key = bakery_base::flatten_keys( source_key_path );

    cluster->map_config( source_key, mapped_name, mapped_key );
  } // end for

  // Add processes.
  for( bakery_base::process_decl_t const & proc_decl : m_bakery.m_processes )
  {
    process::name_t const& proc_name = proc_decl.first;
    process::type_t const& proc_type = proc_decl.second;

    kwiver::vital::config_block_sptr const proc_config = full_config->subblock_view( proc_name );

    cluster->add_process( proc_name, proc_type, proc_config );
  } // end for

  // Add input ports.
  {
    process::port_flags_t const input_flags;

    for( cluster_input_t const & input : info.m_inputs )
    {
      process::port_description_t const& description = input.description;
      process::port_t const& port = input.from;

      cluster->declare_input_port(
        port,       /// \todo How to declare a port's type?
        process::type_any,
        input_flags,
        description );

      process::port_addrs_t const& addrs = input.targets;

      for( process::port_addr_t const & addr : addrs )
      {
        process::name_t const& mapped_name = addr.first;
        process::port_t const& mapped_port = addr.second;

        cluster->map_input(
          port,
          mapped_name,
          mapped_port );
      }
    } // end for
  }

  // Add output ports.
  {
    process::port_flags_t const output_flags;

    for( cluster_output_t const & output : info.m_outputs )
    {
      process::port_description_t const& description = output.description;
      process::port_t const& port = output.to;

      cluster->declare_output_port(
        port,
        /// \todo How to declare a port's type?
        process::type_any,
        output_flags,
        description );

      process::port_addr_t const& addr = output.from;

      process::name_t const& mapped_name = addr.first;
      process::port_t const& mapped_port = addr.second;

      cluster->map_output(
        port,
        mapped_name,
        mapped_port );
    } // end for
  }

  // Add connections.
  for( process::connection_t const & connection : m_bakery.m_connections )
  {
    process::port_addr_t const& upstream_addr = connection.first;
    process::port_addr_t const& downstream_addr = connection.second;

    process::name_t const& upstream_name = upstream_addr.first;
    process::port_t const& upstream_port = upstream_addr.second;
    process::name_t const& downstream_name = downstream_addr.first;
    process::port_t const& downstream_port = downstream_addr.second;

    cluster->connect( upstream_name, upstream_port,
                      downstream_name, downstream_port );
  } // end for

  return cluster; // return process cluster
}

} // end namespace sprokit
