// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * @file   cluster_bakery.cxx
 * @brief  Implementation for cluster_bakery class.
 */

#include "cluster_bakery.h"

#include "pipe_bakery_exception.h"
#include "cluster_splitter.h"

namespace sprokit {

// ------------------------------------------------------------------
cluster_bakery
::cluster_bakery()
  : bakery_base()
{
}

// ------------------------------------------------------------------
/*
 * This method processes a cluster pipe block which contains a vector
 * of cluster configs, input and output specs.
 */
void
cluster_bakery
::operator()( cluster_pipe_block const& cluster_block_ )
{
  // If there is another cluster block already processed, then this is
  // an error.  Only one cluster can be baked at a time.
  if ( m_cluster )
  {
    VITAL_THROW( multiple_cluster_blocks_exception );
  }

  m_type = cluster_block_.type;
  m_description = cluster_block_.description;

  cluster_component_info_t cluster;
  cluster_subblocks_t const& subblocks = cluster_block_.subblocks;
  cluster_splitter splitter( cluster );

  // Run splitter over vector of blocks to make collections by block type.
  // These lists end up in "cluster"
  for ( auto sb : subblocks )
  {
    std::visit( splitter, sb );
  }

  m_cluster = cluster;

  // Even though this is called a map, it is really a vector.
  cluster_component_info_t::config_maps_t const& values = cluster.m_configs;

  // Process all cluster config entries and convert to internal
  // representation for config entries.
  // Result ends up in this->m_configs
  for( cluster_config_t const & value : values )
  {
    config_value_t const& config_value = value.config_value;

    register_config_value( m_type, config_value );
  }
}

// ------------------------------------------------------------------
cluster_bakery::cluster_component_info_t
::cluster_component_info_t()
{
}

cluster_bakery::cluster_component_info_t
::~cluster_component_info_t()
{
}

} // end namespace sprokit
