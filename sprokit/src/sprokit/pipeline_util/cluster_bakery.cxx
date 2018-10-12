/*ckwg +29
 * Copyright 2016-2018 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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


cluster_bakery
::~cluster_bakery()
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
    kwiver::vital::visit( splitter, sb );
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
