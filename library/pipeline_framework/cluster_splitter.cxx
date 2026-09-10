// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * @file   cluster_splitter.cxx
 * @brief  Implementation for cluster_splitter class.
 */

#include "cluster_splitter.h"
#include "pipe_bakery_exception.h"

namespace sprokit {

  // ------------------------------------------------------------------
cluster_splitter
::cluster_splitter(cluster_bakery::cluster_component_info_t& info)
  : m_info(info)
{
}

cluster_splitter
::~cluster_splitter()
{
}

// ------------------------------------------------------------------
void
cluster_splitter
::operator () (cluster_config_t const& config_block)
{
  m_info.m_configs.push_back(config_block);
}

// ------------------------------------------------------------------
void
cluster_splitter
::operator () (cluster_input_t const& input_block)
{
  process::port_t const& port = input_block.from;

  if (m_input_ports.count(port))
  {
    VITAL_THROW( duplicate_cluster_input_port_exception, port);
  }

  m_input_ports.insert(port);

  m_info.m_inputs.push_back(input_block);
}

// ------------------------------------------------------------------
void
cluster_splitter
::operator () (cluster_output_t const& output_block)
{
  process::port_t const& port = output_block.to;

  if (m_output_ports.count(port))
  {
    VITAL_THROW( duplicate_cluster_output_port_exception, port);
  }

  m_output_ports.insert(port);

  m_info.m_outputs.push_back(output_block);
}

} // end namespace sprokit
