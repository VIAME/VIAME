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
