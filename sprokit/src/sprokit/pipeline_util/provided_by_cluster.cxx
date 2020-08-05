/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 * @file   provided_by_cluster.cxx
 * @brief  Implementation for provided_by_cluster class.
 */

#include "provided_by_cluster.h"

#include <vital/util/string.h>
#include <vital/util/tokenize.h>

namespace sprokit {

// ------------------------------------------------------------------
provided_by_cluster
::provided_by_cluster(process::type_t const& name, process::names_t const& procs)
  : m_name(name)
  , m_procs(procs)
{
}


// ------------------------------------------------------------------
provided_by_cluster
::~provided_by_cluster()
{
}


// ------------------------------------------------------------------
/**
 * @brief Determine if config entry is provided by cluster.
 *
 *
 * @return \b true if provided by cluster; \b false otherwise
 */
bool
provided_by_cluster
::operator()( bakery_base::config_decl_t const& decl ) const
{
  bakery_base::config_info_t const& info = decl.second;

  // Mapped configurations must be read-only.
  if ( ! info.read_only )
  {
    return false;
  }

  // Mapped configurations must be a provider_config request.
  kwiver::vital::config_block_value_t const value = info.value;

  // It must be mapped to the the actual cluster.
  if ( ! kwiver::vital::starts_with( value, m_name +
                   kwiver::vital::config_block::block_sep() ) )
  {
    return false;
  }

  /**
   * \todo There should be at least a warning that if the target is being
   * provided by a tunable parameter on the cluster that this will likely not
   * work as intended.
   */

  kwiver::vital::config_block_key_t const& key = decl.first;
  kwiver::vital::config_block_keys_t key_path;

  // Split path into components
  kwiver::vital::tokenize( key, key_path,
                           kwiver::vital::config_block::block_sep(),
                           kwiver::vital::TokenizeTrimEmpty );

  // Is the first component a process name
  bool const is_proc = ( 0 != std::count( m_procs.begin(), m_procs.end(), key_path[0] ) );

  if ( ! is_proc )
  {
    // We can't map to non-processes.
    return false;
  }

  return true;
}

} // end namespace sprokit
