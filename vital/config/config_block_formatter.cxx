/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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

#include "config_block_formatter.h"

#include "format_config_block.h"

#include <vital/util/string.h>

#include <sstream>
#include <iostream>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------------------
config_block_formatter::
config_block_formatter( const config_block_sptr config )
  : m_config( config )
  , m_gen_source_loc( true )
{ }

// ----------------------------------------------------------------------------
void config_block_formatter::
set_prefix( const std::string& pfx )
{
  m_prefix = pfx;
}

// ----------------------------------------------------------------------------
void config_block_formatter::
generate_source_loc( bool opt )
{
  m_gen_source_loc = opt;
}

// ------------------------------------------------------------------

void config_block_formatter::
print( std::ostream& str )
{
  kwiver::vital::config_block_keys_t all_keys = m_config->available_values();

  for( kwiver::vital::config_block_key_t key : all_keys )
  {
    std::string ro;

    auto const val = m_config->get_value< kwiver::vital::config_block_value_t > ( key );

    if ( m_config->is_read_only( key ) )
    {
      ro = "[RO]";
    }

    str << m_prefix << key << ro << " = " << val;

    if ( m_gen_source_loc )
    {
      // Add location information if available
      std::string file;
      int line(0);
      if ( m_config->get_location( key, file, line ) )
      {
        str << m_prefix << "  (" << file << ":" << line << ")";
      }
    }
    str << std::endl;
  }
}
}
}   // end namespace
