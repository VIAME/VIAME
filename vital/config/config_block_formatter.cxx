// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
