// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "config_block_formatter.h"

#include <viame/algorithm_framework/util/string.h>

#include <iostream>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
config_block_formatter
::config_block_formatter( const config_block_sptr config )
  : m_config( config )
{}

// ----------------------------------------------------------------------------

void
config_block_formatter
::print( std::ostream& str )
{
  kwiver::vital::config_block_keys_t all_keys = m_config->available_values();

  for( kwiver::vital::config_block_key_t key : all_keys )
  {
    std::string ro;

    auto const val =
      m_config->get_value< kwiver::vital::config_block_value_t >( key );

    if( m_config->is_read_only( key ) )
    {
      ro = "[RO]";
    }

    str << key << ro << " = " << val;

    // Where the value came from, when it came from a file.
    std::string file;
    int line( 0 );
    if( m_config->get_location( key, file, line ) )
    {
      str << "  (" << file << ":" << line << ")";
    }

    str << std::endl;
  }
}

} // namespace vital

}   // end namespace
