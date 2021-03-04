// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "token_type_config.h"

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
token_type_config::
token_type_config( kwiver::vital::config_block_sptr blk )
  : token_type ("CONFIG")
  , m_config( blk )
{ }

// ----------------------------------------------------------------
token_type_config::
 ~token_type_config()
{ }

// ----------------------------------------------------------------
bool
token_type_config::
lookup_entry (kwiver::vital::config_block_key_t const& name,
              std::string& result) const
{
  bool retcode( true );

  try
  {
    result = m_config->get_value< std::string >( name );
  }
  catch ( kwiver::vital::config_block_exception& )
  {
    retcode = false; // not found
  }
  catch ( ... )
  {
    retcode = false; // not found
  }

  return retcode;
}

} } // end namespace
