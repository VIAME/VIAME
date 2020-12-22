// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "config_difference.h"
#include <vital/util/string.h>
#include <vital/logger/logger.h>
/*
  Possible enhancements

  - Methods to help iterate through a config block given a list of keys.
  - Easy way to drill down to get source_loc for some entries.
 */

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
config_difference
::config_difference( const config_block_sptr reference, const config_block_sptr other )
{
  // keys that are in reference, but not in other.
  auto missing_keys = reference->difference_config( other );
  m_missing_keys = missing_keys->available_values();

  // keys that are in other, but not in reference
  auto extra_config = other->difference_config( reference );
  m_extra_keys = extra_config->available_values();
}

config_difference
::config_difference( config_block_keys_t const& reference, const config_block_sptr other )
{
  auto ref_blk = config_block::empty_config();

  // Make a fake config block
  for ( auto key : reference )
  {
    ref_blk->set_value( key, "X" );
  }

  // keys that are in reference, but not in other.
  auto missing_keys = ref_blk->difference_config( other );
  m_missing_keys = missing_keys->available_values();

  // keys that are in other, but not in reference
  auto extra_config = other->difference_config( ref_blk );
  m_extra_keys = extra_config->available_values();
}

config_difference::
~config_difference()
{ }

// ------------------------------------------------------------------
config_block_keys_t
config_difference::
extra_keys() const
{
  return m_extra_keys;
}

// ------------------------------------------------------------------
config_block_keys_t
config_difference::
unspecified_keys() const
{
  return m_missing_keys;
}

// ------------------------------------------------------------------
bool
config_difference
::warn_extra_keys( logger_handle_t logger ) const
{
  const auto key_list = this->extra_keys();
  if ( ! key_list.empty() )
  {
    // This may be considered an error in some cases
    LOG_WARN( logger, "Additional parameters found in config block that are not required or desired: "
              << kwiver::vital::join( key_list, ", " ) );
    return true;
  }

  return false;
}

// ------------------------------------------------------------------
bool
config_difference
::warn_unspecified_keys( logger_handle_t logger ) const
{
  const auto key_list = this->unspecified_keys();
  if ( ! key_list.empty() )
  {
    LOG_WARN( logger, "Parameters that were not supplied in the config, using default values: "
              << kwiver::vital::join( key_list, ", " ) );
    return true;
  }

  return false;
}

} } // end namespace
