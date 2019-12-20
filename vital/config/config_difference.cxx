/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
