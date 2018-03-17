/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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

#ifndef _VITAL_TYPES_TIMESTAMP_CONFIG_H
#define _VITAL_TYPES_TIMESTAMP_CONFIG_H

#include <vital/types/timestamp.h>
#include <vital/config/config_block.h>

#include <sstream>

namespace kwiver {
namespace vital {

/**
 * @brief Convert string to timestamp for config block.
 *
 * This function is a specialization of the config type converter. It
 * converts a string to a native timestamp object.
 *
 * This is primarily used to supply *default* behaviour for a
 * timestamp when getting data from the confiug.
 *
 * @param value String representation of timestamp.
 *
 * @return Native timestamp.
 */
template<>
inline
timestamp
config_block_get_value_cast( config_block_value_t const& value )
{
  timestamp obj;

  std::stringstream str; // add string to stream
  str << value;

  time_us_t t;
  str >> t;
  obj.set_time_usec( t );

  frame_id_t f;
  str >> f;
  obj.set_frame( f );

  return obj;
}


/**
 * @brief Convert timestamp to string for config block.
 *
 * This function is a specialization of the config type converter. It
 * converts a timestamp to a string representation for use in a config
 * block.
 *
 * @param value Timestamp to be converted to a string.
 *
 * @return String representation of timestamp.
 */
template<>
inline
config_block_value_t
config_block_set_value_cast( timestamp const& value )
{
  std::stringstream str;

  str << value.get_time_usec() << " " << value.get_frame();

  return str.str();
}

} } // end namespace

#endif /* _VITAL_TYPES_TIMESTAMP_CONFIG_H */
