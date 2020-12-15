// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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

  time_usec_t t;
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
