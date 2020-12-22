// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of hex dump functionality
 */

#ifndef VITAL_HEXDUMP_H
#define VITAL_HEXDUMP_H

#include <string>
#include <ostream>
#include <vital/util/vital_util_export.h>

namespace kwiver {
namespace vital {

/**
 * @brief Format a string as a traditional hex dump for debugging
 *
 * See this Stackexchange entry for further discussion:
 * https://codereview.stackexchange.com/questions/165120/printing-hex-dumps-for-diagnostics
 *
 * @param os Output stream for the formatted string
 * @param buffer Input buffer to dump
 * @param bufsize Length of input buffer or number of bytes to dump
 * @param showPrintableCharacter \b true will print printable characters along with hex
 */
VITAL_UTIL_EXPORT std::ostream& hex_dump( std::ostream& os,
                                          const void*   buffer,
                                          std::size_t   bufsize,
                                          bool          showPrintableChars = true );

struct VITAL_UTIL_EXPORT hexDump
{
  const void* buffer;
  std::size_t bufsize;

  hexDump( const void* buf, std::size_t bufsz )
    : buffer{ buf }
    , bufsize{ bufsz }
  {}

  friend std::ostream& operator<<( std::ostream& out, const hexDump& hd )
  {
    return hex_dump( out, hd.buffer, hd.bufsize, true );
  }
};

} // namespace vital
} // namespace kwiver

#endif // VITAL_HEXDUMP_H
