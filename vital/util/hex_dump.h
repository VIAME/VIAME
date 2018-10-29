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
