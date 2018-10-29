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
#include "hex_dump.h"

#include <iomanip>
#include <cctype>
#include <algorithm>

namespace kwiver {
namespace vital  {

std::ostream&
hex_dump( std::ostream& os,
          const void* buffer,
          std::size_t bufsize,
          bool showPrintableChars )
{
  if ( buffer == nullptr )
  {
    return os;
  }

  auto oldFormat = os.flags();
  auto oldFillChar = os.fill();
  constexpr std::size_t maxline { 8 };
  size_t offset { 0 };

  // create a place to store text version of string
  char renderString[maxline + 1];
  char* rsptr { renderString };

  // convenience cast
  const unsigned char* buf { reinterpret_cast< const unsigned char* > ( buffer ) };

  os << std::setw( 4 )  << std::setfill( '0' ) << std::hex
     << offset << ": ";
  ++offset;

  for ( std::size_t linecount = maxline; bufsize; --bufsize, ++buf, ++offset )
  {
    os  << std::setw( 2 ) << std::setfill( '0' ) << std::hex
        << static_cast< unsigned > ( *buf ) << ' ';

    *rsptr++ = std::isprint( *buf ) ? *buf : '.';

    if ( --linecount == 0 )
    {
      *rsptr++ = '\0';        // terminate string
      if ( showPrintableChars )
      {
        os << " | " << renderString;
      }
      os << '\n';

      os << std::setw( 4 )  << std::setfill( '0' ) << std::hex
         << offset << ": ";

      rsptr = renderString;

      linecount = std::min( maxline, bufsize );
    }
  } //end for

  // emit newline if we haven't already
  if ( rsptr != renderString )
  {
    if ( showPrintableChars )
    {
      for ( *rsptr++ = '\0'; rsptr != &renderString[maxline + 1]; ++rsptr )
      {
        os << "   ";
      }
      os << " | " << renderString;
    }
    os << '\n';
  }

  // restore context
  os.fill( oldFillChar );
  os.flags( oldFormat );
  return os;
} // hex_dump


} // namespace vital
} // namespace kwiver
