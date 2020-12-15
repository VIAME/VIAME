// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
