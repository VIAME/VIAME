/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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

#include <stdarg.h>  // For va_start, etc.
#include <memory>    // For std::unique_ptr

namespace kwiver {
namespace vital {

/**
 * @brief Printf style formatting for std::string
 *
 * @param fmt_str Formatting string using embedded printf format specifiers.
 *
 * @return Formatted string.
 */
inline std::string
string_format( const std::string fmt_str, ... )
{
  int final_n, n = ( (int)fmt_str.size() ) * 2; /* Reserve two times as much as the length of the fmt_str */
  std::string str;
  std::unique_ptr< char[] > formatted;
  va_list ap;

  while ( 1 )
  {
    formatted.reset( new char[n] );   /* Wrap the plain char array into the unique_ptr */
    strcpy( &formatted[0], fmt_str.c_str() );
    va_start( ap, fmt_str );
    final_n = vsnprintf( &formatted[0], n, fmt_str.c_str(), ap );
    va_end( ap );
    if ( ( final_n < 0 ) || ( final_n >= n ) )
    {
      n += abs( final_n - n + 1 );
    }
    else
    {
      break;
    }
  }

  return std::string( formatted.get() );
}

} } // end namespace
