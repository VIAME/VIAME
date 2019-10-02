/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
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

#include "file_md5.h"

#include <kwiversys/MD5.h>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>

using std::string;
using std::ifstream;
using std::ostringstream;

namespace kwiver {
namespace vital {

string
file_md5( const string& fn )
{
  ifstream is( fn.c_str(), ifstream::binary );
  if ( !is )
  {
    return "";
  }

  const size_t digest_size = 16;
  unsigned char digest[ digest_size ];

  {
    const size_t bufsize = 16384;
    unsigned char* buf = new unsigned char [ bufsize ];
    kwiversysMD5* md5 = kwiversysMD5_New();
    kwiversysMD5_Initialize( md5 );

    while ( is )
    {
      is.read( reinterpret_cast< char* >( buf ), bufsize );
      if ( is.gcount() )
      {
        kwiversysMD5_Append( md5, buf, is.gcount() );
      }
    }
    kwiversysMD5_Finalize( md5, digest );
    kwiversysMD5_Delete( md5 );
    delete[] buf;
  }

  ostringstream oss;
  {
    const size_t bufsize = 5;
    char buf[ bufsize ];
    for ( size_t i = 0; i < digest_size; ++i )
    {
      snprintf( buf, bufsize, "%x", digest[ i ] );
      oss << buf;
    }
  }
  return oss.str();
}

} // ...vital
} // ...kwiver
