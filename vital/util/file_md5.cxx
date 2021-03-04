// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "file_md5.h"

#include <kwiversys/MD5.h>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iomanip>

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
      //gcount() should never be negative, but fortify does not seem to
      //know that.  And read never fills in a /0 value at the end.
      if ( is.gcount() > 0)
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
    for ( size_t i = 0; i < digest_size; ++i )
    {
      oss << std::hex << static_cast<unsigned int>(digest[ i ]);
    }
  }
  return oss.str();
}

} // ...vital
} // ...kwiver
