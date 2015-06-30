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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS [yas] elisp error!AS IS''
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

#include "sprokit_type_traits.h"

#include <stdio.h>
#include <iostream>

namespace kwiver
{

// This is not robust and should be rewritten as such.
// needs to handle a real polygon
std::istream& operator>> ( std::istream& str, geo_polygon_t& obj )
{
  double val[8];

  std::string line;
  std::getline( str, line );

  obj.clear();

  //+ this is ugly (lat lon pairs)
  sscanf( line.c_str(), "%lf %lf %lf %lf %lf %lf %lf %lf",
          &val[0], &val[1],
          &val[2], &val[3],
          &val[4], &val[5],
          &val[6], &val[7] );

  // process 4 points
  for (int i = 0; i < 4; ++i)
  {
    obj.push_back( kwiver::geo_lat_lon( val[i*2], val[i*2 +1] ) );
  } // end for

  return str;
}

} // end namespace
