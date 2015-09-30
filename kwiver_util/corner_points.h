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

#ifndef KWIVER_CORNER_PTS_H
#define KWIVER_CORNER_PTS_H

#include <vital/types/geo_lat_lon.h>
#include <vital/config/config_block.h>
#include <vital/vital_foreach.h>

#include <vector>
#include <stdio.h>
#include <iostream>
#include <sstream>

namespace kwiver {
namespace vital {

typedef std::vector < kwiver::vital::geo_lat_lon > corner_points_t;

template<>
inline
corner_points_t
config_block_get_value_cast( config_block_value_t const& value )
{
  // This is not robust and should be rewritten as such.
  double val[8];
  corner_points_t obj;

  // this is ugly (lat lon pairs)
  sscanf( value.c_str(), "%lf %lf %lf %lf %lf %lf %lf %lf",
          &val[0], &val[1],
          &val[2], &val[3],
          &val[4], &val[5],
          &val[6], &val[7] );

  // process 4 points
  for (int i = 0; i < 4; ++i)
  {
    obj.push_back( kwiver::vital::geo_lat_lon( val[i*2], val[i*2 +1] ) );
  } // end for

  return obj;
}


// ------------------------------------------------------------------
template<>
inline
config_block_value_t
config_block_set_value_cast( corner_points_t const& value )
{
  std::stringstream str_result;

  str_result.precision( 20 );

  VITAL_FOREACH( geo_lat_lon const& pt, value )
  {
    str_result << pt.get_latitude() << " " << pt.get_longitude() << " ";
  }

  return str_result.str();
}

} } // end namespace

#endif /* KWIVER_CORNER_PTS_H */
