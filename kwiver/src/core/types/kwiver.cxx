/*ckwg +5
 * Copyright 2014 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "kwiver.h"

#include <stdio.h>
#include <iostream>


namespace kwiver
{

// This is not robust and should be rewritten as such.
std::istream& operator>> ( std::istream& str, corner_points_t& obj )
{
  double val[8];

  std::string line;
  std::getline( str, line );

  obj.clear();

  // this is ugly (lat lon pairs)
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
