// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "scorable_mgrs.h"

#include <utility>
#include <limits>
#include <stdexcept>
#include <cmath>

#include <vital/logger/logger.h>
#include <arrows/proj/geo_map.h>

using std::istream;
using std::numeric_limits;
using std::ostream;
using std::runtime_error;
using std::sqrt;
using std::string;
using std::vector;
using std::ios;

static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

double kwiver::track_oracle::scorable_mgrs::INF_DISTANCE = numeric_limits<double>::max();

namespace // anon
{
istream& set_bad( istream& is )
{
  is.setstate( ios::failbit );
  return is;
}

} // anon

namespace kwiver
{
namespace track_oracle
{

void
scorable_mgrs
::init( double e, double n, int z )
{
  this->zone[ DEFAULT_ZONE ] = z;
  this->easting[ DEFAULT_ZONE ] = e;
  this->northing[ DEFAULT_ZONE ] = n;
  this->entry_valid[ DEFAULT_ZONE ] = true;
  this->entry_valid[ ALT_ZONE ] = false;
  this->valid = true;
}

scorable_mgrs
::scorable_mgrs( double lat, double lon )
{
  double e,n;
  int z;
  bool nh;

  kwiver::arrows::proj::geo_map gm;
  gm.latlon_to_utm( lat, lon, e, n, z, nh );
  this->init( e, n, z );
}

scorable_mgrs
::scorable_mgrs( double easting, double northing, int zone, bool north_hemi )
{
  this->init( easting, northing, zone );
}

vector< int >
scorable_mgrs
::align_zones( const scorable_mgrs& other ) const
{
  if (( ! this->valid ) || ( ! other.valid ))
  {
    throw runtime_error( "Aligning uninitialized MGRS positions?\n" );
  }

  vector< int > ret( static_cast<int>( N_ZONES ), static_cast<int>( N_ZONES ));
  for (int p1 = ZONE_BEGIN; p1 < N_ZONES; ++p1)
  {
    for (int p2 = ZONE_BEGIN; p2 < N_ZONES; ++p2)
    {
      if (( this->entry_valid[ p1 ] ) &&
          ( other.entry_valid[ p2 ] ) &&
          (this->zone[p1] == other.zone[p2]) )
      {
        ret[ p1 ] = p2;
      }
    }
  }
  return ret;
}

double
scorable_mgrs
::diff( const scorable_mgrs& pos1, const scorable_mgrs& pos2 )
{
  vector< int > zone_map = pos1.align_zones( pos2 );
  return diff( pos1, pos2, zone_map );
}

double
scorable_mgrs
::diff( const scorable_mgrs& pos1, const scorable_mgrs& pos2, const vector<int>& zone_map )
{
  if ( pos1.valid && pos2.valid )
  {
    // loop over the alignment options until have a match
    for (int p1 = ZONE_BEGIN; p1 < N_ZONES; ++p1)
    {
      int p2 = zone_map[ p1] ;
      if (p2 == N_ZONES) continue;

      if ( pos1.entry_valid[ p1 ] && pos2.entry_valid[ p2 ] )
      {
        // we have matching zones; return euclidian distance in meters
        double dN = pos1.northing[p1] - pos2.northing[p2];
        double dE = pos1.easting[p1] - pos2.easting[p2];
        return sqrt( (dN*dN) + (dE*dE) );
      }
      else
      {
        throw runtime_error( "Invalid MGRS comparison computing difference" );
      }
    }
  }
  // no matches, return infinity
  return INF_DISTANCE;
}

bool
scorable_mgrs
::operator==( const scorable_mgrs& other) const
{
  // valid never equals invalid
  if (this->valid != other.valid) return false;
  // invalid always equals invalid
  if (this->valid == false) return true;

  // the usual issues of equality of doubles apply ... we'll
  // go with bitwise identity for now; anything else implies
  // the user is manipulating the values for reasons we cannot
  // know here.

  for (size_t i = ZONE_BEGIN; i < N_ZONES; ++i)
  {
    if (this->entry_valid[i] != other.entry_valid[i] ) return false;
    if (this->zone[i] != other.zone[i]) return false;
    if (this->northing[i] != other.northing[i]) return false;
    if (this->easting[i] != other.easting[i]) return false;
  }
  return true;
}

ostream&
operator<<( ostream& os,
            const scorable_mgrs& m )
{
  os << "mgrs: ";
  if ( ! m.valid )
  {
    os << " (invalid)";
  }
  else
  {
    os << "default v/z/n/e: " << m.entry_valid[ scorable_mgrs::DEFAULT_ZONE ] << " "
       << m.zone[ scorable_mgrs::DEFAULT_ZONE ] << " "
       << m.northing[ scorable_mgrs::DEFAULT_ZONE ] << " "
       << m.easting[ scorable_mgrs::DEFAULT_ZONE ] << " "
       << "alt v/z/n/e: " << m.entry_valid[ scorable_mgrs::ALT_ZONE ] << " "
       << m.zone[ scorable_mgrs::ALT_ZONE ] << " "
       << m.northing[ scorable_mgrs::ALT_ZONE ] << " "
       << m.easting[ scorable_mgrs::ALT_ZONE ];
  }
  return os;
}

istream&
operator>>( istream& is,
            scorable_mgrs& m )
{
  string tmp;
  if ( (! ( is >> tmp)) && (tmp != "mgrs:")) return set_bad( is );
  if (! ( is >> tmp))  return set_bad( is );
  if ( tmp == "(invalid)" )
  {
    m.valid = false;
    return is;
  }
  if ( (! ( is >> tmp)) && (tmp != "default")) return set_bad( is );
  if ( (! ( is >> tmp)) && (tmp != "v/z/n/e:")) return set_bad( is );
  if (! ( is >> m.entry_valid[ scorable_mgrs::DEFAULT_ZONE ])) return set_bad( is );
  if (! ( is >> m.zone[ scorable_mgrs::DEFAULT_ZONE ])) return set_bad( is );
  if (! ( is >> m.northing[ scorable_mgrs::DEFAULT_ZONE ])) return set_bad( is );
  if (! ( is >> m.easting[ scorable_mgrs::DEFAULT_ZONE ])) return set_bad( is );
  if ( (! ( is >> tmp)) && (tmp != "alt")) return set_bad( is );
  if ( (! ( is >> tmp)) && (tmp != "v/z/n/e:")) return set_bad( is );
  if (! ( is >> m.entry_valid[ scorable_mgrs::ALT_ZONE ])) return set_bad( is );
  if (! ( is >> m.zone[ scorable_mgrs::ALT_ZONE ])) return set_bad( is );
  if (! ( is >> m.northing[ scorable_mgrs::ALT_ZONE ])) return set_bad( is );
  if (! ( is >> m.easting[ scorable_mgrs::ALT_ZONE ])) return set_bad( is );
  m.valid = true;
  return is;
}

void
scorable_mgrs
::mark_zone_invalid( size_t z )
{
  switch (z )
  {
  case DEFAULT_ZONE:
  case ALT_ZONE:
    this->entry_valid[ z ] = false;
    this->zone[ z ] = -1;
    this->northing[ z ] = 0.0;
    this->easting[ z ] = 0.0;
    break;
  default:
    LOG_ERROR( main_logger, "Mark zone invalid on bad zone " << z );
    return;
  }

  if ( (! this->entry_valid[ DEFAULT_ZONE ]) &&
       (! this->entry_valid[ ALT_ZONE ]))
  {
    this->valid = false;
  }
}

int
scorable_mgrs
::find_zone( int z ) const
{
  if ( ! this->valid ) return N_ZONES;
  if ( this->entry_valid[ DEFAULT_ZONE ] && this->zone[ DEFAULT_ZONE ] == z) return DEFAULT_ZONE;
  if ( this->entry_valid[ ALT_ZONE ] && this->zone[ ALT_ZONE ] == z) return ALT_ZONE;
  return N_ZONES;
}

} // ...track_oracle
} // ...kwiver
