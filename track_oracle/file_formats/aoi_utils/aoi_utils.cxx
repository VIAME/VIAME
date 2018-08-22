/*ckwg +5
 * Copyright 2014-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "aoi_utils.h"

#include <utility>
#include <vector>
#include <map>
#include <sstream>

#include <kwiversys/RegularExpression.hxx>

#include <vgl/vgl_point_2d.h>
#include <vgl/vgl_polygon.h>
#include <vgl/vgl_convex.h>

#include <track_oracle/track_scorable_mgrs/scorable_mgrs.h>


using std::vector;
using std::map;
using std::string;
using std::ostringstream;
using std::pair;
using std::make_pair;

namespace // anon
{

struct mgrs_aoi_t
{
  vgl_polygon<double> poly;
  int zone;
  mgrs_aoi_t(): zone(-1) {}
  mgrs_aoi_t( const vgl_polygon<double>& a, int z ): poly(a), zone(z) {}
};

struct pixel_aoi_t
{
  vgl_polygon<double> poly;
};


//
// Parse the string into a list of doubles and an aoi flavor.
//

void
parse_aoi_string( const string& s,
                  ::kwiver::track_oracle::aoi_utils::aoi_t::flavor_t& flavor,
                  vector< vgl_point_2d<double> >& points )
{
  flavor = ::kwiver::track_oracle::aoi_utils::aoi_t::INVALID;
  points.clear();

  kwiversys::RegularExpression pixel_geom_re( "(\\d+)x(\\d+)([\\+\\-]\\d+)([\\+\\-]\\d+)" );
  kwiversys::RegularExpression float_pair_re( "^\\s*\\:?\\s*([\\d\\+\\-\\.eE]+)\\s*\\,\\s*([\\d\\+\\-\\.eE]+)" );
  kwiversys::RegularExpression pixel_tag_re ( "^\\s*[Pp]" );

  // special case: is it a geometry string?
  if ( pixel_geom_re.find( s ))
  {
    int w = std::stoi( pixel_geom_re.match(1) );
    int h = std::stoi( pixel_geom_re.match(2) );
    int x = std::stoi( pixel_geom_re.match(3) );
    int y = std::stoi( pixel_geom_re.match(4) );
    flavor = ::kwiver::track_oracle::aoi_utils::aoi_t::PIXEL;
    points.push_back( vgl_point_2d<double>(x,   y   ));
    points.push_back( vgl_point_2d<double>(x+w, y   ));
    points.push_back( vgl_point_2d<double>(x+w, y+h ));
    points.push_back( vgl_point_2d<double>(x,   y+h ));
    return;
  }

  // otherwise, is it a pixel or a lat/lon?
  if ( pixel_tag_re.find( pixel_tag ))
  {
    flavor = ::kwiver::track_oracle::aoi_utils::aoi_t::PIXEL;
  }
  else
  {
    flavor = ::kwiver::track_oracle::aoi_utils::aoi_t::GEO;
  }

  // start picking off the numbers
#error "aoi_utils.cxx needs to be reworked vis-a-vis regex capabilities"

  // while ( std::regex_search( a, b, what, float_pair_re ) )
  // {
  //   double x = std::stod( what.str(1) );
  //   double y = std::stor( what.str(2) );
  //   points.push_back( vgl_point_2d<double>( x, y ));
  //   a = what[2].second;
  // }
}

//
// Given the set of corners, create a list of up to N_ZONES mgrs_polygons.
//

vector< mgrs_aoi_t >
create_geo_poly( const vector< ::kwiver::track_oracle::scorable_mgrs >& corners )

{
  size_t n_corners = corners.size();
  for (size_t i=0; i < n_corners; ++i)
  {
    if ( ! corners[i].valid )
    {
      throw ::kwiver::track_oracle::aoi_utils::aoi_exception( "AOI: invalid geo corner" );
    }
  }

  vector< mgrs_aoi_t > aoi_list;

  // this is a n-ways variant of scorable_mgrs::align_zones.  We want
  // up to N_ZONES vectors; each vector should have n_corners entries;
  // each entry is a zone index I such that for each corner, zone[I]
  // is the same zone and thus the UTM coordinates are commensurate.
  // Any vector with fewer than n_corners entries is invalid (no
  // single zone in either default or alt holds all corners.)  If
  // there are no valid vectors, throw.

  typedef pair< size_t, unsigned int> corner_zone_pair;  // first = corner (0..n_corner-1); second = zone
  typedef map< int, vector< corner_zone_pair > > zone_to_corner_map_t;
  typedef map< int, vector< corner_zone_pair > >::const_iterator zone_to_corner_map_cit;

  zone_to_corner_map_t zone_to_corner_map;
  for (size_t i=0; i<n_corners; ++i)
  {
    const ::kwiver::track_oracle::scorable_mgrs& s = corners[i];
    for (size_t j=0; j<::kwiver::track_oracle::scorable_mgrs::N_ZONES; ++j)
    {
      if ( ! s.entry_valid[j] ) continue;
      zone_to_corner_map[ s.zone[j] ].push_back( make_pair( i, j ));
    }
  }

  for (zone_to_corner_map_cit i=zone_to_corner_map.begin(); i != zone_to_corner_map.end(); ++i)
  {
    const vector< corner_zone_pair >& v = i->second;
    if (v.size() != n_corners) continue;

    // create the vector of (easting, northing) points
    vector< vgl_point_2d<double> > pts;
    for (size_t j=0; j<v.size(); ++j)
    {
      const corner_zone_pair& czp = v[j];
      const ::kwiver::track_oracle::scorable_mgrs& s = corners[ czp.first ];
      if (! s.entry_valid[czp.second] )
      {
        throw ::kwiver::track_oracle::aoi_utils::aoi_exception( "AOI: invalid MGRS comparison constructing geo AOI" );
      }
      pts.push_back( vgl_point_2d<double>( s.easting[ czp.second ], s.northing[ czp.second ] ));
    }

    // convert to a convex hull

    mgrs_aoi_t a;
    a.zone = i->first;
    a.poly = vgl_convex_hull( pts );
    aoi_list.push_back( a );
  }

  if ( aoi_list.empty() )
  {
    throw ::kwiver::track_oracle::aoi_utils::aoi_exception( "AOI: geo AOI failed to generate any UTM polygons; perhaps no single zone holds all corners?" );
  }

  return aoi_list;
}

} // ...anon

namespace kwiver {
namespace track_oracle {

namespace aoi_utils {

//
// implementations for pixel and geo AOIs.
//

// base class

struct aoi_impl
{
protected:
  aoi_t::flavor_t f;

public:
  virtual aoi_t::flavor_t flavor() const { return f; }

  aoi_impl( aoi_t::flavor_t my_f): f(my_f) {}
  virtual ~aoi_impl() {}

  virtual string to_str() const = 0;
  virtual bool in_aoi( double x, double y ) const = 0;
};


// pixel AOI

struct pixel_aoi_impl: public aoi_impl
{
  pixel_aoi_t pixel_aoi;

  pixel_aoi_impl( aoi_t::flavor_t my_flavor,
                  const vector< vgl_point_2d<double> >& points )
    : aoi_impl( my_flavor )
  {
    this->pixel_aoi.poly = vgl_convex_hull( points );
  }

  virtual ~pixel_aoi_impl() {}

  virtual string to_str() const
  {
    ostringstream oss;
    oss << "pixel-aoi:" << this->pixel_aoi.poly;
    return oss.str();
  }

  virtual bool in_aoi( double x, double y ) const
  {
    return this->pixel_aoi.poly.contains( x, y );
  }
};


// geo AOI

struct geo_aoi_impl: public aoi_impl
{
  vector< mgrs_aoi_t > mgrs_aoi_list;

  geo_aoi_impl( aoi_t::flavor_t my_flavor,
                  const vector< vgl_point_2d<double> >& points )
    : aoi_impl( my_flavor )
  {
    // convert the lon/lat pairs to scorable_mgrs and create the geo-polys
    vector< scorable_mgrs > corners;
    for (size_t i=0; i<points.size(); ++i)
    {
      corners.push_back( scorable_mgrs( geographic::geo_coords( points[i].y(), points[i].x() )));
    }
    this->mgrs_aoi_list = create_geo_poly( corners );
  }

  virtual ~geo_aoi_impl() {}

  virtual string to_str() const
  {
    ostringstream oss;
    oss << "geo-aoi:" << this->mgrs_aoi_list.size() << ":";
    for (size_t i=0; i<this->mgrs_aoi_list.size(); ++i)
    {
      oss << "z=" << this->mgrs_aoi_list[i].zone << ";" << this->mgrs_aoi_list[i].poly << ";";
    }
    return oss.str();
  }

  virtual bool in_aoi( double x, double y) const
  {
    scorable_mgrs probe( geographic::geo_coords( y, x ));
    // find a compatible zone
    int z1 = scorable_mgrs::N_ZONES;
    int aoi_z = scorable_mgrs::N_ZONES;
    for (size_t i=0; i<this->mgrs_aoi_list.size(); ++i)
    {
      for (int p1 = scorable_mgrs::ZONE_BEGIN; p1 < scorable_mgrs::N_ZONES; ++p1)
      {
        if (this->mgrs_aoi_list[i].zone == probe.zone[p1])
        {
          z1 = p1;
          aoi_z = i;
        }
      }
    }

    bool is_in_aoi = false;
    if ( (z1 != scorable_mgrs::N_ZONES) && (aoi_z != scorable_mgrs::N_ZONES))
    {
      const vgl_polygon<double>& box = this->mgrs_aoi_list[ aoi_z ].poly;
      if ( !probe.entry_valid[ z1 ])
      {
        throw aoi_exception( "AOI: invalid MGRS comparison checking frame_within_geo_aoi" );
      }
      is_in_aoi = box.contains( probe.easting[ z1 ], probe.northing[ z1 ] );
    }
    return is_in_aoi;
  }
};


//
// construct an invalid, empty AOI
//

aoi_t
::aoi_t()
  : p(0)
{
}

//
// construct a new AOI given the string.  May throw per set()'s
// semantics.
//

aoi_t
::aoi_t( const string& s )
  : p(0)
{
  this->set( s );
}

//
// destructor
//

aoi_t
::~aoi_t()
{
  delete this->p;
}


//
// Given a string as described in help_text(), set this AOI
// (removing the old one, if any).  May throw if the
// string can't be parsed or doesn't make sense.
//

void
aoi_t
::set( const string& s )
{
  vector< vgl_point_2d<double> > points;
  flavor_t tmp_flavor;
  parse_aoi_string( s, tmp_flavor, points );

  // expand 2-point lists into 4-point bounding boxes
  if ( points.size() == 2 )
  {
    vgl_point_2d<double> p1 = points[0];
    vgl_point_2d<double> p2 = points[1];
    points.push_back( vgl_point_2d<double>( p1.x(), p2.y() ));
    points.push_back( vgl_point_2d<double>( p2.x(), p1.y() ));
  }

  if ( points.size() < 3 )
  {
    ostringstream oss;
    oss << "AOI: string '" << s << "' parsed into " << points.size() << " points; need at least 3";
    throw aoi_exception( oss.str() );
  }

  switch (tmp_flavor)
  {
  case PIXEL:
    this->p = new pixel_aoi_impl( tmp_flavor, points );
    break;
  case GEO:
    this->p = new geo_aoi_impl( tmp_flavor, points );
    break;
  default:
    ostringstream oss;
    oss << "AOI: unhandled ctor for flavor " << tmp_flavor << "?";
    throw aoi_exception( oss.str() );
  }
}

//
// What's the flavor of this AOI?
//

aoi_t::flavor_t
aoi_t
::flavor() const
{
  return ( this->p )
    ? this->p->flavor()
    : INVALID;
}

//
// What's the string representation of this AOI?
//

string
aoi_t
::to_str() const
{
  return ( this->p )
    ? this->p->to_str()
    : "aoi-invalid";
}

//
// Is the point (x,y) in the AOI?
// (x,y) is either pixels or lon/lat; up to the caller
// to verify via the flavor.  Throws if AOI is invalid
//

bool
aoi_t
::in_aoi( double x, double y ) const
{
  if ( ! this->p )
  {
    ostringstream oss;
    oss << "AOI: in_aoi( x=" << x << "; y=" << y << ") called on invalid AOI";
    throw aoi_exception( oss.str() );
  }

  return this->p->in_aoi( x, y );
}

//
// How to set an AOI
//

string
aoi_t
::help_text()
{
  return string( "AOIs are polygons in either pixel or geographic space.\n"
                 "Generally, they are specified as lists of points in the format\n"
                 "'x1,y1 : x2,y2 : x3,y3 : ...'  If only two points are provided,\n"
                 "a bounding box is created.  If more than two points are provided,\n"
                 "the convex hull of the point list is used.\n"
                 "\n"
                 "Pixels vs. geo-coords:\n"
                 "\n"
                 "As a special case, a pixel bounding box may be specified as a geometry\n"
                 "string WxH+x+y, e.g. 240x191+1500+1200, all integers.  Negative crop\n"
                 "strings are also allowed, e.g. 250x250-1000-1000 is a box from (-750,-750)\n"
                 "to (-1000,-1000).\n"
                 "\n"
                 "Otherwise, a list of points is assumed to be longitude-latitude pairs\n"
                 "unless the point string starts with 'p'.\n"
                 "\n"
                 "Longitude-latitude pairs are converted to MGRS.  It is an error if all points\n"
                 "in the AOI are not in the same zone.\n"
                 "\n"
                 "\nExamples\n"
                 "\n"
                 "Pixel point string: 'p19,22:30,30:40.5,20'\n"
                 "Geo-coord string:   '-73.8,42.15:-72.9,41.9' (creates a bounding box)\n"
                 "Geo-coord string:   '-73.8,42.15:-72.9,41.9:-73.8,42:,-71.3,41'\n"
    );
}

} // ...aoi_utils
} // ...track_oracle
} // ...kwiver
