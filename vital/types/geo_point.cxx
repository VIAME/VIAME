// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief This file contains the implementation of a geo point.
 */

#include "geo_point.h"
#include "geodesy.h"

#include <iomanip>
#include <limits>
#include <stdexcept>

namespace kwiver {
namespace vital {

using geo_3d_point_t = geo_point::geo_3d_point_t;

// ----------------------------------------------------------------------------
geo_point::
geo_point()
  : m_original_crs{ -1 }
{ }

// ----------------------------------------------------------------------------
geo_point::
geo_point( geo_2d_point_t const& point, int crs )
  : m_original_crs( crs )
{
  m_loc.insert( std::make_pair( crs, geo_3d_point_t{ point[0],
                                                      point[1],
                                                      0 } ));
}

// ----------------------------------------------------------------------------
geo_point::
geo_point( geo_3d_point_t const& point, int crs )
  : m_original_crs( crs )
{
  m_loc.insert( std::make_pair( crs, point ) );
}

// ----------------------------------------------------------------------------
bool geo_point
::is_empty() const
{
  return m_loc.empty();
}

// ----------------------------------------------------------------------------
geo_3d_point_t geo_point
::location() const
{
  return m_loc.at( m_original_crs );
}

// ----------------------------------------------------------------------------
int geo_point
::crs() const
{
  return m_original_crs;
}

// ----------------------------------------------------------------------------
geo_3d_point_t geo_point
::location( int crs ) const
{
  auto const i = m_loc.find( crs );
  if ( i == m_loc.end() )
  {
    auto const p = geo_conv( location(), m_original_crs, crs );
    m_loc.emplace( crs, p );
    return p;
  }

  return i->second;
}

// ----------------------------------------------------------------------------
void geo_point
::set_location( geo_2d_point_t const& loc, int crs )
{
  m_original_crs = crs;
  m_loc.clear();
  m_loc.insert( std::make_pair( crs, geo_3d_point_t { loc[0], loc[1], 0 } ) );
}

// ----------------------------------------------------------------------------
void geo_point
::set_location( geo_3d_point_t const& loc, int crs )
{
  m_original_crs = crs;
  m_loc.clear();
  m_loc.insert( std::make_pair( crs, loc ) );
}

// ----------------------------------------------------------------------------
std::ostream&
operator<<( std::ostream& str, vital::geo_point const& obj )
{
  str << "geo_point\n";
  if ( obj.is_empty() )
  {
    str << "[ empty ]";
  }
  else
  {
    auto const old_prec = str.precision();
    auto const loc = obj.location();

    str << std::setprecision( std::numeric_limits<double>::digits10 + 2 )
        << "[ " << loc[0]
        << ", " << loc[1]
        << ", " << loc[2]
        << " ] @ " << obj.crs();

    str.precision( old_prec );
  }

  return str;
}

} } // end namespace
