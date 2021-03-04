// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief core polygon implementation
 */

#include "polygon.h"

#include <stdexcept>
#include <sstream>

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
polygon::
polygon()
{ }

polygon::
polygon( const std::vector< point_t > &dat )
  : m_polygon( dat )
{

}

polygon::
polygon( std::initializer_list< point_t > dat )
  : m_polygon( dat )
{

}

// ------------------------------------------------------------------
polygon::
~polygon()
{ }

// ------------------------------------------------------------------
void
polygon::
push_back( double x, double y )
{
  m_polygon.push_back( point_t( x, y ) );
}

// ------------------------------------------------------------------
void
polygon::
push_back( const kwiver::vital::polygon::point_t& pt )
{
  m_polygon.push_back( pt );
}

// ------------------------------------------------------------------
size_t
polygon::
num_vertices() const
{
  return m_polygon.size();
}

// ------------------------------------------------------------------
bool
polygon::
contains( double x, double y )
{
  bool c = false;

  int n = static_cast<int>(m_polygon.size());
  for (int i = 0, j = n-1; i < n; j = i++)
  {
    const point_t& p_i = m_polygon[i];
    const point_t& p_j = m_polygon[j];

    // by definition, corner points and edge points are inside the polygon:
    if ((p_j.x() - x) * (p_i.y() - y) == (p_i.x() - x) * (p_j.y() - y) &&
        (((p_i.x()<=x) && (x<=p_j.x())) || ((p_j.x()<=x) && (x<=p_i.x()))) &&
        (((p_i.y()<=y) && (y<=p_j.y())) || ((p_j.y()<=y) && (y<=p_i.y()))))
    {
      return true;
    }

    // invert c for each edge crossing:
    if ((((p_i.y()<=y) && (y<p_j.y())) || ((p_j.y()<=y) && (y<p_i.y()))) &&
        (x < (p_j.x() - p_i.x()) * (y - p_i.y()) / (p_j.y() - p_i.y()) + p_i.x()))
    {
      c = !c;
    }
  } // end for

  return c;
}

// ------------------------------------------------------------------
bool
polygon::
contains( const kwiver::vital::polygon::point_t& pt )
{
  return contains( pt[0], pt[1] );
}

// ------------------------------------------------------------------
kwiver::vital::polygon::point_t
polygon::
at( size_t idx ) const
{
  if ( idx >= m_polygon.size() )
  {
    std::stringstream str;
    str << "Requested index " << idx
        << " is beyond the end of the polygon. Last valid index is "
        << m_polygon.size()-1;
    throw std::out_of_range( str.str() );
  }

  return m_polygon[idx];
}

// ------------------------------------------------------------------
std::vector< kwiver::vital::polygon::point_t >
polygon::
get_vertices() const
{
  return m_polygon;
}

} }    // end namespace
