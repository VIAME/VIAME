// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief This file contains the implementation of a database query.
 */

#include "database_query.h"
#include <stdexcept>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------------------
database_query
::database_query()
  : m_id{ nullptr, 0 },
    m_type{ SIMILARITY },
    m_threshold{ 0.0 }
{ }

// ----------------------------------------------------------------------------
vital::uid
database_query
::id() const
{
  return m_id;
}

// ----------------------------------------------------------------------------
void
database_query
::set_id( vital::uid const& id )
{
  m_id = id;
}

// ----------------------------------------------------------------------------
database_query::query_type
database_query
::type() const
{
  return m_type;
}

// ----------------------------------------------------------------------------
void
database_query
::set_type( query_type type )
{
  m_type = type;
}

// ----------------------------------------------------------------------------
query_filter
database_query
::temporal_filter() const
{
  return m_temporal_filter;
}

// ----------------------------------------------------------------------------
void
database_query
::set_temporal_filter( query_filter f )
{
  m_temporal_filter = f;
}

// ----------------------------------------------------------------------------
timestamp
database_query
::temporal_lower_bound() const
{
  return m_temporal_lower;
}

// ----------------------------------------------------------------------------
timestamp
database_query
::temporal_upper_bound() const
{
  return m_temporal_upper;
}

// ----------------------------------------------------------------------------
void
database_query
::set_temporal_bounds( timestamp const& lower, timestamp const& upper )
{
  if (upper < lower)
  {
    throw std::logic_error("upper temporal bound less than lower temporal bound");
  }
  m_temporal_lower = lower;
  m_temporal_upper = upper;
}

// ----------------------------------------------------------------------------
query_filter
database_query
::spatial_filter() const
{
  return m_spatial_filter;
}

// ----------------------------------------------------------------------------
void
database_query
::set_spatial_filter( query_filter f )
{
  m_spatial_filter = f;
}

// ----------------------------------------------------------------------------
geo_polygon
database_query
::spatial_region() const
{
  return m_spatial_region;
}

// ----------------------------------------------------------------------------
void
database_query
::set_spatial_region( geo_polygon const& r )
{
  m_spatial_region = r;
}

// ----------------------------------------------------------------------------
std::string
database_query
::stream_filter() const
{
  return m_stream_filter;
}

// ----------------------------------------------------------------------------
void
database_query
::set_stream_filter( std::string const& f )
{
  m_stream_filter = f;
}

// ----------------------------------------------------------------------------
track_descriptor_set_sptr
database_query
::descriptors() const
{
  return m_descriptors;
}

// ----------------------------------------------------------------------------
void
database_query
::set_descriptors( track_descriptor_set_sptr d )
{
  m_descriptors = d;
}

// ----------------------------------------------------------------------------
double
database_query
::threshold() const
{
  return m_threshold;
}

// ----------------------------------------------------------------------------
void
database_query
::set_threshold( double threshold )
{
  m_threshold = threshold;
}

} } // end namespace
