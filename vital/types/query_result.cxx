// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief This file contains the implementation of a query result.
 */

#include "query_result.h"

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------------------
query_result
::query_result()
{
}

// ----------------------------------------------------------------------------
uid
query_result
::query_id() const
{
  return m_query_id;
}

// ----------------------------------------------------------------------------
void
query_result
::set_query_id( uid const& id )
{
  m_query_id = id;
}

// ----------------------------------------------------------------------------
std::string
query_result
::stream_id() const
{
  return m_stream_id;
}

// ----------------------------------------------------------------------------
void
query_result
::set_stream_id( std::string const& id )
{
  m_stream_id = id;
}

// ----------------------------------------------------------------------------
unsigned
query_result
::instance_id() const
{
  return m_instance_id;
}

// ----------------------------------------------------------------------------
void
query_result
::set_instance_id( unsigned id )
{
  m_instance_id = id;
}

// ----------------------------------------------------------------------------
double
query_result
::relevancy_score() const
{
  return m_relevancy_score;
}

// ----------------------------------------------------------------------------
void
query_result
::set_relevancy_score( double s )
{
  m_relevancy_score = s;
}

// ----------------------------------------------------------------------------
timestamp
query_result
::start_time() const
{
  return m_start_time;
}

// ----------------------------------------------------------------------------
timestamp
query_result
::end_time() const
{
  return m_end_time;
}

// ----------------------------------------------------------------------------
void
query_result
::set_temporal_bounds( timestamp const& lower, timestamp const& upper )
{
  m_start_time = lower;
  m_end_time = upper;
}

// ----------------------------------------------------------------------------
vital::geo_point
query_result
::location() const
{
  return m_location;
}

// ----------------------------------------------------------------------------
void
query_result
::set_location( vital::geo_point l )
{
  m_location = l;
}

// ----------------------------------------------------------------------------
vital::object_track_set_sptr
query_result
::tracks() const
{
  return m_tracks;
}

// ----------------------------------------------------------------------------
void
query_result
::set_tracks( vital::object_track_set_sptr t )
{
  m_tracks = t;
}

// ----------------------------------------------------------------------------
vital::track_descriptor_set_sptr
query_result
::descriptors() const
{
  return m_descriptors;
}

// ----------------------------------------------------------------------------
void
query_result
::set_descriptors( vital::track_descriptor_set_sptr d )
{
  m_descriptors = d;
}

// ----------------------------------------------------------------------------
std::vector< image_container_sptr >
query_result
::image_data() const
{
  return m_image_data;
}

// ----------------------------------------------------------------------------
void
query_result
::set_image_data( std::vector< image_container_sptr > const& i )
{
  m_image_data = i;
}

} } // end namespace
