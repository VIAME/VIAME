/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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
