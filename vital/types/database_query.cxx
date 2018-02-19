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
 * \brief This file contains the implementation of a database query.
 */

#include "database_query.h"

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------------------
database_query::
database_query()
  : m_id{ nullptr, 0 },
    m_type{ SIMILARITY },
    m_threshold{ 0.0 }
{ }

// ----------------------------------------------------------------------------
uid database_query
::id() const
{
  return m_id;
}

// ----------------------------------------------------------------------------
void database_query
::set_id( uid const& id )
{
  m_id = id;
}

// ----------------------------------------------------------------------------
database_query::query_type database_query
::type() const
{
  return m_type;
}

// ----------------------------------------------------------------------------
void database_query
::set_type( query_type type )
{
  m_type = type;
}

// ----------------------------------------------------------------------------
query_filter database_query
::temporal_filter() const
{
  return m_temporal_filter;
}

// ----------------------------------------------------------------------------
void database_query
::set_temporal_filter( query_filter f )
{
  m_temporal_filter = f;
}

// ----------------------------------------------------------------------------
timestamp database_query
::temporal_lower_bound() const
{
  return m_temporal_lower;
}

// ----------------------------------------------------------------------------
timestamp database_query
::temporal_upper_bound() const
{
  return m_temporal_upper;
}

// ----------------------------------------------------------------------------
void database_query
::set_temporal_bounds( timestamp const& lower, timestamp const& upper )
{
  m_temporal_lower = lower;
  m_temporal_upper = upper;
}

// ----------------------------------------------------------------------------
query_filter database_query
::spatial_filter() const
{
  return m_spatial_filter;
}

// ----------------------------------------------------------------------------
void database_query
::set_spatial_filter( query_filter f )
{
  m_spatial_filter = f;
}

// ----------------------------------------------------------------------------
geo_polygon database_query
::spatial_region() const
{
  return m_spatial_region;
}

// ----------------------------------------------------------------------------
void database_query
::set_spatial_region( geo_polygon const& r )
{
  m_spatial_region = r;
}

// ----------------------------------------------------------------------------
std::string database_query
::stream_filter() const
{
  return m_stream_filter;
}

// ----------------------------------------------------------------------------
void database_query
::set_stream_filter( std::string const& f )
{
  m_stream_filter = f;
}

// ----------------------------------------------------------------------------
track_descriptor_set_sptr database_query
::descriptors() const
{
  return m_descriptors;
}

// ----------------------------------------------------------------------------
void database_query
::set_descriptors( track_descriptor_set_sptr d )
{
  m_descriptors = d;
}

// ----------------------------------------------------------------------------
double database_query
::threshold() const
{
  return m_threshold;
}

// ----------------------------------------------------------------------------
void database_query
::set_threshold( double threshold )
{
  m_threshold = threshold;
}

} } // end namespace
