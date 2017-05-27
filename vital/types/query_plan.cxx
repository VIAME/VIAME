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
 * \brief This file contains the implementation of a geo polygon.
 */

#include "query_plan.h"

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------------------
query_plan::
query_plan()
  : m_id{ nullptr, 0 },
    m_type{ SIMILARITY },
    m_threshold{ 0.0 }
{ }

// ----------------------------------------------------------------------------
uid query_plan
::id() const
{
  return m_id;
}

// ----------------------------------------------------------------------------
void query_plan
::set_id( uid const& id )
{
  m_id = id;
}

// ----------------------------------------------------------------------------
query_plan::query_type query_plan
::type() const
{
  return m_type;
}

// ----------------------------------------------------------------------------
void query_plan
::set_type( query_type type )
{
  m_type = type;
}

// ----------------------------------------------------------------------------
filter query_plan
::temporal_filter() const
{
  return m_temporal_filter;
}

// ----------------------------------------------------------------------------
void query_plan
::set_temporal_filter( filter f )
{
  m_temporal_filter = f;
}

// ----------------------------------------------------------------------------
timestamp query_plan
::temporal_lower_bound() const
{
  return m_temporal_lower;
}

// ----------------------------------------------------------------------------
timestamp query_plan
::temporal_upper_bound() const
{
  return m_temporal_upper;
}

// ----------------------------------------------------------------------------
void query_plan
::set_temporal_bounds( timestamp const& lower, timestamp const& upper )
{
  m_temporal_lower = lower;
  m_temporal_upper = upper;
}

// ----------------------------------------------------------------------------
filter query_plan
::spatial_filter() const
{
  return m_spatial_filter;
}

// ----------------------------------------------------------------------------
void query_plan
::set_spatial_filter( filter f )
{
  m_spatial_filter = f;
}

// ----------------------------------------------------------------------------
geo_polygon query_plan
::spatial_region() const
{
  return m_spatial_region;
}

// ----------------------------------------------------------------------------
void query_plan
::set_spatial_region( geo_polygon const& r )
{
  m_spatial_region = r;
}

// ----------------------------------------------------------------------------
std::string query_plan
::stream_filter() const
{
  return m_stream_filter;
}

// ----------------------------------------------------------------------------
void query_plan
::set_stream_filter( std::string const& f )
{
  m_stream_filter = f;
}

// ----------------------------------------------------------------------------
track_descriptor_set_sptr query_plan
::descriptors() const
{
  return m_descriptors;
}

// ----------------------------------------------------------------------------
void query_plan
::set_descriptors( track_descriptor_set_sptr d )
{
  m_descriptors = d;
}

// ----------------------------------------------------------------------------
double query_plan
::threshold() const
{
  return m_threshold;
}

// ----------------------------------------------------------------------------
void query_plan
::set_threshold( double threshold )
{
  m_threshold = threshold;
}

} } // end namespace
