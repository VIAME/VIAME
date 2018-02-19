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
 * \brief This file contains the implementation of a descriptor request.
 */

#include "descriptor_request.h"

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------------------
descriptor_request
::descriptor_request()
{
}

// ----------------------------------------------------------------------------
uid
descriptor_request
::id() const
{
  return m_id;
}

// ----------------------------------------------------------------------------
void
descriptor_request
::set_id( uid const& id )
{
  m_id = id;
}

// ----------------------------------------------------------------------------
timestamp
descriptor_request
::temporal_lower_bound() const
{
  return m_temporal_lower;
}

// ----------------------------------------------------------------------------
timestamp
descriptor_request
::temporal_upper_bound() const
{
  return m_temporal_upper;
}

// ----------------------------------------------------------------------------
void
descriptor_request
::set_temporal_bounds( timestamp const& lower, timestamp const& upper )
{
  m_temporal_lower = lower;
  m_temporal_upper = upper;
}

// ----------------------------------------------------------------------------
std::vector< bounding_box_i >
descriptor_request
::spatial_regions() const
{
  return m_spatial_regions;
}

// ----------------------------------------------------------------------------
void
descriptor_request
::set_spatial_regions( std::vector< bounding_box_i > const& r )
{
  m_spatial_regions = r;
}

// ----------------------------------------------------------------------------
std::string
descriptor_request
::data_location() const
{
  return m_data_location;
}

// ----------------------------------------------------------------------------
void
descriptor_request
::set_data_location( std::string const& l )
{
  m_data_location = l;
}

// ----------------------------------------------------------------------------
std::vector< image_container_sptr>
descriptor_request
::image_data() const
{
  return m_image_data;
}

// ----------------------------------------------------------------------------
void
descriptor_request
::set_image_data( std::vector< image_container_sptr> const& i )
{
  m_image_data = i;
}

} } // end namespace
