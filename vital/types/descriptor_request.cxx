// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
