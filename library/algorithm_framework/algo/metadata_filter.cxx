// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief metadata_filter algorithm instantiation

#include <viame/algorithm_framework/algo/metadata_filter.h>

namespace kwiver {

namespace vital {

namespace algo {

const algorithm_capabilities::capability_name_t
metadata_filter::CAN_USE_FRAME_IMAGE( "can-use-frame-image" );

// ----------------------------------------------------------------------------
metadata_filter
::metadata_filter()
{
  attach_logger( "algo.metadata_filter" ); // specify a logger
}

// ----------------------------------------------------------------------------
algorithm_capabilities const&
metadata_filter
::get_implementation_capabilities() const
{
  return m_capabilities;
}

// ----------------------------------------------------------------------------
void
metadata_filter
::set_capability(
  algorithm_capabilities::capability_name_t const& name, bool value )
{
  m_capabilities.set_capability( name, value );
}

} // namespace algo

} // namespace vital

} // namespace kwiver
