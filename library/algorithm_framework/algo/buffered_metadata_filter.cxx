// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// Definition of buffered metadata filter algorithm.

#include <viame/algorithm_framework/algo/buffered_metadata_filter.h>

namespace kwiver {

namespace vital {

namespace algo {

const algorithm_capabilities::capability_name_t
buffered_metadata_filter::CAN_USE_FRAME_IMAGE( "can-use-frame-image" );

// ----------------------------------------------------------------------------
buffered_metadata_filter
::buffered_metadata_filter()
{
  attach_logger( "algo.buffered_metadata_filter" );
}

// ----------------------------------------------------------------------------
algorithm_capabilities const&
buffered_metadata_filter
::get_implementation_capabilities() const
{
  return m_capabilities;
}

// ----------------------------------------------------------------------------
void
buffered_metadata_filter
::set_capability(
  algorithm_capabilities::capability_name_t const& name, bool value )
{
  m_capabilities.set_capability( name, value );
}

} // namespace algo

} // namespace vital

} // namespace kwiver
