// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// video_output algorithm definition instantiation.

#include <viame/algorithm_framework/algo/video_output.h>

namespace kwiver {

namespace vital {

namespace algo {

// ----------------------------------------------------------------------------
const algorithm_capabilities::capability_name_t
video_output::SUPPORTS_FRAME_RATE( "supports-frame-rate" );

// ----------------------------------------------------------------------------
const algorithm_capabilities::capability_name_t
video_output::SUPPORTS_FRAME_TIME( "supports-frame-time" );

// ----------------------------------------------------------------------------
const algorithm_capabilities::capability_name_t
video_output::SUPPORTS_METADATA( "supports-metadata" );

// ----------------------------------------------------------------------------
const algorithm_capabilities::capability_name_t
video_output::SUPPORTS_UNINTERPRETED_DATA( "supports-uninterpreted-data" );

// ----------------------------------------------------------------------------
video_output
::video_output()
{
  attach_logger( "algo.video_output" );
}

// ----------------------------------------------------------------------------
void
video_output
::add_image( [[maybe_unused]] video_raw_image const& image )
{
  throw std::logic_error{
          "video_output: This implementation does not support raw image data" };
}

// ----------------------------------------------------------------------------
void
video_output
::add_metadata( [[maybe_unused]] video_raw_metadata const& md )
{
  throw std::logic_error{
          "video_output: This implementation does not support raw metadata" };
}

// ----------------------------------------------------------------------------
void
video_output
::add_uninterpreted_data(
  [[maybe_unused]] video_uninterpreted_data const& misc_data )
{
  throw std::logic_error{
          "video_output: This implementation does not support uninterpreted data" };
}

// ----------------------------------------------------------------------------
vital::video_settings_sptr
video_output
::implementation_settings() const
{
  return nullptr;
}

// ----------------------------------------------------------------------------
algorithm_capabilities const&
video_output
::get_implementation_capabilities() const
{
  return m_capabilities;
}

// ----------------------------------------------------------------------------
void
video_output
::set_capability(
  algorithm_capabilities::capability_name_t const& name,
  bool val )
{
  m_capabilities.set_capability( name, val );
}

} // namespace algo

} // namespace vital

} // namespace kwiver

/// \endcond
