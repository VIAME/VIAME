// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief video_input algorithm definition instantiation

#include <viame/algorithm_framework/algo/video_input.h>

namespace kwiver {

namespace vital {

namespace algo {

// ----------------------------------------------------------------------------
algorithm_capabilities::capability_name_t const video_input::HAS_EOV(
  "has-eov" );
algorithm_capabilities::capability_name_t const video_input::HAS_FRAME_NUMBERS(
  "has-frame-numbers" );
algorithm_capabilities::capability_name_t const video_input::HAS_FRAME_TIME(
  "has-frame-time" );
algorithm_capabilities::capability_name_t const video_input::HAS_FRAME_DATA(
  "has-frame-data" );
algorithm_capabilities::capability_name_t const video_input::HAS_FRAME_RATE(
  "has-frame-rate" );
algorithm_capabilities::capability_name_t const video_input::
HAS_ABSOLUTE_FRAME_TIME( "has-abs-frame-time" );
algorithm_capabilities::capability_name_t const video_input::HAS_METADATA(
  "has-metadata" );
algorithm_capabilities::capability_name_t const video_input::HAS_TIMEOUT(
  "has-timeout" );
algorithm_capabilities::capability_name_t const
video_input::IS_SEEKABLE_BY_FRAME( "is-seekable-by-frame" );
algorithm_capabilities::capability_name_t const
video_input::IS_SEEKABLE_BY_TIME( "is-seekable-by-time" );

algorithm_capabilities::capability_name_t const video_input::HAS_RAW_IMAGE(
  "has-raw-image" );
algorithm_capabilities::capability_name_t const video_input::HAS_RAW_METADATA(
  "has-raw-metadata" );
algorithm_capabilities::capability_name_t const video_input::
HAS_UNINTERPRETED_DATA( "has-uninterpreted-data" );

// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
video_input
::video_input()
{
  attach_logger( "algo.video_input" );
}

// ----------------------------------------------------------------------------
double
video_input
::frame_rate()
{
  return -1.0;
}

// ----------------------------------------------------------------------------
kwiver::vital::path_t
video_input
::filename() const
{
  return "";
}

// ----------------------------------------------------------------------------
video_raw_image_sptr
video_input
::raw_frame_image()
{
  return nullptr;
}

// ----------------------------------------------------------------------------
video_raw_metadata_sptr
video_input
::raw_frame_metadata()
{
  return nullptr;
}

// ----------------------------------------------------------------------------
video_uninterpreted_data_sptr
video_input
::uninterpreted_frame_data()
{
  return nullptr;
}

// ----------------------------------------------------------------------------
video_settings_sptr
video_input
::implementation_settings() const
{
  return nullptr;
}

// ----------------------------------------------------------------------------
algorithm_capabilities const&
video_input
::get_implementation_capabilities() const
{
  return m_capabilities;
}

// ----------------------------------------------------------------------------
void
video_input
::set_capability(
  algorithm_capabilities::capability_name_t const& name,
  bool val )
{
  m_capabilities.set_capability( name, val );
}

} // namespace algo

} // namespace vital

} // namespace kwiver
