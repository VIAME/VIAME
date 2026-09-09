// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// Definition of base video settings type.

#include <viame/core_types/video_settings.h>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
video_settings::~video_settings()
{}

// ----------------------------------------------------------------------------
simple_video_settings
::simple_video_settings( size_t width, size_t height, double frame_rate )
  : m_width{ width },
    m_height{ height },
    m_frame_rate{ frame_rate }
{}

// ----------------------------------------------------------------------------
size_t
simple_video_settings
::height() const
{
  return m_height;
}

// ----------------------------------------------------------------------------
size_t
simple_video_settings
::width() const
{
  return m_width;
}

// ----------------------------------------------------------------------------
double
simple_video_settings
::frame_rate() const
{
  return m_frame_rate;
}

} // namespace vital

} // namespace kwiver
