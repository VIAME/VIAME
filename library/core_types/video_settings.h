// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// Declaration of base video settings type.

#ifndef VITAL_VIDEO_SETTINGS_H_
#define VITAL_VIDEO_SETTINGS_H_

#include <viame/core_types/vital_types_export.h>

#include <memory>

#include <cstdint>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
/// Base class for holding information about how to encode a video.
class VITAL_TYPES_EXPORT video_settings
{
public:
  virtual ~video_settings();

  /// Return the width of each video frame in pixels.
  virtual size_t width() const = 0;

  /// Return the height of each video frame in pixels.
  virtual size_t height() const = 0;

  /// Return the frame rate of the video, or -1.0 to indicate no value.
  virtual double frame_rate() const = 0;
};

using video_settings_sptr = std::shared_ptr< video_settings >;

// ----------------------------------------------------------------------------
class VITAL_TYPES_EXPORT simple_video_settings : public video_settings
{
public:
  simple_video_settings(
    size_t width, size_t height, double frame_rate = -1.0 );

  size_t width() const override;
  size_t height() const override;
  double frame_rate() const override;

private:
  size_t m_width;
  size_t m_height;
  double m_frame_rate;
};

using simple_video_settings_sptr = std::shared_ptr< simple_video_settings >;

} // namespace vital

} // namespace kwiver

#endif
