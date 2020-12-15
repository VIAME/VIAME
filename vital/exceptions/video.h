// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for image exceptions
 */

#ifndef VITAL_CORE_EXCEPTIONS_VIDEO_H
#define VITAL_CORE_EXCEPTIONS_VIDEO_H

#include <string>

#include <vital/exceptions/base.h>

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
/// Generic video exception
class VITAL_EXCEPTIONS_EXPORT video_exception
  : public vital_exception
{
public:
  /// Constructor
  video_exception() noexcept;

  /// Destructor
  virtual ~video_exception() noexcept;
};

// ------------------------------------------------------------------
/// Timeout getting next video frame.
/*
 * This exception is thrown when the video_input::next_frame() method
 * timeout expires.
 */
class VITAL_EXCEPTIONS_EXPORT video_input_timeout_exception
  : public video_exception
{
public:
  /// Constructor
  video_input_timeout_exception() noexcept;

  /// Destructor
  virtual ~video_input_timeout_exception() noexcept;
};

// ------------------------------------------------------------------
/// Video stream error.
/*
 * This exception is thrown when there is exceptional condition while
 * streaming video.
 */
class VITAL_EXCEPTIONS_EXPORT video_stream_exception
  : public video_exception
{
public:
  /// Constructor
  video_stream_exception( std::string const& msg ) noexcept;

  /// Destructor
  virtual ~video_stream_exception() noexcept;
};

// ------------------------------------------------------------------
/// Video config error.
/*
 * This exception is thrown when there is exceptional condition is
 * found in the configuration.
 */
class VITAL_EXCEPTIONS_EXPORT video_config_exception
  : public video_exception
{
public:
  /// Constructor
  video_config_exception( std::string const& msg ) noexcept;

  /// Destructor
  virtual ~video_config_exception() noexcept;
};

// ------------------------------------------------------------------
/// Video runtime error.
/*
 * This exception is thrown when there is exceptional condition while
 * processing the a video.
 */
class VITAL_EXCEPTIONS_EXPORT video_runtime_exception
  : public video_exception
{
public:
  /// Constructor
  video_runtime_exception( std::string const& msg ) noexcept;

  /// Destructor
  virtual ~video_runtime_exception() noexcept;
};

} } // end namespace

#endif /* VITAL_CORE_EXCEPTIONS_VIDEO_H */
