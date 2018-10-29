/*ckwg +29
 * Copyright 2016-2018 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
