// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation for image exceptions
 */

#include "video.h"

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
video_exception
::video_exception() noexcept
{
  m_what = "Yo, Yo, we have a Vide-o exception";
}

video_exception
::~video_exception() noexcept
{
}

// ------------------------------------------------------------------
video_input_timeout_exception
::video_input_timeout_exception() noexcept
{
  m_what = "End of video exception";
}

video_input_timeout_exception
::~video_input_timeout_exception() noexcept
{
}

// ------------------------------------------------------------------
video_stream_exception
::video_stream_exception( std::string const& msg) noexcept
{
  m_what = "Video stream exception:" + msg;
}

video_stream_exception
::~video_stream_exception() noexcept
{
}

// ------------------------------------------------------------------
video_config_exception
::video_config_exception( std::string const& msg) noexcept
{
  m_what = "Video config exception:" + msg;
}

video_config_exception
::~video_config_exception() noexcept
{
}

// ------------------------------------------------------------------
video_runtime_exception
::video_runtime_exception( std::string const& msg) noexcept
{
  m_what = "Video runtime exception: " + msg;
}

video_runtime_exception
::~video_runtime_exception() noexcept
{
}

} } // end namespace
