/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
