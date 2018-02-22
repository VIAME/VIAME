/*ckwg +29
 * Copyright 2015-2016 by Kitware, Inc.
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
 * \brief video_input algorithm definition instantiation
 */

#include <vital/algo/video_input.h>
#include <vital/algo/algorithm.txx>

namespace kwiver {
namespace vital {
namespace algo {

// ------------------------------------------------------------------
const algorithm_capabilities::capability_name_t video_input::HAS_EOV( "has-eov" );
const algorithm_capabilities::capability_name_t video_input::HAS_FRAME_NUMBERS( "has-frame-numbers" );
const algorithm_capabilities::capability_name_t video_input::HAS_FRAME_TIME( "has-frame-time" );
const algorithm_capabilities::capability_name_t video_input::HAS_FRAME_DATA( "has-frame-data" );
const algorithm_capabilities::capability_name_t video_input::HAS_FRAME_RATE( "has-frame-rate" );
const algorithm_capabilities::capability_name_t video_input::HAS_ABSOLUTE_FRAME_TIME( "has-abs-frame-time" );
const algorithm_capabilities::capability_name_t video_input::HAS_METADATA( "has-metadata" );
const algorithm_capabilities::capability_name_t video_input::HAS_TIMEOUT( "has-timeout" );


// ------------------------------------------------------------------
video_input
::video_input()
{
  attach_logger( "video_input" );
}


video_input
::~video_input()
{
}


// ------------------------------------------------------------------
double
video_input
::frame_rate()
{
  return -1.0;
}


// ------------------------------------------------------------------
algorithm_capabilities const&
video_input
::get_implementation_capabilities() const
{
  return m_capabilities;
}


// ------------------------------------------------------------------
void
video_input
::set_capability( algorithm_capabilities::capability_name_t const& name, bool val )
{
  m_capabilities.set_capability( name, val );
}


} } } // end namespace

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::video_input);
/// \endcond
