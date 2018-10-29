/*ckwg +29
 * Copyright 2017-2018 by Kitware, Inc.
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

#include "interpolate_track.h"

#include <vital/algo/algorithm.txx>

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::interpolate_track);
/// \endcond

namespace kwiver {
namespace vital {
namespace algo {

// ----------------------------------------------------------------------------
interpolate_track::
interpolate_track()
  : m_progress_callback( nullptr )
{
  attach_logger( "algo.interpolate_track" );
}

// ----------------------------------------------------------------------------
void
interpolate_track::
set_video_input( video_input_sptr input )
{
  m_video_input = input;
}

// ----------------------------------------------------------------------------
void
interpolate_track::
set_progress_callback( progress_callback_t cb )
{
  m_progress_callback = cb;
}

// ----------------------------------------------------------------------------
void
interpolate_track::
do_callback( float progress )
{
  static constexpr auto total = 1 << 25;

  do_callback( static_cast<int>( std::ldexp( progress, 25 ) ), total );
}

// ----------------------------------------------------------------------------
void
interpolate_track::
do_callback( int progress, int total )
{
  if ( m_progress_callback != nullptr )
  {
    m_progress_callback( progress, total );
  }
}

} } } // end namespace
