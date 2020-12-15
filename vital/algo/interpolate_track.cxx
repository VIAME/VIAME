// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
