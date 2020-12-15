// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
const algorithm_capabilities::capability_name_t video_input::IS_SEEKABLE( "is-seekable" );

// ------------------------------------------------------------------
video_input
::video_input()
{
  attach_logger( "algo.video_input" );
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
