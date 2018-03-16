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

#include "interpolate_track_spline.h"

#include <vital/types/object_track_set.h>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace core {

/// Private implementation class
class interpolate_track_spline::priv
{
public:
};

namespace {

// ----------------------------------------------------------------------------
time_us_t
lerp( time_us_t a, time_us_t b, double k )
{
  auto const x = static_cast< double >( a );
  auto const y = static_cast< double >( b );
  return static_cast< time_us_t >( ( ( 1.0 - k ) * x ) + ( k * y ) );
}

// ----------------------------------------------------------------------------
vector_2d
lerp( vector_2d const& a, vector_2d const& b, double k )
{
  return ( ( 1.0 - k ) * a ) + ( k * b );
}

// ----------------------------------------------------------------------------
bounding_box_d
lerp( bounding_box_d const& a, bounding_box_d const& b, double k )
{
  auto const& ul = lerp( a.upper_left(), b.upper_left(), k );
  auto const& lr = lerp( a.lower_right(), b.lower_right(), k );
  return { ul, lr };
}

} // end anonymous namespace

// ----------------------------------------------------------------------------
interpolate_track_spline
::interpolate_track_spline()
: d_(new priv)
{
}

// ----------------------------------------------------------------------------
interpolate_track_spline
::~interpolate_track_spline()
{
}

// ----------------------------------------------------------------------------
void
interpolate_track_spline
::set_configuration( vital::config_block_sptr /*in_config*/ )
{
}

// ----------------------------------------------------------------------------
bool
interpolate_track_spline
::check_configuration( vital::config_block_sptr /*config*/ ) const
{
  return true;
}

// ----------------------------------------------------------------------------
track_sptr
interpolate_track_spline::
interpolate( track_sptr input_track )
{
  if ( !input_track ) return nullptr;

  // Extract states, for easier iteration over intervals to be filled
  std::map< frame_id_t, std::pair< time_us_t, detected_object_sptr > > states;
  for ( auto const& sp : *input_track )
  {
    auto const osp = std::dynamic_pointer_cast< object_track_state >( sp );
    if ( !osp ) continue;

    auto const detection = osp->detection;
    if ( !detection ) continue;

    states.emplace( osp->frame(), std::make_pair( osp->time(), detection ) );
  }

  // Create result track
  auto new_track = track::create( input_track->data() );
  new_track->set_id( input_track->id() );

  auto append = [&new_track]( frame_id_t frame, time_us_t time,
                              detected_object_sptr detection ){
    new_track->append(
      std::make_shared< object_track_state >( frame, time, detection ) );
  };

  // Iterate over intervals to be filled
  for ( auto i = states.begin(), n = states.begin();; i = n )
  {
    append( i->first, i->second.first, i->second.second );
    if ( ( ++n ) == states.end() ) break;

    // Get end points of interval
    auto const f0 = i->first;
    auto const f1 = n->first;
    auto const xk = 1.0 / static_cast< double >( f1 - f0 );

    auto const& p0 = i->second.second->bounding_box();
    auto const& p1 = n->second.second->bounding_box();
    auto const c0 = i->second.second->confidence();
    auto const c1 = n->second.second->confidence();

    // Iterate over frame numbers in interval
    for ( auto fn = f0 + 1; fn < f1; ++fn )
    {
      auto const x = static_cast< double >( fn - f0 ) * xk;
      auto const tn = lerp( i->second.first, n->second.first, x );

      auto const& bbox = lerp( p0, p1, x );
      auto const c = ( pow( x, 2.0 ) * c0 ) + ( pow( 1.0 - x, 2.0 ) * c1 );

      auto const d = std::make_shared< detected_object >( bbox, c );
      new_track->append(
        std::make_shared< object_track_state >( fn, tn, d ) );
    }
  }

  return new_track;
}

} } } // end namespace
