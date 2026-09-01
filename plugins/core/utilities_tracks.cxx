/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "utilities_tracks.h"

#include <vital/types/object_track_set.h>
#include <vital/types/timestamp.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace viame {

namespace
{

struct track_entry
{
  kv::frame_id_t frame_id;
  kv::detected_object_sptr detection;
};

// Frame-position tolerance for treating an output frame as exactly aligned
// with an annotated input frame
constexpr double frame_align_eps = 1e-3;

kv::timestamp
make_timestamp( kv::frame_id_t frame, double rate )
{
  kv::time_usec_t time_usec = static_cast< kv::time_usec_t >(
    std::llround( static_cast< double >( frame ) * 1e6 / rate ) );

  return kv::timestamp( time_usec, frame );
}

} // end anonymous namespace


// -----------------------------------------------------------------------------
kv::detected_object_sptr
interpolate_detection( const kv::detected_object_sptr& d1,
                       const kv::detected_object_sptr& d2,
                       double alpha )
{
  const kv::detected_object_sptr& nearer = ( alpha < 0.5 ? d1 : d2 );

  kv::detected_object_sptr result = nearer->clone();

  const kv::bounding_box_d& b1 = d1->bounding_box();
  const kv::bounding_box_d& b2 = d2->bounding_box();

  result->set_bounding_box( kv::bounding_box_d(
    b1.min_x() * ( 1.0 - alpha ) + b2.min_x() * alpha,
    b1.min_y() * ( 1.0 - alpha ) + b2.min_y() * alpha,
    b1.max_x() * ( 1.0 - alpha ) + b2.max_x() * alpha,
    b1.max_y() * ( 1.0 - alpha ) + b2.max_y() * alpha ) );

  result->set_confidence(
    d1->confidence() * ( 1.0 - alpha ) + d2->confidence() * alpha );

  return result;
}


// -----------------------------------------------------------------------------
kv::object_track_set_sptr
resample_object_tracks( const kv::object_track_set_sptr& tracks,
                        double input_rate,
                        double output_rate,
                        bool interpolate_states,
                        unsigned max_interp_gap )
{
  std::vector< kv::track_sptr > output_tracks;

  if( !tracks || input_rate <= 0.0 || output_rate <= 0.0 )
  {
    return std::make_shared< kv::object_track_set >( output_tracks );
  }

  // Output frames per input frame
  const double ratio = output_rate / input_rate;

  for( const auto& trk : tracks->tracks() )
  {
    if( !trk )
    {
      continue;
    }

    std::vector< track_entry > entries;

    for( const auto& state_sptr : *trk )
    {
      auto ots = std::dynamic_pointer_cast< kv::object_track_state >( state_sptr );

      if( ots && ots->detection() )
      {
        entries.push_back( { ots->frame(), ots->detection() } );
      }
    }

    if( entries.empty() )
    {
      continue;
    }

    std::sort( entries.begin(), entries.end(),
      []( const track_entry& a, const track_entry& b )
      {
        return a.frame_id < b.frame_id;
      } );

    kv::track_sptr out_trk = kv::track::create();
    out_trk->set_id( trk->id() );

    kv::frame_id_t out_first = static_cast< kv::frame_id_t >(
      std::ceil( entries.front().frame_id * ratio - frame_align_eps ) );
    kv::frame_id_t out_last = static_cast< kv::frame_id_t >(
      std::floor( entries.back().frame_id * ratio + frame_align_eps ) );

    if( out_first > out_last )
    {
      // Whole track lies between two output frames; snap to the nearest one
      // so short tracks survive downsampling
      kv::frame_id_t snap = static_cast< kv::frame_id_t >(
        std::llround( entries.front().frame_id * ratio ) );

      out_trk->append( std::make_shared< kv::object_track_state >(
        make_timestamp( snap, output_rate ), entries.front().detection ) );

      output_tracks.push_back( out_trk );
      continue;
    }

    for( kv::frame_id_t g = out_first; g <= out_last; ++g )
    {
      // Position of this output frame in input frame coordinates
      const double p = static_cast< double >( g ) / ratio;

      auto next = std::lower_bound( entries.begin(), entries.end(),
        p - frame_align_eps,
        []( const track_entry& e, double f )
        {
          return static_cast< double >( e.frame_id ) < f;
        } );

      kv::detected_object_sptr det;

      if( next != entries.end() &&
          std::fabs( static_cast< double >( next->frame_id ) - p ) <=
            frame_align_eps )
      {
        det = next->detection;
      }
      else if( next != entries.begin() && next != entries.end() )
      {
        auto prev = std::prev( next );

        const double gap =
          static_cast< double >( next->frame_id - prev->frame_id );

        if( max_interp_gap > 0 && gap > max_interp_gap )
        {
          continue;
        }

        if( interpolate_states )
        {
          const double alpha = ( p - prev->frame_id ) / gap;
          det = interpolate_detection( prev->detection, next->detection, alpha );
        }
        else
        {
          det = ( p - prev->frame_id <= next->frame_id - p )
                  ? prev->detection : next->detection;
        }
      }

      if( det )
      {
        out_trk->append( std::make_shared< kv::object_track_state >(
          make_timestamp( g, output_rate ), det ) );
      }
    }

    if( !out_trk->empty() )
    {
      output_tracks.push_back( out_trk );
    }
  }

  return std::make_shared< kv::object_track_set >( output_tracks );
}

} // end namespace viame
