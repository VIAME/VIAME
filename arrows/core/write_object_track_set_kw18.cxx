/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * \brief Implementation of detected object set csv output
 */

#include "write_object_track_set_kw18.h"

#include <time.h>

#include <vital/vital_foreach.h>

namespace kwiver {
namespace arrows {
namespace core {

/// This format should only be used for tracks.
///
/// \li Column(s) 1: Track-id
/// \li Column(s) 2: Track-length (# of detections)
/// \li Column(s) 3: Frame-number (-1 if not available)
/// \li Column(s) 4-5: Tracking-plane-loc(x,y) (Could be same as World-loc)
/// \li Column(s) 6-7: Velocity(x,y)
/// \li Column(s) 8-9: Image-loc(x,y)
/// \li Column(s) 10-13: Img-bbox(TL_x,TL_y,BR_x,BR_y)
/// \li Column(s) 14: Area (0 - when not available)
/// \li Column(s) 15-17: World-loc(x,y,z) (long, lat, 0 - when not available)
/// \li Column(s) 18: Timesetamp(-1 if not available)
/// \li Column(s) 19: Track-confidence(-1_when_not_available)

// -------------------------------------------------------------------------------
class write_object_track_set_kw18::priv
{
public:
  priv( write_object_track_set_kw18* parent)
    : m_parent( parent )
    , m_logger( kwiver::vital::get_logger( "write_object_track_set_kw18" ) )
    , m_first( true )
    , m_frame_number( 1 )
    , m_delim( "," )
  { }

  ~priv() { }

  write_object_track_set_kw18* m_parent;
  kwiver::vital::logger_handle_t m_logger;
  bool m_first;
  int m_frame_number;
  std::string m_delim;
  std::map< unsigned, vital::track_sptr > m_tracks;
};


// ===============================================================================
write_object_track_set_kw18
::write_object_track_set_kw18()
  : d( new write_object_track_set_kw18::priv( this ) )
{
}


write_object_track_set_kw18
::~write_object_track_set_kw18()
{
  VITAL_FOREACH( auto trk_pair, d->m_tracks )
  {
    auto trk_ptr = trk_pair.second;

    VITAL_FOREACH( auto ts_ptr, *trk_ptr )
    {
      vital::object_track_state* ts =
        dynamic_cast< vital::object_track_state* >( ts_ptr.get() );

      if( !ts )
      {
        LOG_ERROR( d->m_logger, "MISSED STATE " << trk_ptr->id() << " " << trk_ptr->size() );
        continue;
      }

      vital::detected_object_sptr det = ts->detection;
      const vital::bounding_box_d empty_box = vital::bounding_box_d( -1, -1, -1, -1 );
      vital::bounding_box_d bbox = ( det ? det->bounding_box() : empty_box );

      stream() << trk_ptr->id() << " "     // 1: track id
               << trk_ptr->size() << " "   // 2: track length
               << ts->frame() << " "       // 3: frame number
               << "0 "                     // 4: tracking plane x
               << "0 "                     // 5: tracking plane y
               << "0 "                     // 6: velocity x
               << "0 "                     // 7: velocity y
               << bbox.center()[0] << " "  // 8: image location x
               << bbox.center()[1] << " "  // 9: image location y
               << bbox.min_x() << " "      // 10: TL-x
               << bbox.min_y() << " "      // 11: TL-y
               << bbox.max_x() << " "      // 12: BR-x
               << bbox.max_y() << " "      // 13: BR-y
               << bbox.area() << " "       // 14: area
               << "0 "                     // 15: world-loc x
               << "0 "                     // 16: world-loc y
               << "0 "                     // 17: world-loc z
               << ts->frame() << " "       // 18: timestamp
               << det->confidence()        // 19: confidence
               << std::endl;
    }
  }
}


// -------------------------------------------------------------------------------
void
write_object_track_set_kw18
::set_configuration(vital::config_block_sptr config)
{
  d->m_delim = config->get_value<std::string>( "delimiter", d->m_delim );
}


// -------------------------------------------------------------------------------
bool
write_object_track_set_kw18
::check_configuration(vital::config_block_sptr config) const
{
  return true;
}


// -------------------------------------------------------------------------------
void
write_object_track_set_kw18
::write_set( const kwiver::vital::object_track_set_sptr set )
{
  if( d->m_first )
  {
    // Write file header(s)
    stream() << "# 1:Track-id "
             << "2:Track-length "
             << "3:Frame-number "
             << "4:Tracking-plane-loc(x) "
             << "5:Tracking-plane-loc(y) "
             << "6:velocity(x) "
             << "7:velocity(y) "

             << "8:Image-loc(x) "
             << "9:Image-loc(y) "
             << "10:Img-bbox(TL_x) "
             << "11:Img-bbox(TL_y) "
             << "12:Img-bbox(BR_x) "
             << "13:Img-bbox(BR_y) "
             << "14:Area "

             << "15:World-loc(x) "
             << "16:World-loc(y) "
             << "17:World-loc(z) "
             << "18:timestamp "
             << "19:track-confidence"
             << std::endl;

    d->m_first = false;
  }

  VITAL_FOREACH( auto trk, set->tracks() )
  {
    d->m_tracks[ trk->id() ] = trk;
  }
}

} } } // end namespace
