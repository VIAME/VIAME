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

#include "detected_object_set_output_kw18.h"
#include <vital/vital_foreach.h>

#include <time.h>

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
/// \li Column(s) 10-13: Img-bbox(TL_x,TL_y,BR_x,BR_y) (location of top-left & bottom-right vertices)
/// \li Column(s) 14: Area
/// \li Column(s) 15-17: World-loc(x,y,z) (longitude, latitude, 0 - when available)
/// \li Column(s) 18: Timesetamp(-1 if not available)
/// \li Column(s) 19: Track-confidence(-1_when_not_available)

// ------------------------------------------------------------------
class detected_object_set_output_kw18::priv
{
public:
  priv( detected_object_set_output_kw18* parent)
    : m_parent( parent )
    , m_logger( kwiver::vital::get_logger( "detected_object_set_output_kw18" ) )
    , m_first( true )
    , m_frame_number( 1 )
  { }

  ~priv() { }

  void read_all();

  detected_object_set_output_kw18* m_parent;
  kwiver::vital::logger_handle_t m_logger;
  bool m_first;
  int m_frame_number;
};


// ==================================================================
detected_object_set_output_kw18::
detected_object_set_output_kw18()
  : d( new detected_object_set_output_kw18::priv( this ) )
{
}


detected_object_set_output_kw18::
~detected_object_set_output_kw18()
{
}


// ------------------------------------------------------------------
void
detected_object_set_output_kw18::
set_configuration(vital::config_block_sptr config)
{ }


// ------------------------------------------------------------------
bool
detected_object_set_output_kw18::
check_configuration(vital::config_block_sptr config) const
{
  return true;
}


// ------------------------------------------------------------------
void
detected_object_set_output_kw18::
write_set( const kwiver::vital::detected_object_set_sptr set, std::string const& image_name )
{

  if (d->m_first)
  {
    time_t rawtime;
    struct tm * timeinfo;

    time ( &rawtime );
    timeinfo = localtime ( &rawtime );
    char* cp =  asctime( timeinfo );
    cp[ strlen( cp )-1 ] = 0; // remove trailing newline
    const std::string atime( cp );

    // Write file header(s)
    stream() << "# 1:Track-id [-1_for_detections] "
             << "2:Track-length [-1_for_detections] "
             << "3:Frame-number  [-1_for_detections] "
             << "4:Tracking-plane-loc(x) [-1_for_detections] "
             << "5:Tracking-plane-loc(y) [-1_for_detections] "
             << "6:velocity(x) [-1_for_detections] "
             << "7:velocity(y) [-1_for_detections] "

             << "8:Image-loc(x)"
             << " 9:Image-loc(y)"
             << " 10:Img-bbox(TL_x)"
             << " 11:Img-bbox(TL_y)"
             << " 12:Img-bbox(BR_x)"
             << " 13:Img-bbox(BR_y)"
             << " 14:Area"

             << " 15:World-loc(x)"
             << " 16:World-loc(y)"
             << " 17:World-loc(z)"
             << " 18:timestamp"
             << " 19:track-confidence"
             << std::endl

      // Provide some provenience to the file. Could have a config
      // parameter that is copied to the file as a configurable
      // comment or marker.

             << "# Written on: " << atime
             << "   by: detected_object_set_output_kw18"
             << std::endl;

    d->m_first = false;
  } // end first

  // Get detections from set
  const auto detections = set->select();
  VITAL_FOREACH( const auto det, detections )
  {
    const kwiver::vital::bounding_box_d bbox( det->bounding_box() );
    double ilx = ( bbox.min_x() + bbox.max_x() ) / 2.0;
    double ily = ( bbox.min_y() + bbox.max_y() ) / 2.0;

    stream() << "-1 "               // 1: track id
             << "-1 "               // 2: track length
             << d->m_frame_number   // 3: frame number / set number
             << " -1 "              // 4: tracking plane x
             << " -1 "              // 5: tracking plane y
             << "-1 "               // 6: velocity x
             << "-1 "               // 7: velocity y
             << ilx << " "          // 8: image location x
             << ily << " "          // 9: image location y
             << bbox.min_x() << " " // 10: TL-x
             << bbox.min_y() << " " // 11: TL-y
             << bbox.max_x() << " " // 12: BR-x
             << bbox.max_y() << " " // 13: BR-y
             << bbox.area() << " "  // 14: area
             << "-1 "               // 15: world-loc x
             << "-1 "               // 16: world-loc y
             << "-1 "               // 17: world-loc z
             << "0 "                // 18 timestamp
             << det->confidence()   // 19: confidence
             << std::endl;
  } // end foreach

  // Put each set on a new frame
  ++d->m_frame_number;
}

} } } // end namespace
