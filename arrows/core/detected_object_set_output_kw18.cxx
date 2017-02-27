/*ckwg +29
 * Copyright 2016-2017 by Kitware, Inc.
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

#include <fstream>
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
    , m_tot_writer( NULL )
  { }

  ~priv() { }

  void read_all();

  detected_object_set_output_kw18* m_parent;
  kwiver::vital::logger_handle_t m_logger;
  bool m_first;
  int m_frame_number;
  std::ofstream* m_tot_writer;
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
  if( d->m_tot_writer && d->m_tot_writer->is_open() )
  {
    d->m_tot_writer->close();
  }
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
    stream() << "# 1:Track-id "
             << "2:Track-length "
             << "3:Frame-number "
             << "4:Tracking-plane-loc(x) "
             << "5:Tracking-plane-loc(y) "
             << "6:velocity(x) "
             << "7:velocity(y) "

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
    d->m_tot_writer = new std::ofstream( filename() + ".txt" );
    
  } // end first

  // Get detections from set
  const auto detections = set->select();
  VITAL_FOREACH( const auto det, detections )
  {
    const kwiver::vital::bounding_box_d bbox( det->bounding_box() );
    double ilx = ( bbox.min_x() + bbox.max_x() ) / 2.0;
    double ily = ( bbox.min_y() + bbox.max_y() ) / 2.0;

    static int counter = 0;
    const int id = counter++;

    stream() << id                  // 1: track id
             << " 1 "               // 2: track length
             << d->m_frame_number-1 // 3: frame number / set number
             << " 0 "               // 4: tracking plane x
             << " 0 "               // 5: tracking plane y
             << "0 "                // 6: velocity x
             << "0 "                // 7: velocity y
             << ilx << " "          // 8: image location x
             << ily << " "          // 9: image location y
             << bbox.min_x() << " " // 10: TL-x
             << bbox.min_y() << " " // 11: TL-y
             << bbox.max_x() << " " // 12: BR-x
             << bbox.max_y() << " " // 13: BR-y
             << bbox.area() << " "  // 14: area
             << "0 "                // 15: world-loc x
             << "0 "                // 16: world-loc y
             << "0 "                // 17: world-loc z
             << "0 "                // 18: timestamp
             << det->confidence()   // 19: confidence
             << std::endl;

    vital::detected_object_type_sptr clf = det->type();

    double f = 0.0, s = 0.0, o = 0.0;

    if( clf->has_class_name( "scallop" ) )
    {
      s = clf->score( "scallop" );
    }
    if( clf->has_class_name( "LIVE_SCALLOP" ) )
    {
      s = clf->score( "LIVE_SCALLOP" );
    }
    if( clf->has_class_name( "fish" ) )
    {
      f = clf->score( "fish" );
    }
    if( clf->has_class_name( "background" ) )
    {
      f = clf->score( "background" );
    }
    else
    {
      o = 1.0 - f - s;
    }

    (*d->m_tot_writer) << id        // 1: track id
                       << " " << f  // 2: fish prob
                       << " " << s  // 3: scallop prob
                       << " " << o  // 4: other prob
                       << std::endl;
  } // end foreach

  // Put each set on a new frame
  ++d->m_frame_number;
}

} } } // end namespace
