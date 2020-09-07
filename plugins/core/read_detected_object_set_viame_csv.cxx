/*ckwg +29
 * Copyright 2018-2019 by Kitware, Inc.
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
 * \brief Implementation for read_detected_object_set_viame_csv
 */

#include "read_detected_object_set_viame_csv.h"
#include "notes_to_attributes.h"

#include <vital/util/tokenize.h>
#include <vital/util/data_stream_reader.h>
#include <vital/exceptions.h>

#include <kwiversys/SystemTools.hxx>

#include <map>
#include <sstream>
#include <cstdlib>

namespace viame {

enum
{
  COL_DET_ID=0,  // 0: Object ID
  COL_SOURCE_ID, // 1
  COL_FRAME_ID,  // 2
  COL_MIN_X,     // 3
  COL_MIN_Y,     // 4
  COL_MAX_X,     // 5
  COL_MAX_Y,     // 6
  COL_CONFIDENCE,// 7
  COL_LENGTH,    // 8
  COL_TOT        // 9
};

// -----------------------------------------------------------------------------------
class read_detected_object_set_viame_csv::priv
{
public:
  priv( read_detected_object_set_viame_csv* parent )
    : m_parent( parent )
    , m_first( true )
    , m_confidence_override( -1.0 )
    , m_current_idx( 0 )
    , m_last_idx( 0 )
  { }

  ~priv() { }

  void read_all();

  read_detected_object_set_viame_csv* m_parent;
  bool m_first;
  double m_confidence_override;

  int m_current_idx;
  int m_last_idx;

  // Map of detected objects indexed by frame number. Each set
  // contains all detections for a single frame.
  std::map< int, kwiver::vital::detected_object_set_sptr > m_detection_by_id;

  // Map of detected objects indexed by frame name. Each set
  // contains all detections for a single frame.
  std::map< std::string, kwiver::vital::detected_object_set_sptr > m_detection_by_str;
};


// ===================================================================================
read_detected_object_set_viame_csv
::read_detected_object_set_viame_csv()
  : d( new read_detected_object_set_viame_csv::priv( this ) )
{
  attach_logger( "viame.core.read_detected_object_set_viame_csv" );
}


read_detected_object_set_viame_csv
::~read_detected_object_set_viame_csv()
{
}


// -----------------------------------------------------------------------------------
void
read_detected_object_set_viame_csv
::set_configuration( kwiver::vital::config_block_sptr config )
{
  d->m_confidence_override =
    config->get_value< double >( "confidence_override", d->m_confidence_override );
}


// -----------------------------------------------------------------------------------
bool
read_detected_object_set_viame_csv
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  return true;
}


// -----------------------------------------------------------------------------------
bool
read_detected_object_set_viame_csv
::read_set( kwiver::vital::detected_object_set_sptr& set, std::string& image_name )
{
  if( d->m_first )
  {
    // Read in all detections
    d->read_all();
    d->m_first = false;

    // set up iterators for returning sets.
    d->m_current_idx = 0;

    if( d->m_detection_by_id.empty() )
    {
      d->m_last_idx = 0;
    }
    else
    {
      d->m_last_idx = d->m_detection_by_id.rbegin()->first;
    }
  } // end first

  // External image name provided, use that
  if( !image_name.empty() && !d->m_detection_by_str.empty() )
  {
    // return detection set at current index if there is one
    if( d->m_detection_by_str.find( image_name ) == d->m_detection_by_str.end() )
    {
      // return empty set
      set = std::make_shared< kwiver::vital::detected_object_set>();
    }
    else
    {
      // Return detections for this frame.
      set = d->m_detection_by_str[ image_name ];
    }
    return true;
  }

  // Test for end of all loaded detections
  if( d->m_current_idx > d->m_last_idx )
  {
    return false;
  }

  // Return detection set at current index if there is one
  if( d->m_detection_by_id.count( d->m_current_idx ) == 0 )
  {
    // Return empty set
    set = std::make_shared< kwiver::vital::detected_object_set>();
  }
  else
  {
    // Return detections for this frame.
    set = d->m_detection_by_id[d->m_current_idx];
  }

  ++d->m_current_idx;

  return true;
}


// -----------------------------------------------------------------------------------
void
read_detected_object_set_viame_csv
::new_stream()
{
  d->m_first = true;
}


// ===================================================================================
void
read_detected_object_set_viame_csv::priv
::read_all()
{
  std::string line;
  kwiver::vital::data_stream_reader stream_reader( m_parent->stream() );

  // Read detections
  m_detection_by_id.clear();
  m_detection_by_str.clear();

  while( stream_reader.getline( line ) )
  {
    std::vector< std::string > col;
    kwiver::vital::tokenize( line, col, ",", false );

    if( col.empty() || ( !col[0].empty() && col[0][0] == '#' ) )
    {
      continue;
    }

    if( col.size() < 9 )
    {
      std::stringstream str;
      str << "This is not a viame_csv file; found " << col.size()
          << " columns in\n\"" << line << "\"";
      throw kwiver::vital::invalid_data( str.str() );
    }

    /*
     * Check to see if we have seen this frame before. If we have,
     * then retrieve the frame's index into our output map. If not
     * seen before, add frame -> detection set index to our map and
     * press on.
     *
     * This allows for track states to be written in a non-contiguous
     * manner as may be done by streaming writers.
     */
    int frame_id = atoi( col[COL_FRAME_ID].c_str() );
    std::string str_id = col[COL_SOURCE_ID];

    if( m_detection_by_id.count( frame_id ) == 0 )
    {
      // create a new detection set entry
      m_detection_by_id[ frame_id ] =
        std::make_shared<kwiver::vital::detected_object_set>();
    }

    if( !str_id.empty() &&
        m_detection_by_str.count( str_id ) == 0 )
    {
      // create a new detection set entry
      m_detection_by_str[ str_id ] =
        std::make_shared<kwiver::vital::detected_object_set>();
    }

    kwiver::vital::bounding_box_d bbox(
      atof( col[COL_MIN_X].c_str() ),
      atof( col[COL_MIN_Y].c_str() ),
      atof( col[COL_MAX_X].c_str() ),
      atof( col[COL_MAX_Y].c_str() ) );

    double conf = atof( col[COL_CONFIDENCE].c_str() );

    if( conf == -1.0 )
    {
      conf = 1.0;
    }

    if( m_confidence_override > 0.0 )
    {
      conf = m_confidence_override;
    }

    // Create detection
    kwiver::vital::detected_object_sptr dob;

    kwiver::vital::class_map_sptr dot =
      std::make_shared<kwiver::vital::class_map>();

    bool found_attribute = false;

    for( unsigned i = COL_TOT; i < col.size(); i+=2 )
    {
      if( col[i].empty() || col[i][0] == '(' )
      {
        found_attribute = true;
        break;
      }

      if( col.size() < i + 2 )
      {
        std::stringstream str;
        str << "Every species pair must contain a confidence; error "
            << "at\n\"" << line << "\"";
        throw kwiver::vital::invalid_data( str.str() );
      }

      std::string spec_id = col[i];

      double spec_conf = atof( col[i+1].c_str() );

      if( m_confidence_override > 0.0 )
      {
        spec_conf = m_confidence_override;
      }

      dot->set_score( spec_id, spec_conf );
    }

    if( COL_TOT < col.size() )
    {
      dob = std::make_shared< kwiver::vital::detected_object>( bbox, conf, dot );
    }
    else
    {
      dob = std::make_shared< kwiver::vital::detected_object>( bbox, conf );
    }

    if( found_attribute )
    {
      add_attributes_to_detection( *dob, col );
    }

    // Add detection to set for the frame
    m_detection_by_id[frame_id]->add( dob );

    if( !str_id.empty() )
    {
      m_detection_by_str[str_id]->add( dob );
    }
  } // ...while !eof
} // read_all

} // end namespace
