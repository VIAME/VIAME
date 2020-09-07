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
 * \brief Implementation for read_detected_object_set_fishnet
 */

#include "read_detected_object_set_fishnet.h"

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
  COL_FRAME_ID=0,  // 0: Object ID
  COL_DET_ID,      // 1: Detection ID
  COL_MIN_X,       // 2: Min X
  COL_MAX_X,       // 3: Max X
  COL_MIN_Y,       // 4: Min Y
  COL_MAX_Y,       // 5: Max Y
  COL_LABEL,       // 6: Label
  COL_TOT          // 7: Total Required Columns
};

// -----------------------------------------------------------------------------------
class read_detected_object_set_fishnet::priv
{
public:
  priv( read_detected_object_set_fishnet* parent )
    : m_parent( parent )
    , m_first( true )
  {}

  ~priv() { }

  void read_all();

  read_detected_object_set_fishnet* m_parent;
  bool m_first;

  typedef std::map< std::string, kwiver::vital::detected_object_set_sptr > map_type;

  // Map of detected objects indexed by frame name. Each set
  // contains all detections for a single frame.
  map_type m_detection_by_str;

  map_type::iterator m_current_idx;
};


// ===================================================================================
read_detected_object_set_fishnet
::read_detected_object_set_fishnet()
  : d( new read_detected_object_set_fishnet::priv( this ) )
{
  attach_logger( "viame.core.read_detected_object_set_fishnet" );
}


read_detected_object_set_fishnet
::~read_detected_object_set_fishnet()
{
}


// -----------------------------------------------------------------------------------
void
read_detected_object_set_fishnet
::set_configuration( kwiver::vital::config_block_sptr config )
{
}


// -----------------------------------------------------------------------------------
bool
read_detected_object_set_fishnet
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  return true;
}


// -----------------------------------------------------------------------------------
bool
read_detected_object_set_fishnet
::read_set( kwiver::vital::detected_object_set_sptr& set, std::string& image_name )
{
  if( d->m_first )
  {
    // Read in all detections
    d->read_all();
    d->m_first = false;

    // set up iterators for returning sets.
    d->m_current_idx = d->m_detection_by_str.begin();
  }

  // External image name provided, use that
  if( !image_name.empty() && !d->m_detection_by_str.empty() )
  {
    // return detection set at current index if there is one
    std::string name_no_ext = image_name.substr( 0, image_name.find_last_of( "." ) );

    if( d->m_detection_by_str.find( name_no_ext ) == d->m_detection_by_str.end() )
    {
      // return empty set
      set = std::make_shared< kwiver::vital::detected_object_set>();
    }
    else
    {
      // Return detections for this frame.
      set = d->m_detection_by_str[ name_no_ext ];
    }
    return true;
  }

  // Test for end of all loaded detections
  if( d->m_current_idx == d->m_detection_by_str.end() )
  {
    return false;
  }

  // Return detection set at current index if there is one
  set = d->m_current_idx->second;
  d->m_current_idx++;

  return true;
}


// -----------------------------------------------------------------------------------
void
read_detected_object_set_fishnet
::new_stream()
{
  d->m_first = true;
}


// ===================================================================================
void
read_detected_object_set_fishnet::priv
::read_all()
{
  std::string line;
  kwiver::vital::data_stream_reader stream_reader( m_parent->stream() );

  // Read detections
  m_detection_by_str.clear();

  while( stream_reader.getline( line ) )
  {
    std::vector< std::string > col;
    kwiver::vital::tokenize( line, col, ",", false );

    if( col.empty() || ( !col[0].empty() && col[0][0] == '#' ) )
    {
      continue;
    }

    if( col.size() < COL_TOT )
    {
      std::stringstream str;
      str << "This is not a fishnet file; found " << col.size()
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
    std::string str_id = col[ COL_FRAME_ID ];

    if( !str_id.empty() &&
        m_detection_by_str.count( str_id ) == 0 )
    {
      // create a new detection set entry
      m_detection_by_str[ str_id ] =
        std::make_shared<kwiver::vital::detected_object_set>();
    }

    kwiver::vital::bounding_box_d bbox(
      atof( col[ COL_MIN_X ].c_str() ),
      atof( col[ COL_MIN_Y ].c_str() ),
      atof( col[ COL_MAX_X ].c_str() ),
      atof( col[ COL_MAX_Y ].c_str() ) );

    // Create detection
    kwiver::vital::class_map_sptr dot =
      std::make_shared< kwiver::vital::class_map >();

    dot->set_score( col[ COL_LABEL ], 1.0 );

    kwiver::vital::detected_object_sptr dob =
      std::make_shared< kwiver::vital::detected_object>( bbox, 1.0, dot );

    // Add detection to set for the frame
    m_detection_by_str[ str_id ]->add( dob );

  } // ...while !eof
} // read_all

} // end namespace
