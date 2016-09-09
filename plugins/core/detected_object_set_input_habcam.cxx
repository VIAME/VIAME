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

#include "detected_object_set_input_habcam.h"

#include <vital/util/tokenize.h>
#include <vital/util/data_stream_reader.h>
#include <vital/logger/logger.h>
#include <vital/exceptions.h>

#include <map>
#include <sstream>
#include <memory>
#include <cmath>
#include <cstdlib>

namespace viame {

/// Expected format
/// typical input looks like the following:
///
/// 201503.20150517.074921957.9593.png 469 201501
/// 201503.20150517.074921957.9593.png 527 201501 boundingBox 458.6666666666667 970.4166666666666 521.3333333333334 1021.0833333333334
///
/// 1: image name
/// 2: species code
/// 3: date?
/// - the following fields are optional
/// 4: annotation type
/// 5: annotation data depends on type

// ------------------------------------------------------------------
class detected_object_set_input_habcam::priv
{
public:
  priv( detected_object_set_input_habcam* parent)
    : m_parent( parent )
    , m_logger( kwiver::vital::get_logger( "detected_object_set_input_habcam" ) )
    , m_first( true )
    , m_delim( " " )
    , m_point_dilation( 50 )
  {
    init_species_map();
  }

  ~priv() { }

  bool get_input();
  void add_detection();
  void init_species_map();
  std::string decode_species( int code );


  // -- initialized data --
  detected_object_set_input_habcam* m_parent;
  kwiver::vital::logger_handle_t m_logger;
  bool m_first;

  // -- config data --
  std::string m_delim;
  double m_point_dilation;      // in pixels

  std::map< int, std::string > m_species_map;

  // -- algo state data --
  std::shared_ptr< kwiver::vital::data_stream_reader > m_stream_reader;
  std::vector< std::string > m_input_buffer;
  kwiver::vital::detected_object_set_sptr m_current_set;
  std::string m_image_name;
};


// ==================================================================
detected_object_set_input_habcam::
detected_object_set_input_habcam()
  : d( new detected_object_set_input_habcam::priv( this ) )
{
}


detected_object_set_input_habcam::
detected_object_set_input_habcam( detected_object_set_input_habcam const& other)
  : d(new priv(*other.d))
{
}

detected_object_set_input_habcam::
~detected_object_set_input_habcam()
{
}


// ------------------------------------------------------------------
void
detected_object_set_input_habcam::
set_configuration( kwiver::vital::config_block_sptr config )
{
  d->m_delim = config->get_value<std::string>( "delimiter", d->m_delim );
  d->m_point_dilation = config->get_value<double>( "point_dilation", d->m_point_dilation );

  // Test for no specification which can happen due to config parsing issues.
  if ( d->m_delim.empty() )
  {
    d->m_delim = " ";
  }
}


// ------------------------------------------------------------------
bool
detected_object_set_input_habcam::
check_configuration( kwiver::vital::config_block_sptr config ) const
{
  return true;
}


// ------------------------------------------------------------------
bool
detected_object_set_input_habcam::
read_set( kwiver::vital::detected_object_set_sptr & set, std::string& image_name )
{
  if ( d->m_first )
  {
    d->m_first = false;

    if ( ! d->get_input() )
    {
      return false; // indicate end of file.
    }
  } // end first

  // test for end of stream
  if (this->at_eof())
  {
    return false;
  }

  // Allocate return set and reset image name
  d->m_current_set = std::make_shared<kwiver::vital::detected_object_set>();
  d->m_image_name = d->m_input_buffer[0];

  bool valid_line( true );

  while( true )
  {
    // check buffer to see if it has the current frame number
    if ( valid_line && ( d->m_input_buffer[0] == d->m_image_name ) )
    {
      // We are in the same frame, so add this detection to current set
      d->add_detection();

      // Get next input line
      valid_line = d->get_input();
    }
    else
    {
      // Don't return empty sets at this point.
      if (d->m_current_set->size() > 0)
      {
        image_name = d->m_image_name;
        set = d->m_current_set;
        return true;
      }

      d->m_image_name = d->m_input_buffer[0];
    }
  } // end while

  return false;
}


// ------------------------------------------------------------------
void
detected_object_set_input_habcam::
new_stream()
{
  d->m_first = true;
  d->m_stream_reader = std::make_shared< kwiver::vital::data_stream_reader>( stream() );
}


// ==================================================================
bool
detected_object_set_input_habcam::priv::
get_input()
{
  std::string line;
  if ( ! m_stream_reader->getline( line ) )
  {
    return false; // end of file.
  }

  m_input_buffer.clear();
  kwiver::vital::tokenize( line, m_input_buffer, m_delim, true );

  // Test the minimum number of fields.
  if ( m_input_buffer.size() < 3 )
  {
    std::stringstream str;
    str << "Too few field in input at line " << m_stream_reader->line_number() << std::endl
        << "\"" << line << "\"";
    throw kwiver::vital::invalid_data( str.str() );
  }

  return true;
}


// ------------------------------------------------------------------
void
detected_object_set_input_habcam::priv::
add_detection()
{
  if ( m_input_buffer.size() < 4 )
  {
    // This is an image level annotation.
    // Not handled at this point
    return;
  }

  kwiver::vital::detected_object_type_sptr dot = std::make_shared< kwiver::vital::detected_object_type > ();
  std::string class_name = decode_species( atoi( m_input_buffer[1].c_str() ) );

  dot->set_score( class_name, 1.0 );

  kwiver::vital::bounding_box_d bbox( 0, 0, 0, 0 );

  // Generate bbox based on annotation type
  if ( "boundingBox" == m_input_buffer[3] )
  {
    if ( m_input_buffer.size() == 8 )
    {
      bbox = kwiver::vital::bounding_box_d(
        atof( m_input_buffer[4].c_str() ),
        atof( m_input_buffer[5].c_str() ),
        atof( m_input_buffer[6].c_str() ),
        atof( m_input_buffer[7].c_str() ) );
    }
    else
    {
      // invalid line format
      LOG_WARN( m_logger, "Invalid line format for boundingBox annotation" );
      return;
    }
  }
  else if ( "line" == m_input_buffer[3] )
  {
    if ( m_input_buffer.size() == 8 )
    {
      const double x1 = atof( m_input_buffer[4].c_str() );
      const double y1 = atof( m_input_buffer[5].c_str() );
      const double x2 = atof( m_input_buffer[6].c_str() );
      const double y2 = atof( m_input_buffer[7].c_str() );

      const double cx = ( x1 + x2 ) / 2;
      const double cy = ( y1 + y2 ) / 2;

      const double dx = x1 - cx;
      const double dy = y1 - cy;
      const double r = sqrt( ( dx * dx ) + ( dy * dy ) );

      bbox = kwiver::vital::bounding_box_d(
        cx - r, cy - r,
        cx + r, cy + r );
    }
    else
    {
      // invalid line format
      LOG_WARN( m_logger, "Invalid line format for boundingBox annotation" );
      return;
    }

  }
  else if ( "point" == m_input_buffer[3] )
  {
    if ( m_input_buffer.size() == 6 )
    {
      const double cx = atof( m_input_buffer[4].c_str() );
      const double cy = atof( m_input_buffer[5].c_str() );

      bbox = kwiver::vital::bounding_box_d(
        cx - m_point_dilation, cy - m_point_dilation,
        cx + m_point_dilation, cy + m_point_dilation );
    }
    else
    {
      // invalid line format
      LOG_WARN( m_logger, "Invalid line format for point annotation" );
      return;
    }
  }
  else
  {
    // Unknown annotation type
    LOG_WARN( m_logger, "Unknown annotation type \"" << m_input_buffer[3] << "\"" );
    return;
  }

  m_current_set->add( std::make_shared< kwiver::vital::detected_object > ( bbox, 1.0, dot ) );
} // detected_object_set_input_habcam::priv::add_detection


// ------------------------------------------------------------------
std::string
detected_object_set_input_habcam::priv::
decode_species( int code )
{
  std::stringstream str;

  auto ix = m_species_map.find( code );
  if (ix != m_species_map.end() )
  {
    str << ix->second;
  }

  str << "(" << code << ")";
  return  str.str();
}


// ------------------------------------------------------------------
void
detected_object_set_input_habcam::priv::
init_species_map()
{
  // Could read the species definition from a file.
  // The map will be constant and large, so this could be class static.
  // Probably not more than one reader instantiated at a time.
  m_species_map[185] = "Live Scallop";
  m_species_map[197] = "Live Scallop";
  m_species_map[207] = "Live Scallop";
  m_species_map[211] = "Live Scallop";
  m_species_map[515] = "Live Scallop";
  m_species_map[912] = "Live Scallop";
  m_species_map[919] = "Live Scallop";
  m_species_map[920] = "Live Scallop";
  m_species_map[188] = "Dead Scallop";
  m_species_map[403] = "Sand-eel";
  m_species_map[524] = "Skate";
  m_species_map[533] = "Fish";
  m_species_map[1003] = "Fish";
  m_species_map[1001] = "Fish";
  m_species_map[158] = "Crab";
  m_species_map[258] = "Crab";
}

} // end namespace
