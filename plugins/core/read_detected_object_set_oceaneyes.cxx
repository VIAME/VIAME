/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Implementation for read_detected_object_set_oceaneyes
 */

#include "read_detected_object_set_oceaneyes.h"

#include <vital/util/tokenize.h>
#include <vital/util/data_stream_reader.h>
#include <vital/exceptions.h>

#include <kwiversys/SystemTools.hxx>

#include <map>
#include <sstream>
#include <cstdlib>

namespace viame
{

double filter_number( std::string str )
{
  str.erase( std::remove( str.begin(), str.end(), '('), str.end() );
  str.erase( std::remove( str.begin(), str.end(), ')'), str.end() );
  str.erase( std::remove( str.begin(), str.end(), '"'), str.end() );
  str.erase( std::remove( str.begin(), str.end(), ' '), str.end() );

  double output;

  try
  {
    output = std::stod( str );
  }
  catch( const std::invalid_argument& e )
  {
    std::cout << "Unable to convert string to number: " << str << std::endl;
    throw e;
  }

  return output;
}

// -----------------------------------------------------------------------------------
class read_detected_object_set_oceaneyes::priv
{
public:
  priv( read_detected_object_set_oceaneyes& parent )
    : m_parent( &parent )
    , m_first( true )
  {}

  ~priv() { }

  void read_all();

  read_detected_object_set_oceaneyes* m_parent;
  bool m_first;

  typedef std::map< std::string, kwiver::vital::detected_object_set_sptr > map_type;

  // Map of detected objects indexed by file name. Each set contains all detections
  // for a single frame (unsorted).
  map_type m_detection_by_str;

  map_type::iterator m_current_idx;
};


// ===================================================================================
read_detected_object_set_oceaneyes
::read_detected_object_set_oceaneyes()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d );
  attach_logger( "viame.core.read_detected_object_set_oceaneyes" );
}


read_detected_object_set_oceaneyes
::~read_detected_object_set_oceaneyes()
{
}


// -----------------------------------------------------------------------------------
bool
read_detected_object_set_oceaneyes
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  return true;
}


// -----------------------------------------------------------------------------------
bool
read_detected_object_set_oceaneyes
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
read_detected_object_set_oceaneyes
::new_stream()
{
  d->m_first = true;
}


// ===================================================================================
void
read_detected_object_set_oceaneyes::priv
::read_all()
{
  std::string line;
  kwiver::vital::data_stream_reader stream_reader( m_parent->stream() );

  // Read detections
  m_detection_by_str.clear();

  // Determine version based on header type
  unsigned version = 1;

  while( stream_reader.getline( line ) )
  {
    std::vector< std::string > col;
    kwiver::vital::tokenize( line, col, ",", false );

    if( col.empty() || ( !col[0].empty() && col[0][0] == '#' ) )
    {
      continue;
    }

    if( col[0] == "filename"  )
    {
      if( line.find( "\"photo location\"" ) != std::string::npos )
      {
        version = 2;
      }
      continue;
    }
    else if( col[0] == "photo location" )
    {
      version = 3;
      continue;
    }

    // Object ID
    const unsigned  COL_FRAME_ID = ( version == 3 ? 1 : ( version == 2 ? 0 : 0 ) );
    // Species Label
    const unsigned  COL_SPECIES_ID = ( version == 3 ? 5 : ( version == 2 ? 4 : 4 ) );
    // Fish confidence
    const int COL_FISH_CONF = ( version == 3 ? -1 : ( version == 2 ? -1 : 6 ) );
    // Species confidence
    const int COL_SPEC_CONF = ( version == 3 ? 8 : ( version == 2 ? -1 : 7 ) );
    // Is head tail valid
    const int COL_IS_HEAD_TAIL = ( version == 3 ? 11 : ( version == 2 ? -1 : 10 ) );
    // Head tail locations
    const int COL_HEAD_TAIL = ( version == 3 ? 12 : ( version == 2 ? 5 : 11 ) );

    if( col.size() < COL_SPECIES_ID )
    {
      std::stringstream str;
      str << "This is not a oceaneyes file; found " << col.size()
          << " columns in\n\"" << line << "\"";
      throw kwiver::vital::invalid_data( str.str() );
    }

    // Get frame ID and remove extension to make filetype agnostic
    std::string str_id = col[ COL_FRAME_ID ];
    str_id = str_id.substr( 0, str_id.find_last_of( "." ) );

    if( !str_id.empty() &&
        m_detection_by_str.count( str_id ) == 0 )
    {
      // create a new detection set entry
      m_detection_by_str[ str_id ] =
        std::make_shared<kwiver::vital::detected_object_set>();
    }

    if( COL_SPECIES_ID > 0 && col[ COL_SPECIES_ID ] == m_parent->c_no_fish_string )
    {
      continue;
    }

    double x1 = filter_number( col[ COL_HEAD_TAIL + 0 ] );
    double y1 = filter_number( col[ COL_HEAD_TAIL + 1 ] );
    double x2 = filter_number( col[ COL_HEAD_TAIL + 2 ] );
    double y2 = filter_number( col[ COL_HEAD_TAIL + 3 ] );

    bool is_valid_head_tail =
      ( COL_IS_HEAD_TAIL < 0 || col[ COL_IS_HEAD_TAIL ] == "yes" );

    double x_min = std::min( x1, x2 );
    double y_min = std::min( y1, y2 );
    double x_max = std::max( x1, x2 );
    double y_max = std::max( y1, y2 );

    double c_x = ( x_min + x_max ) / 2.0;
    double c_y = ( y_min + y_max ) / 2.0;

    double width = ( x_max - x_min ) * ( 1.0 + m_parent->c_box_expansion );
    double height = ( y_max - y_min ) * ( 1.0 + m_parent->c_box_expansion );

    if( width == 0 || height == 0 )
    {
      continue;
    }

    if( height / width > m_parent->c_max_aspect_ratio )
    {
      width = height / m_parent->c_max_aspect_ratio;
    }
    if( width / height > m_parent->c_max_aspect_ratio )
    {
      height = width / m_parent->c_max_aspect_ratio;
    }

    kwiver::vital::bounding_box_d bbox(
      c_x - width / 2.0,
      c_y - height / 2.0,
      c_x + width / 2.0,
      c_y + height / 2.0 );

    // Create detection
    kwiver::vital::detected_object_type_sptr dot =
      std::make_shared< kwiver::vital::detected_object_type >();

    std::string species_label = col[ COL_SPECIES_ID ];

    double species_conf = 1.0;

    if( COL_SPEC_CONF > 0 && COL_FISH_CONF > 0 )
    {
      species_conf = std::max( std::stod( col[ COL_SPEC_CONF ] ),
                               std::stod( col[ COL_FISH_CONF ] ) );
    }
    else if( COL_SPEC_CONF > 0 )
    {
      species_conf = std::stod( col[ COL_SPEC_CONF ] );
    }

    species_label = ( species_label.empty() ? "other" : species_label );
    species_conf = ( species_conf == 0.0 ? 0.10 : species_conf );

    dot->set_score( species_label, species_conf );

    kwiver::vital::detected_object_sptr dob =
      std::make_shared< kwiver::vital::detected_object>(
        bbox, species_conf, dot );

    if( is_valid_head_tail )
    {
      dob->add_keypoint( "head",
        kwiver::vital::point_2d( x1, y1 ) );
      dob->add_keypoint( "tail",
        kwiver::vital::point_2d( x2, y2 ) );
    }

    // Add detection to set for the frame
    m_detection_by_str[ str_id ]->add( dob );

  } // ...while !eof
} // read_all

} // end namespace
