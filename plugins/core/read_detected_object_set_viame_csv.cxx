/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Implementation for read_detected_object_set_viame_csv
 */

#include "read_detected_object_set_viame_csv.h"

#include "convert_notes_to_attributes.h"
#include "convert_polygons_to_mask.h"

#include <vital/util/tokenize.h>
#include <vital/util/data_stream_reader.h>
#include <vital/types/image.h>
#include <vital/types/image_container.h>
#include <vital/exceptions.h>

#include <kwiversys/SystemTools.hxx>

#include <map>
#include <memory>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include <algorithm>


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
  priv( read_detected_object_set_viame_csv& parent )
    : m_parent( &parent )
    , m_first( true )
    , m_current_idx( 0 )
    , m_last_idx( 0 )
    , m_error_writer()
  { }

  ~priv() { }

  void read_all();

  read_detected_object_set_viame_csv* m_parent;
  bool m_first;

  int m_current_idx;
  int m_last_idx;

  // Optional error writer
  std::unique_ptr< std::ofstream > m_error_writer;

  // Map of detected objects indexed by frame number. Each set
  // contains all detections for a single frame.
  std::map< int, kwiver::vital::detected_object_set_sptr > m_detection_by_id;

  // Map of detected objects indexed by frame name. Each set
  // contains all detections for a single frame.
  std::map< std::string, kwiver::vital::detected_object_set_sptr > m_detection_by_str;

  // Alternative basepaths for strings as the above frame name might ref a full path.
  std::map< std::string, std::string > m_alt_filenames;

  // A list of all input filename strings used for error checking.
  std::vector< std::string > m_searched_filenames;
};


// ===================================================================================
read_detected_object_set_viame_csv
::read_detected_object_set_viame_csv()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d );
  attach_logger( "viame.core.read_detected_object_set_viame_csv" );
}


read_detected_object_set_viame_csv
::~read_detected_object_set_viame_csv()
{
  if( d->m_error_writer )
  {
    for( auto itr : d->m_detection_by_str )
    {
      if( std::find( d->m_searched_filenames.begin(),
                     d->m_searched_filenames.end(),
                     itr.first ) == d->m_searched_filenames.end() )
      {
        *d->m_error_writer << "Image not found: " << itr.first << std::endl;
      }
    }

    d->m_error_writer->close();
  }
}


// -----------------------------------------------------------------------------------
void
read_detected_object_set_viame_csv
::initialize()
{
  if( !c_warning_file.empty() )
  {
    d->m_error_writer.reset( new std::ofstream( c_warning_file.c_str(), std::ios::app ) );
  }

#ifndef VIAME_ENABLE_VXL
  if( c_poly_to_mask )
  {
    throw std::runtime_error( "Must have VXL turned on to use poly_to_mask" );
  }
#endif
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
      // backup case, an alternative specification of the filename exists
      auto alt_itr = d->m_alt_filenames.find( image_name );

      if( alt_itr != d->m_alt_filenames.end() &&
          d->m_detection_by_str.find( alt_itr->second ) != d->m_detection_by_str.end() )
      {
        // Return detections for this frame.
        set = d->m_detection_by_str[ alt_itr->second ];
      }
      else
      {
        // return empty set
        set = std::make_shared< kwiver::vital::detected_object_set>();

        if( d->m_error_writer )
        {
          *d->m_error_writer << "No annotations for file: " << image_name << std::endl;
        }
      }
    }
    else
    {
      // Return detections for this frame.
      set = d->m_detection_by_str[ image_name ];
    }

    if( d->m_error_writer )
    {
      d->m_searched_filenames.push_back( image_name );
    }
    return true;
  }

  // Test for end of all loaded detections
  if( image_name.empty() && d->m_current_idx > d->m_last_idx )
  {
    set = std::make_shared< kwiver::vital::detected_object_set>();
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
    set = d->m_detection_by_id[ d->m_current_idx ];
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

      // if this name contains a path, populate synonyms
      std::string tmp = str_id;
      while( tmp.find( '/' ) != std::string::npos ||
             tmp.find( '\\' ) != std::string::npos )
      {
        tmp = tmp.substr( tmp.find_first_of( "/\\" ) + 1 );
        if( !tmp.empty() )
        {
          m_alt_filenames[ tmp ] = str_id;
        }
      }
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

    if( m_parent->c_confidence_override > 0.0 )
    {
      conf = m_parent->c_confidence_override;
    }

    // Create detection
    kwiver::vital::detected_object_sptr dob;

    kwiver::vital::detected_object_type_sptr dot =
      std::make_shared< kwiver::vital::detected_object_type >();

    bool found_optional_field = false;

    for( unsigned i = COL_TOT; i < col.size(); i+=2 )
    {
      if( col[i].empty() || col[i][0] == '(' )
      {
        found_optional_field = true;
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

      if( m_parent->c_confidence_override > 0.0 )
      {
        spec_conf = m_parent->c_confidence_override;
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

    std::vector< std::string > poly_strings;

    if( found_optional_field )
    {
      for( unsigned i = COL_TOT; i < col.size(); i++ )
      {
        if( ( col[i].size() >= 6 && col[i].substr( 0, 6 ) == "(poly)" ) ||
            ( col[i].size() >= 7 && col[i].substr( 0, 7 ) == "(+poly)" ) )
        {
          poly_strings.push_back( col[i] );
        }
      }
    }

    std::vector< std::string > poly_string_vertices;
    std::vector< double > poly_floats;

    if( !poly_strings.empty() )
    {
      // Only use the first polygon
      kwiver::vital::tokenize( poly_strings[0], poly_string_vertices, " ", true );
      for( size_t i = 1; i < poly_string_vertices.size(); ++i )
      {
        poly_floats.push_back( std::stof( poly_string_vertices[ i ] ) );
      }
      dob->set_flattened_polygon( poly_floats );
    }

#ifdef VIAME_ENABLE_VXL
    if( m_parent->c_poly_to_mask && found_optional_field )
    {
      kwiver::vital::image_of< uint8_t > mask_data;

      convert_polys_to_mask( poly_strings, bbox, mask_data );

      kwiver::vital::image_container_scptr computed_mask =
        std::make_shared< kwiver::vital::simple_image_container >( mask_data );

      dob->set_mask( computed_mask );
    }
#endif

    if( found_optional_field )
    {
      add_attributes_to_detection( *dob, col );
    }

    // Add detection to set for the frame
    m_detection_by_id[ frame_id ]->add( dob );

    if( !str_id.empty() )
    {
      m_detection_by_str[ str_id ]->add( dob );
    }
  } // ...while !eof

  // Check if all frame names are timestamps, if so don't use them in favor of frame ids
  unsigned timestamp_count = 0;
  unsigned frame_count = 0;

  for( auto itr : m_detection_by_str )
  {
    const std::string& entry = itr.first;

    if( ( ( std::count( entry.begin(), entry.end(), ':' ) == 2 ||
            std::count( entry.begin(), entry.end(), ':' ) == 1 ) &&
          std::count( entry.begin(), entry.end(), '.' ) == 1 ) ||
         entry.find( ".data@" ) != std::string::npos )
    {
      timestamp_count++;
    }
    else
    {
      frame_count++;
    }
  }

  if( timestamp_count > 0 && frame_count <= 1 && timestamp_count > frame_count )
  {
    m_detection_by_str.clear();
  }
} // read_all

} // end namespace
