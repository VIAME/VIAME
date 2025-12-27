/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "read_detected_object_set_habcam.h"

#include <vital/util/tokenize.h>
#include <vital/util/data_stream_reader.h>
#include <vital/logger/logger.h>
#include <vital/exceptions.h>

#include <map>
#include <sstream>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <string>

namespace viame {

/// Expected format
/// typical input looks like the following for version 1:
///
/// 201503.20150517.074921957.9593.png 469 201501
/// 201503.20150517.074921957.9593.png 527 201501 boundingBox ...
/// ...458.6666666666667 970.4166666666666 521.3333333333334 1021.0833333333334
///
/// or alternatively for version 2:
///
/// 201503.20150525.101430169.572475.png,185,live sea scallop,"""line"": ...
/// ...[[963.3651240907222, 1011.8916075814615], [964.870387904199, 966.7336931771592]]"
///
/// or alternatively for version 3:
///
/// "201503.20150525.102214751.575250.png" 185 "boundingBox" 1195 380 1239 424
///
/// 1: image name
/// 2: species code
/// 3: date?
/// - the following fields are optional
/// 4: annotation type
/// 5: annotation data depends on type

// -----------------------------------------------------------------------------
class read_detected_object_set_habcam::priv
{
public:
  priv( read_detected_object_set_habcam* parent)
    : m_parent( parent )
    , m_first( true )
    , m_current_idx( 0 )
    , m_last_idx( 0 )
    , m_delim( "" )
    , m_point_dilation( 50 )
    , m_use_number_labels( true )
    , m_use_internal_table( false )
    , m_detected_version( 1 )
  {
    init_species_map();
  }

  ~priv() {}

  void read_all();
  void init_species_map();
  std::string decode_species( int code );
  void parse_detection( const std::vector< std::string >& parsed_line );

  bool parse_box( const std::vector< std::string >& parsed_line,
                  unsigned index,
                  kwiver::vital::bounding_box_d& bbox );

  // -- initialized data --
  read_detected_object_set_habcam* m_parent;
  bool m_first;

  int m_current_idx;
  int m_last_idx;

  // -- config data --
  std::string m_delim;
  double m_point_dilation;      // in pixels
  bool m_use_number_labels;
  bool m_use_internal_table;
  int m_detected_version;

  std::map< int, std::string > m_species_map;
  std::map< std::string, kwiver::vital::detected_object_set_sptr > m_gt_sets;
  std::vector< std::string > m_filenames;
};


// =============================================================================
read_detected_object_set_habcam
::read_detected_object_set_habcam()
  : d( new read_detected_object_set_habcam::priv( this ) )
{
}


read_detected_object_set_habcam
::~read_detected_object_set_habcam()
{
}


// -----------------------------------------------------------------------------
void
read_detected_object_set_habcam
::set_configuration( kwiver::vital::config_block_sptr config )
{
  d->m_delim =
    config->get_value<std::string>( "delimiter", d->m_delim );
  d->m_point_dilation =
    config->get_value<double>( "point_dilation", d->m_point_dilation );
  d->m_use_internal_table =
    config->get_value<bool>( "point_dilation", d->m_use_internal_table );
}


// -----------------------------------------------------------------------------
bool
read_detected_object_set_habcam
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  return true;
}


// -----------------------------------------------------------------------------
bool
read_detected_object_set_habcam
::read_set( kwiver::vital::detected_object_set_sptr& set, std::string& image_name )
{
  if( d->m_first )
  {
    d->m_first = false;
    d->init_species_map();
    d->read_all();

    d->m_current_idx = 0;
    d->m_last_idx = d->m_filenames.size();
  }

  // External image name provided, use that
  if( !image_name.empty() )
  {
    // return detection set at current index if there is one
    if( d->m_gt_sets.find( image_name ) == d->m_gt_sets.end() )
    {
      // return empty set
      set = std::make_shared< kwiver::vital::detected_object_set>();
    }
    else
    {
      // Return detections for this frame.
      set = d->m_gt_sets[ image_name ];
    }
    return true;
  }

  // External image name not provided, iterate through all images alphabetically

  // Test for end of stream
  if( d->m_current_idx >= d->m_last_idx )
  {
    return false;
  }

  // Return detections for this frame.
  image_name = d->m_filenames[ d->m_current_idx ];
  set = d->m_gt_sets[ image_name ];

  ++d->m_current_idx;
  return true;
}


// -----------------------------------------------------------------------------
void
read_detected_object_set_habcam
::new_stream()
{
  d->m_first = true;
  d->m_filenames.clear();
  d->m_gt_sets.clear();
}


// -----------------------------------------------------------------------------
void
read_detected_object_set_habcam::priv
::parse_detection( const std::vector< std::string >& parsed_line )
{
  if ( parsed_line.size() < 4 )
  {
    // This is an image level annotation.
    // Not handled at this point
    return;
  }

  if( m_gt_sets.find( parsed_line[0] ) == m_gt_sets.end() )
  {
    // create a new detection set entry
    m_gt_sets[ parsed_line[0] ] =
      std::make_shared<kwiver::vital::detected_object_set>();

    m_filenames.push_back( parsed_line[0] );
  }

  kwiver::vital::detected_object_type_sptr dot
    = std::make_shared< kwiver::vital::detected_object_type >();

  std::string class_name;

  if( !m_use_number_labels )
  {
    class_name = parsed_line[2];

    std::replace( class_name.begin(), class_name.end(), ' ', '_' );

    class_name.erase(
      std::remove( class_name.begin(), class_name.end(), '(' ),
      class_name.end() );
    class_name.erase(
      std::remove( class_name.begin(), class_name.end(), ')' ),
      class_name.end() );
  }
  else if( m_use_internal_table )
  {
    class_name = decode_species( atoi( parsed_line[1].c_str() ) );
  }
  else
  {
    class_name = parsed_line[1];
  }

  dot->set_score( class_name, 1.0 );

  kwiver::vital::bounding_box_d bbox( 0, 0, 0, 0 );

  // Generate bbox based on annotation type
  if ( !parse_box( parsed_line, 3, bbox ) && !parse_box( parsed_line, 2, bbox ) )
  {
    // Unknown annotation type
    LOG_WARN( m_parent->logger(), "Unknown annotation type \"" << parsed_line[3] << "\"" );
    return;
  }

  m_gt_sets[ parsed_line[0] ]->add(
    std::make_shared< kwiver::vital::detected_object >( bbox, 1.0, dot ) );
} // read_detected_object_set_habcam::priv::add_detection


// -----------------------------------------------------------------------------
bool
read_detected_object_set_habcam::priv
::parse_box( const std::vector< std::string >& parsed_line,
             unsigned index,
             kwiver::vital::bounding_box_d& bbox )
{
  // Generate bbox based on annotation type
  if ( "boundingBox" == parsed_line[ index ] )
  {
    if ( parsed_line.size() > index + 4 )
    {
      bbox = kwiver::vital::bounding_box_d(
        atof( parsed_line[ index + 1 ].c_str() ),
        atof( parsed_line[ index + 2 ].c_str() ),
        atof( parsed_line[ index + 3 ].c_str() ),
        atof( parsed_line[ index + 4 ].c_str() ) );
    }
    else
    {
      LOG_WARN( m_parent->logger(), "Invalid format for boundingBox annotation" );
      return false;
    }
  }
  else if ( "line" == parsed_line[ index ] )
  {
    if ( parsed_line.size() > index + 4 )
    {
      const double x1 = atof( parsed_line[ index + 1 ].c_str() );
      const double y1 = atof( parsed_line[ index + 2 ].c_str() );
      const double x2 = atof( parsed_line[ index + 3 ].c_str() );
      const double y2 = atof( parsed_line[ index + 4 ].c_str() );

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
      LOG_WARN( m_parent->logger(), "Invalid format for line annotation" );
      return false;
    }
  }
  else if ( "point" == parsed_line[ index ] )
  {
    if ( parsed_line.size() > index + 2 )
    {
      const double cx = atof( parsed_line[ index + 1 ].c_str() );
      const double cy = atof( parsed_line[ index + 2 ].c_str() );

      bbox = kwiver::vital::bounding_box_d(
        cx - m_point_dilation, cy - m_point_dilation,
        cx + m_point_dilation, cy + m_point_dilation );
    }
    else
    {
      LOG_WARN( m_parent->logger(), "Invalid format for point annotation" );
      return false;
    }
  }
  else if ( "circle" == parsed_line[index] )
  {
    if ( parsed_line.size() > index + 4 )
    {
      const double x1 = atof( parsed_line[ index + 1 ].c_str() );
      const double y1 = atof( parsed_line[ index + 2 ].c_str() );
      const double x2 = atof( parsed_line[ index + 3 ].c_str() );
      const double y2 = atof( parsed_line[ index + 4 ].c_str() );

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
      LOG_WARN( m_parent->logger(), "Invalid format for circle annotation" );
      return false;
    }
  }
  else
  {
    return false;
  }

  return true;
} // read_detected_object_set_habcam::priv::parse_box


// -----------------------------------------------------------------------------
std::string
read_detected_object_set_habcam::priv
::decode_species( int code )
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


// =============================================================================
void
read_detected_object_set_habcam::priv
::read_all()
{
  std::string line;
  kwiver::vital::data_stream_reader stream_reader( m_parent->stream() );

  m_gt_sets.clear();

  while( stream_reader.getline( line ) )
  {
    if( line.substr(0,6) == "Image," || line.empty() ) //ignore header
    {
      continue;
    }

    // Automatically figure out delim if not set
    if( m_delim.empty() )
    {
      if( line.find( ',' ) != std::string::npos )
      {
        m_delim = ",";
        m_detected_version = 2;
      }
      else
      {
        m_delim = " ";
        m_detected_version = 1;
      }
    }

    if( m_detected_version == 2 )
    {
      line.erase( std::remove( line.begin(), line.end(), '[' ), line.end() );
      line.erase( std::remove( line.begin(), line.end(), ']' ), line.end() );
      line.erase( std::remove( line.begin(), line.end(), '\"' ), line.end() );
      line.erase( std::remove( line.begin(), line.end(), ':' ), line.end() );
    }
    else
    {
      line.erase( std::remove( line.begin(), line.end(), '"' ), line.end() );
    }

    std::vector< std::string > parsed_line;
    kwiver::vital::tokenize( line, parsed_line, m_delim, true );

    // Test the minimum number of fields.
    if ( parsed_line.size() < 4 )
    {
      std::cout << "Invalid line: " << line << std::endl;
      continue;
    }

    // Make 'v2' formats look like 'v1' so they can share parsing code
    if( m_detected_version == 2 )
    {
      std::vector< std::string > parsed_loc;
      kwiver::vital::tokenize( parsed_line[3], parsed_loc, " ", true );
      parsed_line.erase( parsed_line.begin() + 3 );

      if( parsed_loc.size() != 2 )
      {
        throw kwiver::vital::invalid_data( "Invalid line: " + line );
      }

      parsed_line.insert( parsed_line.begin() + 3, parsed_loc.begin(), parsed_loc.end() );
    }

    parse_detection( parsed_line );
  }

  std::sort( m_filenames.begin(), m_filenames.end() );
}


// -----------------------------------------------------------------------------
void
read_detected_object_set_habcam::priv
::init_species_map()
{
  // Could read the species definition from a file.
  // The map will be constant and large, so this could be class static.
  // Probably not more than one reader instantiated at a time.
  m_species_map[185] = "live_scallop";
  m_species_map[197] = "live_scallop";
  m_species_map[207] = "live_scallop";
  m_species_map[211] = "live_scallop";
  m_species_map[515] = "live_scallop";
  m_species_map[912] = "live_scallop";
  m_species_map[919] = "live_scallop";
  m_species_map[920] = "live_scallop";
  m_species_map[188] = "dead_scallop";
  m_species_map[403] = "sand_eel";
  m_species_map[524] = "skate";
  m_species_map[533] = "fish";
  m_species_map[1003] = "fish";
  m_species_map[1001] = "fish";
  m_species_map[158] = "crab";
  m_species_map[258] = "crab";
}

} // end namespace
