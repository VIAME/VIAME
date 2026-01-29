/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Implementation for read_detected_object_set_cvat
 */

#include "read_detected_object_set_cvat.h"

#include <vital/util/data_stream_reader.h>
#include <vital/exceptions.h>

#include <tinyxml.h>

#include <map>
#include <memory>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <algorithm>


namespace viame {

// -----------------------------------------------------------------------------------
class read_detected_object_set_cvat::priv
{
public:
  priv( read_detected_object_set_cvat& parent )
    : m_parent( &parent )
    , m_first( true )
    , m_current_idx( 0 )
  { }

  ~priv() { }

  void read_all();
  void parse_xml_file( std::string const& filename );
  kwiver::vital::detected_object_set_sptr parse_image_element( TiXmlElement* image_elem );
  std::vector< double > parse_polygon_points( std::string const& points_str );

  read_detected_object_set_cvat* m_parent;
  bool m_first;

  // List of image names in order
  std::vector< std::string > m_image_list;
  int m_current_idx;

  // Map of detected objects indexed by image name
  std::map< std::string, kwiver::vital::detected_object_set_sptr > m_detection_by_str;
};


// ===================================================================================
read_detected_object_set_cvat
::~read_detected_object_set_cvat()
{
}


// -----------------------------------------------------------------------------------
void
read_detected_object_set_cvat
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d );
  attach_logger( "viame.core.read_detected_object_set_cvat" );
}


// -----------------------------------------------------------------------------------
bool
read_detected_object_set_cvat
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  return true;
}


// -----------------------------------------------------------------------------------
bool
read_detected_object_set_cvat
::read_set( kwiver::vital::detected_object_set_sptr& set, std::string& image_name )
{
  if( d->m_first )
  {
    // Read in all detections from the XML file
    d->read_all();
    d->m_first = false;
    d->m_current_idx = 0;
  }

  // External image name provided, use that to look up detections
  if( !image_name.empty() && !d->m_detection_by_str.empty() )
  {
    auto itr = d->m_detection_by_str.find( image_name );
    if( itr != d->m_detection_by_str.end() )
    {
      set = itr->second;
    }
    else
    {
      // Try matching by filename only (without path)
      std::string basename = image_name;
      size_t pos = image_name.find_last_of( "/\\" );
      if( pos != std::string::npos )
      {
        basename = image_name.substr( pos + 1 );
      }

      bool found = false;
      for( auto const& entry : d->m_detection_by_str )
      {
        std::string entry_basename = entry.first;
        pos = entry.first.find_last_of( "/\\" );
        if( pos != std::string::npos )
        {
          entry_basename = entry.first.substr( pos + 1 );
        }

        if( entry_basename == basename )
        {
          set = entry.second;
          found = true;
          break;
        }
      }

      if( !found )
      {
        set = std::make_shared< kwiver::vital::detected_object_set >();
      }
    }
    return true;
  }

  // Test for end of all loaded images
  if( d->m_current_idx >= static_cast< int >( d->m_image_list.size() ) )
  {
    set = std::make_shared< kwiver::vital::detected_object_set >();
    return false;
  }

  // Return detection set for current image
  image_name = d->m_image_list[ d->m_current_idx ];

  auto itr = d->m_detection_by_str.find( image_name );
  if( itr != d->m_detection_by_str.end() )
  {
    set = itr->second;
  }
  else
  {
    set = std::make_shared< kwiver::vital::detected_object_set >();
  }

  ++d->m_current_idx;
  return true;
}


// -----------------------------------------------------------------------------------
void
read_detected_object_set_cvat
::new_stream()
{
  d->m_first = true;
  d->m_image_list.clear();
  d->m_detection_by_str.clear();
  d->m_current_idx = 0;
}


// ===================================================================================
void
read_detected_object_set_cvat::priv
::read_all()
{
  m_image_list.clear();
  m_detection_by_str.clear();

  // Read the XML filename from the stream (first non-empty, non-comment line)
  std::string line;
  kwiver::vital::data_stream_reader stream_reader( m_parent->stream() );

  while( stream_reader.getline( line ) )
  {
    // Trim whitespace
    size_t start = line.find_first_not_of( " \t\r\n" );
    if( start == std::string::npos )
    {
      continue; // Skip empty lines
    }
    if( line[start] == '#' )
    {
      continue; // Skip comments
    }

    size_t end = line.find_last_not_of( " \t\r\n" );
    std::string xml_file = line.substr( start, end - start + 1 );

    // Parse the XML file
    parse_xml_file( xml_file );
  }

  LOG_DEBUG( m_parent->logger(),
             "Loaded detections for " << m_image_list.size() << " images from CVAT XML" );
}


// -----------------------------------------------------------------------------------
void
read_detected_object_set_cvat::priv
::parse_xml_file( std::string const& filename )
{
  TiXmlDocument doc( filename.c_str() );

  if( !doc.LoadFile() )
  {
    LOG_ERROR( m_parent->logger(),
               "TinyXML couldn't load CVAT file '" << filename << "': "
               << doc.ErrorDesc() );
    return;
  }

  TiXmlElement* root = doc.RootElement();
  if( !root )
  {
    LOG_ERROR( m_parent->logger(), "CVAT XML file has no root element" );
    return;
  }

  // Verify this is an annotations element
  if( std::string( root->Value() ) != "annotations" )
  {
    LOG_ERROR( m_parent->logger(),
               "CVAT XML root element should be 'annotations', got '"
               << root->Value() << "'" );
    return;
  }

  // Iterate through image elements
  for( TiXmlElement* elem = root->FirstChildElement();
       elem != nullptr;
       elem = elem->NextSiblingElement() )
  {
    if( std::string( elem->Value() ) == "image" )
    {
      // Get image name
      char const* name_attr = elem->Attribute( "name" );
      if( !name_attr )
      {
        LOG_WARN( m_parent->logger(), "Image element missing 'name' attribute" );
        continue;
      }

      std::string image_name( name_attr );
      m_image_list.push_back( image_name );

      // Parse detections for this image
      m_detection_by_str[ image_name ] = parse_image_element( elem );
    }
  }
}


// -----------------------------------------------------------------------------------
kwiver::vital::detected_object_set_sptr
read_detected_object_set_cvat::priv
::parse_image_element( TiXmlElement* image_elem )
{
  auto det_set = std::make_shared< kwiver::vital::detected_object_set >();

  // Iterate through child elements (box, polygon, etc.)
  for( TiXmlElement* elem = image_elem->FirstChildElement();
       elem != nullptr;
       elem = elem->NextSiblingElement() )
  {
    std::string elem_type( elem->Value() );

    if( elem_type == "box" )
    {
      // Parse box: label, xtl, ytl, xbr, ybr
      char const* label_attr = elem->Attribute( "label" );
      char const* xtl_attr = elem->Attribute( "xtl" );
      char const* ytl_attr = elem->Attribute( "ytl" );
      char const* xbr_attr = elem->Attribute( "xbr" );
      char const* ybr_attr = elem->Attribute( "ybr" );

      if( !label_attr || !xtl_attr || !ytl_attr || !xbr_attr || !ybr_attr )
      {
        LOG_WARN( m_parent->logger(), "Box element missing required attributes" );
        continue;
      }

      std::string label( label_attr );
      double xtl = std::atof( xtl_attr );
      double ytl = std::atof( ytl_attr );
      double xbr = std::atof( xbr_attr );
      double ybr = std::atof( ybr_attr );

      kwiver::vital::bounding_box_d bbox( xtl, ytl, xbr, ybr );

      // Create detected object type
      auto dot = std::make_shared< kwiver::vital::detected_object_type >();
      dot->set_score( label, m_parent->c_default_confidence );

      // Create and add detection
      auto det = std::make_shared< kwiver::vital::detected_object >(
        bbox, m_parent->c_default_confidence, dot );
      det_set->add( det );
    }
    else if( elem_type == "polygon" )
    {
      // Parse polygon: label, points
      char const* label_attr = elem->Attribute( "label" );
      char const* points_attr = elem->Attribute( "points" );

      if( !label_attr || !points_attr )
      {
        LOG_WARN( m_parent->logger(), "Polygon element missing required attributes" );
        continue;
      }

      std::string label( label_attr );
      std::vector< double > points = parse_polygon_points( points_attr );

      if( points.size() < 6 ) // Need at least 3 points (6 coordinates)
      {
        LOG_WARN( m_parent->logger(), "Polygon has fewer than 3 points" );
        continue;
      }

      // Compute bounding box from polygon points
      double min_x = points[0], max_x = points[0];
      double min_y = points[1], max_y = points[1];

      for( size_t i = 2; i < points.size(); i += 2 )
      {
        min_x = std::min( min_x, points[i] );
        max_x = std::max( max_x, points[i] );
        min_y = std::min( min_y, points[i + 1] );
        max_y = std::max( max_y, points[i + 1] );
      }

      kwiver::vital::bounding_box_d bbox( min_x, min_y, max_x, max_y );

      // Create detected object type
      auto dot = std::make_shared< kwiver::vital::detected_object_type >();
      dot->set_score( label, m_parent->c_default_confidence );

      // Create detection with polygon
      auto det = std::make_shared< kwiver::vital::detected_object >(
        bbox, m_parent->c_default_confidence, dot );
      det->set_flattened_polygon( points );
      det_set->add( det );
    }
    else if( elem_type == "polyline" )
    {
      // Parse polyline similar to polygon
      char const* label_attr = elem->Attribute( "label" );
      char const* points_attr = elem->Attribute( "points" );

      if( !label_attr || !points_attr )
      {
        LOG_WARN( m_parent->logger(), "Polyline element missing required attributes" );
        continue;
      }

      std::string label( label_attr );
      std::vector< double > points = parse_polygon_points( points_attr );

      if( points.size() < 4 ) // Need at least 2 points
      {
        continue;
      }

      // Compute bounding box from polyline points
      double min_x = points[0], max_x = points[0];
      double min_y = points[1], max_y = points[1];

      for( size_t i = 2; i < points.size(); i += 2 )
      {
        min_x = std::min( min_x, points[i] );
        max_x = std::max( max_x, points[i] );
        min_y = std::min( min_y, points[i + 1] );
        max_y = std::max( max_y, points[i + 1] );
      }

      kwiver::vital::bounding_box_d bbox( min_x, min_y, max_x, max_y );

      // Create detected object type
      auto dot = std::make_shared< kwiver::vital::detected_object_type >();
      dot->set_score( label, m_parent->c_default_confidence );

      // Create detection
      auto det = std::make_shared< kwiver::vital::detected_object >(
        bbox, m_parent->c_default_confidence, dot );
      det_set->add( det );
    }
  }

  return det_set;
}


// -----------------------------------------------------------------------------------
std::vector< double >
read_detected_object_set_cvat::priv
::parse_polygon_points( std::string const& points_str )
{
  // CVAT polygon points format: "x1,y1;x2,y2;x3,y3;..."
  std::vector< double > result;

  std::string::size_type pos = 0;
  std::string::size_type prev = 0;

  while( ( pos = points_str.find( ';', prev ) ) != std::string::npos )
  {
    std::string point_pair = points_str.substr( prev, pos - prev );
    std::string::size_type comma = point_pair.find( ',' );

    if( comma != std::string::npos )
    {
      result.push_back( std::atof( point_pair.substr( 0, comma ).c_str() ) );
      result.push_back( std::atof( point_pair.substr( comma + 1 ).c_str() ) );
    }

    prev = pos + 1;
  }

  // Don't forget the last point
  std::string point_pair = points_str.substr( prev );
  std::string::size_type comma = point_pair.find( ',' );

  if( comma != std::string::npos )
  {
    result.push_back( std::atof( point_pair.substr( 0, comma ).c_str() ) );
    result.push_back( std::atof( point_pair.substr( comma + 1 ).c_str() ) );
  }

  return result;
}

} // end namespace viame
