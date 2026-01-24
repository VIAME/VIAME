/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Implementation for read_detected_object_set_yolo
 */

#include "read_detected_object_set_yolo.h"

#include <vital/util/tokenize.h>
#include <vital/util/data_stream_reader.h>
#include <vital/exceptions.h>

#include <kwiversys/SystemTools.hxx>

#include <map>
#include <memory>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <algorithm>

#ifdef VIAME_ENABLE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#endif


namespace viame {

// -----------------------------------------------------------------------------------
// Helper function to read image dimensions from file header
// Works for common formats without requiring full image load
// -----------------------------------------------------------------------------------
namespace {

bool get_image_dimensions_from_header( std::string const& filename,
                                       int& width, int& height )
{
  std::ifstream ifs( filename, std::ios::binary );
  if( !ifs )
  {
    return false;
  }

  unsigned char header[32];
  ifs.read( reinterpret_cast< char* >( header ), 32 );
  if( ifs.gcount() < 24 )
  {
    return false;
  }

  // Check for PNG signature: 89 50 4E 47 0D 0A 1A 0A
  if( header[0] == 0x89 && header[1] == 0x50 && header[2] == 0x4E &&
      header[3] == 0x47 && header[4] == 0x0D && header[5] == 0x0A &&
      header[6] == 0x1A && header[7] == 0x0A )
  {
    // PNG: width and height are at bytes 16-23 (big-endian)
    width = ( header[16] << 24 ) | ( header[17] << 16 ) |
            ( header[18] << 8 ) | header[19];
    height = ( header[20] << 24 ) | ( header[21] << 16 ) |
             ( header[22] << 8 ) | header[23];
    return true;
  }

  // Check for JPEG signature: FF D8 FF
  if( header[0] == 0xFF && header[1] == 0xD8 && header[2] == 0xFF )
  {
    // JPEG: need to find SOF0/SOF2 marker for dimensions
    ifs.seekg( 2 );
    while( ifs.good() )
    {
      unsigned char marker[2];
      ifs.read( reinterpret_cast< char* >( marker ), 2 );
      if( marker[0] != 0xFF )
      {
        break;
      }

      // SOF0 (0xC0) or SOF2 (0xC2) contain image dimensions
      if( marker[1] == 0xC0 || marker[1] == 0xC2 )
      {
        unsigned char sof[7];
        ifs.read( reinterpret_cast< char* >( sof ), 7 );
        if( ifs.gcount() == 7 )
        {
          height = ( sof[3] << 8 ) | sof[4];
          width = ( sof[5] << 8 ) | sof[6];
          return true;
        }
        break;
      }

      // Skip this segment
      unsigned char len[2];
      ifs.read( reinterpret_cast< char* >( len ), 2 );
      int segment_len = ( len[0] << 8 ) | len[1];
      ifs.seekg( segment_len - 2, std::ios::cur );
    }
  }

  return false;
}

} // anonymous namespace


// -----------------------------------------------------------------------------------
class read_detected_object_set_yolo::priv
{
public:
  priv( read_detected_object_set_yolo* parent )
    : m_parent( parent )
    , m_first( true )
    , m_image_width( 0 )
    , m_image_height( 0 )
    , m_default_confidence( 1.0 )
    , m_current_idx( 0 )
  { }

  ~priv() { }

  void read_all();
  bool load_classes( std::string const& filename );
  std::string find_label_file( std::string const& image_path );
  std::string find_classes_file( std::string const& image_path );
  bool detect_image_dimensions( std::string const& image_path );
  kwiver::vital::detected_object_set_sptr read_labels( std::string const& label_path );

  read_detected_object_set_yolo* m_parent;
  bool m_first;

  // Configuration
  std::string m_classes_file;
  int m_image_width;
  int m_image_height;
  double m_default_confidence;

  // Class names loaded from file
  std::vector< std::string > m_class_names;

  // List of image paths from the input file
  std::vector< std::string > m_image_list;
  int m_current_idx;

  // Map of detected objects indexed by image path
  std::map< std::string, kwiver::vital::detected_object_set_sptr > m_detection_by_str;
};


// ===================================================================================
read_detected_object_set_yolo
::read_detected_object_set_yolo()
  : d( new read_detected_object_set_yolo::priv( this ) )
{
  attach_logger( "viame.core.read_detected_object_set_yolo" );
}


read_detected_object_set_yolo
::~read_detected_object_set_yolo()
{
}


// -----------------------------------------------------------------------------------
kwiver::vital::config_block_sptr
read_detected_object_set_yolo
::get_configuration() const
{
  auto config = kwiver::vital::algo::detected_object_set_input::get_configuration();

  config->set_value( "classes_file", d->m_classes_file,
    "Path to file containing class names, one per line. "
    "Line number corresponds to class ID (0-indexed). "
    "If empty, will search for 'labels.txt' in image directory or parent." );

  config->set_value( "image_width", d->m_image_width,
    "Width of images in pixels. If 0, will auto-detect from first image." );

  config->set_value( "image_height", d->m_image_height,
    "Height of images in pixels. If 0, will auto-detect from first image." );

  config->set_value( "default_confidence", d->m_default_confidence,
    "Default confidence score to use when not specified in label file." );

  return config;
}


// -----------------------------------------------------------------------------------
void
read_detected_object_set_yolo
::set_configuration( kwiver::vital::config_block_sptr config )
{
  d->m_classes_file =
    config->get_value< std::string >( "classes_file", d->m_classes_file );
  d->m_image_width =
    config->get_value< int >( "image_width", d->m_image_width );
  d->m_image_height =
    config->get_value< int >( "image_height", d->m_image_height );
  d->m_default_confidence =
    config->get_value< double >( "default_confidence", d->m_default_confidence );

  // Load class names if file specified
  if( !d->m_classes_file.empty() )
  {
    d->load_classes( d->m_classes_file );
  }
}


// -----------------------------------------------------------------------------------
bool
read_detected_object_set_yolo
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  // No longer require image dimensions - they can be auto-detected
  return true;
}


// -----------------------------------------------------------------------------------
bool
read_detected_object_set_yolo
::read_set( kwiver::vital::detected_object_set_sptr& set, std::string& image_name )
{
  if( d->m_first )
  {
    // Read in all image paths and their detections
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
      // Try to find label file for this image on-the-fly
      std::string label_path = d->find_label_file( image_name );
      set = d->read_labels( label_path );
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
read_detected_object_set_yolo
::new_stream()
{
  d->m_first = true;
  d->m_image_list.clear();
  d->m_detection_by_str.clear();
  d->m_current_idx = 0;
}


// ===================================================================================
bool
read_detected_object_set_yolo::priv
::load_classes( std::string const& filename )
{
  m_class_names.clear();

  std::ifstream ifs( filename );
  if( !ifs )
  {
    LOG_WARN( m_parent->logger(), "Could not open classes file: " << filename );
    return false;
  }

  std::string line;
  while( std::getline( ifs, line ) )
  {
    // Trim whitespace
    size_t start = line.find_first_not_of( " \t\r\n" );
    if( start == std::string::npos )
    {
      continue; // Empty line
    }
    size_t end = line.find_last_not_of( " \t\r\n" );
    m_class_names.push_back( line.substr( start, end - start + 1 ) );
  }

  LOG_DEBUG( m_parent->logger(), "Loaded " << m_class_names.size() << " class names from " << filename );
  return true;
}


// -----------------------------------------------------------------------------------
std::string
read_detected_object_set_yolo::priv
::find_classes_file( std::string const& image_path )
{
  std::string image_dir = kwiversys::SystemTools::GetFilenamePath( image_path );

  // Strategy 1: labels.txt in same directory as images
  std::string classes_path = image_dir + "/labels.txt";
  if( kwiversys::SystemTools::FileExists( classes_path ) )
  {
    return classes_path;
  }

  // Strategy 2: labels.txt in parent directory
  std::string parent_dir = kwiversys::SystemTools::GetFilenamePath( image_dir );
  classes_path = parent_dir + "/labels.txt";
  if( kwiversys::SystemTools::FileExists( classes_path ) )
  {
    return classes_path;
  }

  // Strategy 3: classes.txt in same directory (alternative name)
  classes_path = image_dir + "/classes.txt";
  if( kwiversys::SystemTools::FileExists( classes_path ) )
  {
    return classes_path;
  }

  // Strategy 4: classes.txt in parent directory
  classes_path = parent_dir + "/classes.txt";
  if( kwiversys::SystemTools::FileExists( classes_path ) )
  {
    return classes_path;
  }

  // Strategy 5: Check in data.yaml or similar YOLO config files
  // Look for classes.names (common YOLO convention)
  classes_path = image_dir + "/classes.names";
  if( kwiversys::SystemTools::FileExists( classes_path ) )
  {
    return classes_path;
  }

  classes_path = parent_dir + "/classes.names";
  if( kwiversys::SystemTools::FileExists( classes_path ) )
  {
    return classes_path;
  }

  return std::string();
}


// -----------------------------------------------------------------------------------
bool
read_detected_object_set_yolo::priv
::detect_image_dimensions( std::string const& image_path )
{
  if( !kwiversys::SystemTools::FileExists( image_path ) )
  {
    LOG_WARN( m_parent->logger(), "Image file does not exist: " << image_path );
    return false;
  }

#ifdef VIAME_ENABLE_OPENCV
  // Use OpenCV to get image dimensions
  cv::Mat img = cv::imread( image_path, cv::IMREAD_UNCHANGED );
  if( img.empty() )
  {
    LOG_WARN( m_parent->logger(), "OpenCV could not read image: " << image_path );
    return false;
  }

  m_image_width = img.cols;
  m_image_height = img.rows;

  LOG_INFO( m_parent->logger(), "Auto-detected image dimensions: "
            << m_image_width << "x" << m_image_height << " from " << image_path );
  return true;
#else
  // Try to read dimensions from image header
  int width = 0, height = 0;
  if( get_image_dimensions_from_header( image_path, width, height ) )
  {
    m_image_width = width;
    m_image_height = height;

    LOG_INFO( m_parent->logger(), "Auto-detected image dimensions: "
              << m_image_width << "x" << m_image_height << " from " << image_path );
    return true;
  }

  LOG_ERROR( m_parent->logger(),
             "Could not auto-detect image dimensions. "
             "Please specify image_width and image_height in configuration, "
             "or enable OpenCV support." );
  return false;
#endif
}


// -----------------------------------------------------------------------------------
std::string
read_detected_object_set_yolo::priv
::find_label_file( std::string const& image_path )
{
  // Get the base filename and replace extension with .txt
  std::string base_name = kwiversys::SystemTools::GetFilenameWithoutLastExtension( image_path );
  std::string txt_name = base_name + ".txt";
  std::string image_dir = kwiversys::SystemTools::GetFilenamePath( image_path );

  // Strategy 1: Same directory as image
  std::string label_path = image_dir + "/" + txt_name;
  if( kwiversys::SystemTools::FileExists( label_path ) )
  {
    return label_path;
  }

  // Strategy 2: ../labels/subdir/image.txt (parallel labels directory structure)
  std::string parent_dir = kwiversys::SystemTools::GetFilenamePath( image_dir );
  std::string subdir_name = kwiversys::SystemTools::GetFilenameName( image_dir );
  label_path = parent_dir + "/labels/" + subdir_name + "/" + txt_name;
  if( kwiversys::SystemTools::FileExists( label_path ) )
  {
    return label_path;
  }

  // Strategy 3: ../labels/image.txt (simpler structure)
  label_path = parent_dir + "/labels/" + txt_name;
  if( kwiversys::SystemTools::FileExists( label_path ) )
  {
    return label_path;
  }

  // No label file found - return empty string
  return std::string();
}


// -----------------------------------------------------------------------------------
kwiver::vital::detected_object_set_sptr
read_detected_object_set_yolo::priv
::read_labels( std::string const& label_path )
{
  auto det_set = std::make_shared< kwiver::vital::detected_object_set >();

  if( label_path.empty() || !kwiversys::SystemTools::FileExists( label_path ) )
  {
    // No label file means no detections for this image
    return det_set;
  }

  // Check if we have valid image dimensions
  if( m_image_width <= 0 || m_image_height <= 0 )
  {
    LOG_WARN( m_parent->logger(),
              "Cannot read labels without valid image dimensions" );
    return det_set;
  }

  std::ifstream ifs( label_path );
  if( !ifs )
  {
    LOG_WARN( m_parent->logger(), "Could not open label file: " << label_path );
    return det_set;
  }

  std::string line;
  int line_num = 0;

  while( std::getline( ifs, line ) )
  {
    ++line_num;

    // Skip empty lines and comments
    size_t start = line.find_first_not_of( " \t\r\n" );
    if( start == std::string::npos || line[start] == '#' )
    {
      continue;
    }

    std::vector< std::string > tokens;
    kwiver::vital::tokenize( line, tokens, " \t", kwiver::vital::TokenizeTrimEmpty );

    if( tokens.size() < 5 )
    {
      LOG_WARN( m_parent->logger(),
                "Invalid YOLO format at " << label_path << ":" << line_num
                << " - expected at least 5 fields, got " << tokens.size() );
      continue;
    }

    // Parse YOLO format: class_id x_center y_center width height [confidence]
    int class_id = std::atoi( tokens[0].c_str() );
    double x_center_norm = std::atof( tokens[1].c_str() );
    double y_center_norm = std::atof( tokens[2].c_str() );
    double width_norm = std::atof( tokens[3].c_str() );
    double height_norm = std::atof( tokens[4].c_str() );

    double confidence = m_default_confidence;
    if( tokens.size() > 5 )
    {
      confidence = std::atof( tokens[5].c_str() );
    }

    // Convert normalized coordinates to absolute pixels
    double x_center = x_center_norm * m_image_width;
    double y_center = y_center_norm * m_image_height;
    double width = width_norm * m_image_width;
    double height = height_norm * m_image_height;

    // Convert center format to corner format (TL-x, TL-y, BR-x, BR-y)
    double x1 = x_center - width / 2.0;
    double y1 = y_center - height / 2.0;
    double x2 = x_center + width / 2.0;
    double y2 = y_center + height / 2.0;

    kwiver::vital::bounding_box_d bbox( x1, y1, x2, y2 );

    // Get class name
    std::string class_name;
    if( class_id >= 0 && static_cast< size_t >( class_id ) < m_class_names.size() )
    {
      class_name = m_class_names[class_id];
    }
    else
    {
      class_name = std::to_string( class_id );
    }

    // Create detected object type
    auto dot = std::make_shared< kwiver::vital::detected_object_type >();
    dot->set_score( class_name, confidence );

    // Create and add detection
    auto det = std::make_shared< kwiver::vital::detected_object >( bbox, confidence, dot );
    det_set->add( det );
  }

  return det_set;
}


// -----------------------------------------------------------------------------------
void
read_detected_object_set_yolo::priv
::read_all()
{
  std::string line;
  kwiver::vital::data_stream_reader stream_reader( m_parent->stream() );

  m_image_list.clear();
  m_detection_by_str.clear();

  // First pass: Read list of image paths from the input file
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
    std::string image_path = line.substr( start, end - start + 1 );

    m_image_list.push_back( image_path );
  }

  if( m_image_list.empty() )
  {
    LOG_WARN( m_parent->logger(), "No images found in input file" );
    return;
  }

  // Auto-detect image dimensions if not specified
  if( m_image_width <= 0 || m_image_height <= 0 )
  {
    // Try to detect from the first valid image
    for( auto const& image_path : m_image_list )
    {
      if( detect_image_dimensions( image_path ) )
      {
        break;
      }
    }

    if( m_image_width <= 0 || m_image_height <= 0 )
    {
      LOG_ERROR( m_parent->logger(),
                 "Could not auto-detect image dimensions. "
                 "Labels will not be loaded correctly." );
    }
  }

  // Auto-detect classes file if not specified
  if( m_classes_file.empty() && m_class_names.empty() )
  {
    std::string auto_classes = find_classes_file( m_image_list[0] );
    if( !auto_classes.empty() )
    {
      LOG_INFO( m_parent->logger(), "Auto-detected classes file: " << auto_classes );
      load_classes( auto_classes );
    }
    else
    {
      LOG_DEBUG( m_parent->logger(),
                 "No classes file found - using numeric class IDs" );
    }
  }

  // Second pass: Load detections for each image
  for( auto const& image_path : m_image_list )
  {
    std::string label_path = find_label_file( image_path );
    m_detection_by_str[ image_path ] = read_labels( label_path );
  }

  LOG_DEBUG( m_parent->logger(),
             "Loaded detections for " << m_image_list.size() << " images" );
}

} // end namespace viame
