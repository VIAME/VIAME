/*ckwg +29
 * Copyright 2016-2018 by Kitware, Inc.
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
 * \brief Implementation for detected_object_set_input_csv
 */

#include "detected_object_set_input_csv.h"

#include <vital/util/tokenize.h>
#include <vital/util/data_stream_reader.h>
#include <vital/exceptions.h>

#include <sstream>
#include <cstdlib>

namespace kwiver {
namespace arrows {
namespace core {

/// Expected format:
/// - 1: frame number
/// - 2: file name
/// - 3: TL-x
/// - 4: TL-y
/// - 5: BR-x
/// - 6: BR-y
/// - 7: confidence
/// - 8,9  : class-name   score  (this pair may be omitted or may repeat any number of times)
///

// ------------------------------------------------------------------
class detected_object_set_input_csv::priv
{
public:
  priv( detected_object_set_input_csv* parent)
    : m_parent( parent )
    , m_first( true )
    , m_frame_number( 0 )
    , m_delim( "," )
  {
  }

  ~priv() { }

  bool get_input();
  void add_detection();
  void add_detection( const std::vector< std::string >& parsed_line);
  void read_all();

  // -------------------------------------
  detected_object_set_input_csv* m_parent;
  bool m_first;
  int m_frame_number;
  std::string m_delim;

  std::shared_ptr< kwiver::vital::data_stream_reader > m_stream_reader;
  std::vector< std::string > m_input_buffer;
  kwiver::vital::detected_object_set_sptr m_current_set;

  std::map< std::string, kwiver::vital::detected_object_set_sptr > m_gt_sets;
  std::vector< std::string > m_filenames;
  std::string m_image_name;
};


// ==================================================================
detected_object_set_input_csv::
detected_object_set_input_csv()
  : d( new detected_object_set_input_csv::priv( this ) )
{
  attach_logger( "arrows.core.detected_object_set_input_csv" );
}


detected_object_set_input_csv::
~detected_object_set_input_csv()
{
}


// ------------------------------------------------------------------
void
detected_object_set_input_csv::
set_configuration(vital::config_block_sptr config)
{
  d->m_delim = config->get_value<std::string>( "delimiter", d->m_delim );

  // Test for no specification which can happen due to config parsing issues.
  if ( d->m_delim.empty() )
  {
    d->m_delim = " ";
  }
}


// ------------------------------------------------------------------
bool
detected_object_set_input_csv::
check_configuration(vital::config_block_sptr config) const
{
  return true;
}


void
detected_object_set_input_csv::priv::
read_all()
{
  std::string line;
  kwiver::vital::data_stream_reader stream_reader( m_parent->stream() );
  //Reset the groundtruth sets
  m_gt_sets.clear();
  
  //Run through all the lines in the groundtruth file
  while( stream_reader.getline( line ) )
  {
    if( line.substr(0,6) == "Image,") //ignore header
    {
      continue;
    }

    line.erase( std::remove( line.begin(), line.end(), '[' ), line.end() );
    line.erase( std::remove( line.begin(), line.end(), ']' ), line.end() );
    line.erase( std::remove( line.begin(), line.end(), '\"' ), line.end() );
    line.erase( std::remove( line.begin(), line.end(), ':' ), line.end() );
    
    //Parse the file on commas, not that whitespace is not stripped 
    std::vector< std::string > parsed_line;
    kwiver::vital::tokenize( line, parsed_line, ",", true );

    // Test the minimum number of fields.
    if ( parsed_line.size() < 7 )
    {
      LOG_WARN( m_parent->logger(), "Invalid line, too few fields: " << line );
      continue;
    }

    // Add the detection to the map from image name to detection set
    add_detection( parsed_line );
  }

  std::sort( m_filenames.begin(), m_filenames.end() );
}


// ------------------------------------------------------------------
bool
detected_object_set_input_csv::
read_set( kwiver::vital::detected_object_set_sptr & set, std::string& image_name )
{
  if ( d->m_first )
  {
    d->m_first = false;
    if ( ! d->get_input() )
    {
      return false; // indicate end of file.
    }

    // Setting up the map from file name to detection set only if we are going to use it
    if ( !image_name.empty() )
    {
      d->read_all();
    }

    // allocate first detection set
    d->m_current_set = std::make_shared<kwiver::vital::detected_object_set>();

    // set current frame number from line in buffer
    d->m_frame_number = atoi( d->m_input_buffer[0].c_str() );
  } // end first

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

    // Set the image name so it can be used by calling function
    image_name = d->m_image_name;
    return true;
  }
  // All further code is run only if no file name is passed in

  // test for end of stream
  if (this->at_eof())
  {
    return false;
  }

  bool valid_line( true );
  int frame;

  while( true )
  {
    // check buffer to see if it has the current frame number
    frame = atoi( d->m_input_buffer[0].c_str() );
    if ( valid_line && ( frame == d->m_frame_number ) )
    {
      // We are in the same frame, so add this detection to current set
      d->add_detection();
      // Get next input line
      valid_line = d->get_input();
    }
    else
    {
      break;
    }
  } // end while

  // we have found end of file or a new frame number. Return current set
  set = d->m_current_set;
  image_name = d->m_image_name;

  d->m_frame_number = frame;
  d->m_current_set = std::make_shared<kwiver::vital::detected_object_set>();

  return true;
}


// ------------------------------------------------------------------
void
detected_object_set_input_csv::
new_stream()
{
  d->m_first = true;
  d->m_stream_reader = std::make_shared< kwiver::vital::data_stream_reader>( stream() );
  d->m_filenames.clear();
  d->m_gt_sets.clear();
}


// ==================================================================
bool
detected_object_set_input_csv::priv::
get_input()
{
  std::string line;
  if ( ! m_stream_reader->getline( line ) )
  {
    return false; // end of file.
  }

  m_input_buffer.clear();
  kwiver::vital::tokenize( line, m_input_buffer, m_delim, kwiver::vital::TokenizeNoTrimEmpty );

  // Test the minimum number of fields.
  if ( m_input_buffer.size() < 7 )
  {
    std::stringstream str;
    str << "Too few field in input at line " << m_stream_reader->line_number() << std::endl
        << "\"" << line << "\"";
    throw kwiver::vital::invalid_data( str.str() );
  }

  if ( ! ( m_input_buffer.size() & 0x001 ) )
  {
    std::stringstream str;
    str << "Invalid format in input at line " << m_stream_reader->line_number() << std::endl
        << "\"" << line << "\"";
    throw kwiver::vital::invalid_data( str.str() );
  }

  return true;
}


// ------------------------------------------------------------------
void
detected_object_set_input_csv::priv::
add_detection( const std::vector< std::string >& parsed_line)
{
  kwiver::vital::detected_object_type_sptr dot;
  //The first seven fields are required in any detection
  const size_t num_required_fields( 7 );

  // Create DOT object if classifiers are present
  if ( parsed_line.size() > num_required_fields )
  {
    dot = std::make_shared<kwiver::vital::detected_object_type>();
    const size_t limit( parsed_line.size() );

    //The required fields may be followed by an arbitrary number of (class-name, score) pairs 
    for (size_t i = num_required_fields; i < limit; i += 2 )
    {
      double score = atof( parsed_line[i+1].c_str() );
      dot->set_score( parsed_line[i], score );
    }
  } // end classes

  const kwiver::vital::bounding_box_d bbox(
    atof( parsed_line[2].c_str() ),
    atof( parsed_line[3].c_str() ),
    atof( parsed_line[4].c_str() ),
    atof( parsed_line[5].c_str() ) );

  const double confid( atof( parsed_line[6].c_str() ) );

  //initializing the map
  if( m_gt_sets.find( parsed_line[0] ) == m_gt_sets.end() )
  {
    // create a new detection set entry
    m_gt_sets[ parsed_line[0] ] =
      std::make_shared<kwiver::vital::detected_object_set>();

    m_filenames.push_back( parsed_line[0] );
  }

  //adding a detection to the map from image name to detection set
  m_gt_sets[ parsed_line[0] ]->add( std::make_shared<kwiver::vital::detected_object>( bbox, confid, dot ) );
  
  m_image_name = parsed_line[1];
}


void
detected_object_set_input_csv::priv::
add_detection()
{
  kwiver::vital::detected_object_type_sptr dot;
  //The first seven fields are required in any detection
  const size_t num_required_fields( 7 );
                             
  // Create DOT object if classifiers are present
  if ( m_input_buffer.size() > num_required_fields )
  {
    dot = std::make_shared<kwiver::vital::detected_object_type>();
    const size_t limit( m_input_buffer.size() );
    //The required fields may be followed by an arbitrary number of (class-name, score) pairs 
    for (size_t i = num_required_fields; i < limit; i += 2 )
    {
      double score = atof( m_input_buffer[i+1].c_str() );
      dot->set_score( m_input_buffer[i], score );
    }
  } // end classes

  const kwiver::vital::bounding_box_d bbox(
    atof( m_input_buffer[2].c_str() ),
    atof( m_input_buffer[3].c_str() ),
    atof( m_input_buffer[4].c_str() ),
    atof( m_input_buffer[5].c_str() ) );

  const double confid( atof( m_input_buffer[6].c_str() ) );

  m_current_set->add( std::make_shared<kwiver::vital::detected_object>( bbox, confid, dot ) );

  m_image_name = m_input_buffer[1];
}

} } } // end namespace
