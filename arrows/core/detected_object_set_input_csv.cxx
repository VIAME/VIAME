// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation for detected_object_set_input_csv
 */

#include "detected_object_set_input_csv.h"

#include <vital/util/tokenize.h>
#include <vital/util/data_stream_reader.h>
#include <vital/exceptions.h>
#include <vital/vital_config.h>

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

  // -------------------------------------
  detected_object_set_input_csv* m_parent;
  bool m_first;
  int m_frame_number;
  std::string m_delim;

  std::shared_ptr< kwiver::vital::data_stream_reader > m_stream_reader;
  std::vector< std::string > m_input_buffer;
  kwiver::vital::detected_object_set_sptr m_current_set;
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
check_configuration( VITAL_UNUSED vital::config_block_sptr config) const
{
  return true;
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

    // allocate first detection set
    d->m_current_set = std::make_shared<kwiver::vital::detected_object_set>();

    // set current frame number from line in buffer
    d->m_frame_number = atoi( d->m_input_buffer[0].c_str() );
  } // end first

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
    VITAL_THROW( kwiver::vital::invalid_data, str.str() );
  }

  if ( ! ( m_input_buffer.size() & 0x001 ) )
  {
    std::stringstream str;
    str << "Invalid format in input at line " << m_stream_reader->line_number() << std::endl
        << "\"" << line << "\"";
    VITAL_THROW( kwiver::vital::invalid_data, str.str() );
  }

  return true;
}

// ------------------------------------------------------------------
void
detected_object_set_input_csv::priv::
add_detection()
{
  kwiver::vital::detected_object_type_sptr dot;

  // Create DOT object if classifiers are present
  if ( m_input_buffer.size() > 7 )
  {
    dot = std::make_shared<kwiver::vital::detected_object_type>();
    const size_t limit( m_input_buffer.size() );

    for (size_t i = 7; i < limit; i += 2 )
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
