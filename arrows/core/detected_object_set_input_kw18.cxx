// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation for detected_object_set_input_kw18
 */

#include "detected_object_set_input_kw18.h"

#include <vital/util/tokenize.h>
#include <vital/util/data_stream_reader.h>
#include <vital/exceptions.h>
#include <vital/vital_config.h>

#include <map>
#include <sstream>
#include <cstdlib>

namespace kwiver {
namespace arrows {
namespace core {

// field numbers for KW18 file format
enum{
  COL_ID = 0,             // 0: Object ID
  COL_LEN,                // 1: Track length (always 1 for detections)
  COL_FRAME,              // 2: in this case, set index
  COL_LOC_X,    // 3
  COL_LOC_Y,    // 4
  COL_VEL_X,    // 5
  COL_VEL_Y,    // 6
  COL_IMG_LOC_X,// 7
  COL_IMG_LOC_Y,// 8
  COL_MIN_X,    // 9
  COL_MIN_Y,    // 10
  COL_MAX_X,    // 11
  COL_MAX_Y,    // 12
  COL_AREA,     // 13
  COL_WORLD_X,  // 14
  COL_WORLD_Y,  // 15
  COL_WORLD_Z,  // 16
  COL_TIME,     // 17
  COL_CONFIDENCE// 18
};

// ------------------------------------------------------------------
class detected_object_set_input_kw18::priv
{
public:
  priv( detected_object_set_input_kw18* parent)
    : m_parent( parent )
    , m_first( true )
  { }

  ~priv() { }

  void read_all();

  detected_object_set_input_kw18* m_parent;
  bool m_first;

  int m_current_idx;
  int m_last_idx;

  // Map of detected objects indexed by frame number. Each set
  // contains all detections for a single frame.
  std::map< int, kwiver::vital::detected_object_set_sptr > m_detected_sets;
};

// ==================================================================
detected_object_set_input_kw18::
detected_object_set_input_kw18()
  : d( new detected_object_set_input_kw18::priv( this ) )
{
  attach_logger( "arrows.core.detected_object_set_input_kw18" );
}

detected_object_set_input_kw18::
~detected_object_set_input_kw18()
{
}

// ------------------------------------------------------------------
void
detected_object_set_input_kw18::
set_configuration( VITAL_UNUSED vital::config_block_sptr config )
{
}

// ------------------------------------------------------------------
bool
detected_object_set_input_kw18::
check_configuration( VITAL_UNUSED vital::config_block_sptr config ) const
{
  return true;
}

// ------------------------------------------------------------------
bool
detected_object_set_input_kw18::
read_set( kwiver::vital::detected_object_set_sptr & set,
          VITAL_UNUSED std::string& image_name )
{
  if ( d->m_first )
  {
    // Read in all detections
    d->read_all();
    d->m_first = false;

    // set up iterators for returning sets.
    d->m_current_idx = d->m_detected_sets.begin()->first;
    d->m_last_idx = d->m_detected_sets.rbegin()->first;
  } // end first

  // test for end of all loaded detections
  if (d->m_current_idx > d->m_last_idx)
  {
    return false;
  }

  // return detection set at current index if there is one
  if ( 0 == d->m_detected_sets.count( d->m_current_idx ) )
  {
    // return empty set
    set = std::make_shared< kwiver::vital::detected_object_set>();
  }
  else
  {
    // Return detections for this frame.
    set = d->m_detected_sets[d->m_current_idx];
  }

  ++d->m_current_idx;

  return true;
}

// ------------------------------------------------------------------
void
detected_object_set_input_kw18::
new_stream()
{
  d->m_first = true;
}

// ==================================================================
void
detected_object_set_input_kw18::priv::
read_all()
{
  std::string line;
  kwiver::vital::data_stream_reader stream_reader( m_parent->stream() );

  m_detected_sets.clear();

  while ( stream_reader.getline( line ) )
  {
    std::vector< std::string > col;
    kwiver::vital::tokenize( line, col, " ", kwiver::vital::TokenizeTrimEmpty );

    if ( ( col.size() < 18 ) || ( col.size() > 20 ) )
    {
      std::stringstream str;
      str << "This is not a kw18 kw19 or kw20 file; found " << col.size()
          << " columns in\n\"" << line << "\"";
      VITAL_THROW( kwiver::vital::invalid_data, str.str() );
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
    int index = atoi( col[COL_FRAME].c_str() );
    if ( 0 == m_detected_sets.count( index ) )
    {
      // create a new detection set entry
      m_detected_sets[ index ] = std::make_shared<kwiver::vital::detected_object_set>();
    }

    kwiver::vital::bounding_box_d bbox(
      atof( col[COL_MIN_X].c_str() ),
      atof( col[COL_MIN_Y].c_str() ),
      atof( col[COL_MAX_X].c_str() ),
      atof( col[COL_MAX_Y].c_str() ) );

    double conf(1.0);
    if ( col.size() == 19 )
    {
      conf = atof( col[COL_CONFIDENCE].c_str() );
    }

    // Create detection
    kwiver::vital::detected_object_sptr dob = std::make_shared< kwiver::vital::detected_object>( bbox, conf );

    // Add detection to set for the frame
    m_detected_sets[index]->add( dob );
  } // ...while !eof
} // read_all

} } } // end namespace
