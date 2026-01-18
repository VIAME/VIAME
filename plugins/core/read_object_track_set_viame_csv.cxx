/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Implementation of read_object_track_set_viame_csv
 */

#include "read_object_track_set_viame_csv.h"

#include "filename_to_timestamp.h"
#include "convert_notes_to_attributes.h"
#include "convert_polygons_to_mask.h"

#include <vital/util/tokenize.h>
#include <vital/util/data_stream_reader.h>

#include <kwiversys/SystemTools.hxx>

#include <set>

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

// -------------------------------------------------------------------------------
class read_object_track_set_viame_csv::priv
{
public:
  priv( read_object_track_set_viame_csv* parent )
    : m_parent( parent )
    , m_logger( kwiver::vital::get_logger( "read_object_track_set_viame_csv" ) )
    , m_first( true )
    , m_current_idx( 0 )
    , m_last_idx( 1 )
  {}

  ~priv() {}

  typedef std::vector< kwiver::vital::track_sptr > track_vector;
  typedef int frame_id_t;

  read_object_track_set_viame_csv* m_parent;
  kwiver::vital::logger_handle_t m_logger;

  // Internal counters
  bool m_first;
  frame_id_t m_current_idx;
  frame_id_t m_last_idx;

  // Helper function - read all states
  void read_all();

  // Helper function - format tracks for current frame
  track_vector format_tracks( const track_vector& tracks, const frame_id_t frame_id );

  // Map of object tracks indexed by frame number. Each set contains all tracks
  // referenced (active) on that individual frame.
  std::map< frame_id_t, track_vector > m_tracks_by_frame_id;

  // Compilation of all loaded tracks, track id -> track sptr mapping
  std::map< frame_id_t, kwiver::vital::track_sptr > m_all_tracks;

  // Compilation of all loaded track IDs, track id -> type string
  std::map< frame_id_t, std::string > m_track_ids;
};


// ===============================================================================
void
read_object_track_set_viame_csv
::initialize()
{
  d.reset( new read_object_track_set_viame_csv::priv( this ) );
  d->m_current_idx = c_frame_id_adjustment;
}


// -------------------------------------------------------------------------------
void
read_object_track_set_viame_csv
::open( std::string const& filename )
{
  kwiver::vital::algo::read_object_track_set::open( filename );

  d->m_first = true;

  d->m_tracks_by_frame_id.clear();
  d->m_all_tracks.clear();
  d->m_track_ids.clear();
}


// -------------------------------------------------------------------------------
bool
read_object_track_set_viame_csv
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  if( c_multi_state_only && c_single_state_only )
  {
    return false;
  }
  return true;
}


// -------------------------------------------------------------------------------
bool
read_object_track_set_viame_csv
::read_set( kwiver::vital::object_track_set_sptr& set )
{
  bool was_first = d->m_first;

  if( was_first )
  {
    // Read in all detections
    d->read_all();
    d->m_first = false;
  }

  if( c_batch_load )
  {
    if( !was_first )
    {
      return false;
    }
    std::vector< kwiver::vital::track_sptr > trks;

    for( auto it = d->m_all_tracks.begin(); it != d->m_all_tracks.end(); ++it )
    {
      if( c_single_state_only && it->second->size() > 1 )
      {
        continue;
      }
      if( c_multi_state_only && it->second->size() == 1 )
      {
        continue;
      }
      trks.push_back( it->second );
    }

    set = kwiver::vital::object_track_set_sptr(
      new kwiver::vital::object_track_set( trks ) );

    return true;
  }

  // Return detection set at current index if there is one
  if( d->m_tracks_by_frame_id.count( d->m_current_idx ) == 0 )
  {
    // Return empty set
    set = std::make_shared< kwiver::vital::object_track_set>();
  }
  else
  {
    // Return tracks for this frame
    kwiver::vital::object_track_set_sptr new_set(
      new kwiver::vital::object_track_set(
        d->format_tracks( 
          d->m_tracks_by_frame_id[ d->m_current_idx ],
          d->m_current_idx ) ) );

    set = new_set;
  }

  ++d->m_current_idx;
  return true;
}


// -------------------------------------------------------------------------------
void
read_object_track_set_viame_csv::priv
::read_all()
{
  std::string line;
  kwiver::vital::data_stream_reader stream_reader( m_parent->stream() );

  m_tracks_by_frame_id.clear();
  m_all_tracks.clear();
  m_track_ids.clear();

  if( m_parent->c_single_state_only || m_parent->c_multi_state_only )
  {
    m_tracks_by_frame_id.clear();
    m_all_tracks.clear();
    m_track_ids.clear();
  }

  // Read track file
  while( stream_reader.getline( line ) )
  {
    if( !line.empty() && line[0] == '#' )
    {
      continue;
    }

    std::vector< std::string > col;
    kwiver::vital::tokenize( line, col, m_parent->c_delimiter, false );

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
    int trk_id = atoi( col[COL_DET_ID].c_str() );
    frame_id_t frame_id = atoi( col[COL_FRAME_ID].c_str() );
    frame_id = frame_id + m_parent->c_frame_id_adjustment;
    kwiver::vital::time_usec_t frame_time;
    std::string str_id = col[COL_SOURCE_ID];

    kwiver::vital::bounding_box_d bbox(
      atof( col[COL_MIN_X].c_str() ),
      atof( col[COL_MIN_Y].c_str() ),
      atof( col[COL_MAX_X].c_str() ),
      atof( col[COL_MAX_Y].c_str() ) );

    double conf = atof( col[COL_CONFIDENCE].c_str() );

    if( m_parent->c_confidence_override > 0.0 )
    {
      conf = m_parent->c_confidence_override;
    }

    // Create detection object
    kwiver::vital::detected_object_sptr dob;

    kwiver::vital::detected_object_type_sptr dot =
      std::make_shared<kwiver::vital::detected_object_type>();

    bool found_attribute = false;

    for( unsigned i = COL_TOT; i < col.size(); i+=2 )
    {
      if( col[i].empty() || col[i][0] == '(' )
      {
        found_attribute = true;
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

    try
    {
      frame_time = convert_to_timestamp( str_id );
    }
    catch( ... )
    {
      frame_time = frame_id;
    }

    std::vector< std::string > poly_strings;

    if( found_attribute )
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
    if( m_parent->c_poly_to_mask && found_attribute )
    {
      kwiver::vital::image_of< uint8_t > mask_data;

      convert_polys_to_mask( poly_strings, bbox, mask_data );

      kwiver::vital::image_container_scptr computed_mask =
        std::make_shared< kwiver::vital::simple_image_container >( mask_data );

      dob->set_mask( computed_mask );
    }
#endif

    if( found_attribute )
    {
      add_attributes_to_detection( *dob, col );
    }

    // Create new object track state
    kwiver::vital::track_state_sptr ots =
      std::make_shared< kwiver::vital::object_track_state >(
        frame_id, frame_time, dob );

    // Assign object track state to track
    kwiver::vital::track_sptr trk;

    if( m_all_tracks.count( trk_id ) == 0 )
    {
      trk = kwiver::vital::track::create();
      trk->set_id( trk_id );
      m_all_tracks[ trk_id ] = trk;
    }
    else
    {
      trk = m_all_tracks[ trk_id ];
    }

    trk->append( ots );

    // Add track to indexes
    if( !m_parent->c_batch_load )
    {
      m_tracks_by_frame_id[ frame_id ].push_back( trk );
      m_last_idx = std::max( m_last_idx, frame_id );
    }
  }
}

read_object_track_set_viame_csv::priv::track_vector
read_object_track_set_viame_csv::priv
::format_tracks( const track_vector& tracks, const frame_id_t frame_id )
{
  if( m_parent->c_single_state_only || m_parent->c_multi_state_only )
  {
    track_vector output;

    for( auto trk : tracks )
    {
      if( m_parent->c_single_state_only && trk->size() > 1 )
      {
        continue;
      }
      if( m_parent->c_multi_state_only && trk->size() == 1 )
      {
        continue;
      }

      output.push_back( trk );
    }

    return output;
  }

  return tracks;
}

} // end namespace
