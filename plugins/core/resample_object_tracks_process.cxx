/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Resample object tracks from one downsample rate to another
 */

#include "resample_object_tracks_process.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/object_track_set.h>
#include <vital/types/detected_object.h>
#include <vital/types/detected_object_type.h>
#include <vital/types/bounding_box.h>
#include <vital/util/tokenize.h>

#include <sprokit/processes/kwiver_type_traits.h>
#include <sprokit/pipeline/process_exception.h>

#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

namespace kv = kwiver::vital;

namespace viame
{

namespace core
{

create_config_trait( track_file, std::string, "",
  "Path to input VIAME CSV track file" );
create_config_trait( input_rate, unsigned, "1",
  "The downsample rate the input tracks were generated at "
  "(e.g. 5 = every 5th frame)" );
create_config_trait( output_rate, unsigned, "1",
  "The desired output downsample rate "
  "(e.g. 2 = every 2nd frame)" );

// Column indices for VIAME CSV format
enum
{
  COL_DET_ID = 0,   // Track ID
  COL_SOURCE_ID,     // Video/Image Identifier
  COL_FRAME_ID,      // Frame Number
  COL_MIN_X,         // Bbox top-left X
  COL_MIN_Y,         // Bbox top-left Y
  COL_MAX_X,         // Bbox bottom-right X
  COL_MAX_Y,         // Bbox bottom-right Y
  COL_CONFIDENCE,    // Confidence
  COL_LENGTH,        // Target Length
  COL_TOT            // Start of class/score pairs
};

// Per-track state: frame id and associated detected object
struct track_entry
{
  kv::frame_id_t frame_id;
  kv::detected_object_sptr detection;
};

// =============================================================================
// Private implementation class
class resample_object_tracks_process::priv
{
public:
  priv();
  ~priv();

  void read_track_file( const std::string& filename );

  kv::detected_object_sptr interpolate(
    const track_entry& e1,
    const track_entry& e2,
    kv::frame_id_t target_frame ) const;

  // Configuration settings
  std::string m_track_file;
  unsigned m_input_rate;
  unsigned m_output_rate;

  // Loaded tracks: track_id -> sorted vector of track entries
  std::map< int, std::vector< track_entry > > m_tracks;

  kv::logger_handle_t m_logger;
};


// -----------------------------------------------------------------------------
resample_object_tracks_process::priv
::priv()
  : m_input_rate( 1 )
  , m_output_rate( 1 )
  , m_logger( kv::get_logger( "resample_object_tracks_process" ) )
{
}


resample_object_tracks_process::priv
::~priv()
{
}


// -----------------------------------------------------------------------------
void
resample_object_tracks_process::priv
::read_track_file( const std::string& filename )
{
  std::ifstream fin( filename );

  if( !fin )
  {
    std::stringstream ss;
    ss << "Unable to open track file: " << filename;
    VITAL_THROW( kv::file_not_found_exception, filename, ss.str() );
  }

  std::string line;

  while( std::getline( fin, line ) )
  {
    // Skip empty lines and comments
    if( line.empty() || line[0] == '#' )
    {
      continue;
    }

    std::vector< std::string > col;
    kv::tokenize( line, col, ",", false );

    if( col.size() < 9 )
    {
      LOG_WARN( m_logger, "Skipping malformed CSV line (fewer than 9 columns): "
                << line );
      continue;
    }

    int trk_id = atoi( col[COL_DET_ID].c_str() );
    kv::frame_id_t frame_id = atoi( col[COL_FRAME_ID].c_str() );

    kv::bounding_box_d bbox(
      atof( col[COL_MIN_X].c_str() ),
      atof( col[COL_MIN_Y].c_str() ),
      atof( col[COL_MAX_X].c_str() ),
      atof( col[COL_MAX_Y].c_str() ) );

    double conf = atof( col[COL_CONFIDENCE].c_str() );

    // Parse class/score pairs
    kv::detected_object_type_sptr dot =
      std::make_shared< kv::detected_object_type >();

    for( unsigned i = COL_TOT; i + 1 < col.size(); i += 2 )
    {
      if( col[i].empty() || col[i][0] == '(' )
      {
        break;
      }

      std::string spec_id = col[i];
      double spec_conf = atof( col[i + 1].c_str() );
      dot->set_score( spec_id, spec_conf );
    }

    kv::detected_object_sptr dob;

    if( COL_TOT < col.size() && !col[COL_TOT].empty() && col[COL_TOT][0] != '(' )
    {
      dob = std::make_shared< kv::detected_object >( bbox, conf, dot );
    }
    else
    {
      dob = std::make_shared< kv::detected_object >( bbox, conf );
    }

    track_entry entry;
    entry.frame_id = frame_id;
    entry.detection = dob;

    m_tracks[ trk_id ].push_back( entry );
  }

  // Sort each track's entries by frame id
  for( auto& pair : m_tracks )
  {
    std::sort( pair.second.begin(), pair.second.end(),
      []( const track_entry& a, const track_entry& b )
      {
        return a.frame_id < b.frame_id;
      } );
  }

  LOG_INFO( m_logger, "Loaded " << m_tracks.size() << " tracks from " << filename );
}


// -----------------------------------------------------------------------------
kv::detected_object_sptr
resample_object_tracks_process::priv
::interpolate(
  const track_entry& e1,
  const track_entry& e2,
  kv::frame_id_t target_frame ) const
{
  double range = static_cast< double >( e2.frame_id - e1.frame_id );
  double alpha = static_cast< double >( target_frame - e1.frame_id ) / range;

  const kv::bounding_box_d& b1 = e1.detection->bounding_box();
  const kv::bounding_box_d& b2 = e2.detection->bounding_box();

  kv::bounding_box_d interp_bbox(
    b1.min_x() * ( 1.0 - alpha ) + b2.min_x() * alpha,
    b1.min_y() * ( 1.0 - alpha ) + b2.min_y() * alpha,
    b1.max_x() * ( 1.0 - alpha ) + b2.max_x() * alpha,
    b1.max_y() * ( 1.0 - alpha ) + b2.max_y() * alpha );

  // Carry forward confidence and classification from earlier state
  double conf = e1.detection->confidence();

  kv::detected_object_sptr result;

  if( e1.detection->type() )
  {
    result = std::make_shared< kv::detected_object >(
      interp_bbox, conf, e1.detection->type() );
  }
  else
  {
    result = std::make_shared< kv::detected_object >( interp_bbox, conf );
  }

  return result;
}


// =============================================================================

resample_object_tracks_process
::resample_object_tracks_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new resample_object_tracks_process::priv() )
{
  make_ports();
  make_config();
}


resample_object_tracks_process
::~resample_object_tracks_process()
{
}


// -----------------------------------------------------------------------------
void
resample_object_tracks_process
::make_ports()
{
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( timestamp, required );

  // -- output --
  declare_output_port_using_trait( object_track_set, optional );
}


// -----------------------------------------------------------------------------
void
resample_object_tracks_process
::make_config()
{
  declare_config_using_trait( track_file );
  declare_config_using_trait( input_rate );
  declare_config_using_trait( output_rate );
}


// -----------------------------------------------------------------------------
void
resample_object_tracks_process
::_configure()
{
  d->m_track_file = config_value_using_trait( track_file );
  d->m_input_rate = config_value_using_trait( input_rate );
  d->m_output_rate = config_value_using_trait( output_rate );

  if( d->m_track_file.empty() )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception,
                 name(), "track_file must be specified" );
  }

  if( d->m_input_rate == 0 )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception,
                 name(), "input_rate must be greater than 0" );
  }

  if( d->m_output_rate == 0 )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception,
                 name(), "output_rate must be greater than 0" );
  }

  d->read_track_file( d->m_track_file );
}


// -----------------------------------------------------------------------------
void
resample_object_tracks_process
::_step()
{
  kv::timestamp timestamp = grab_from_port_using_trait( timestamp );

  kv::frame_id_t frame_id = timestamp.get_frame();

  std::vector< kv::track_sptr > output_tracks;

  for( const auto& track_pair : d->m_tracks )
  {
    int trk_id = track_pair.first;
    const std::vector< track_entry >& entries = track_pair.second;

    if( entries.empty() )
    {
      continue;
    }

    // Check if this frame is within the track's time span
    kv::frame_id_t first_frame = entries.front().frame_id;
    kv::frame_id_t last_frame = entries.back().frame_id;

    if( frame_id < first_frame || frame_id > last_frame )
    {
      continue;
    }

    // Binary search for the position of frame_id in the sorted entries
    auto it = std::lower_bound( entries.begin(), entries.end(), frame_id,
      []( const track_entry& e, kv::frame_id_t f )
      {
        return e.frame_id < f;
      } );

    kv::detected_object_sptr det;

    if( it != entries.end() && it->frame_id == frame_id )
    {
      // Exact match - use directly
      det = it->detection;
    }
    else if( it != entries.begin() )
    {
      // Interpolate between the two surrounding states
      auto prev = std::prev( it );

      if( it != entries.end() )
      {
        det = d->interpolate( *prev, *it, frame_id );
      }
      else
      {
        // Past the last entry but within range (shouldn't happen given
        // the range check above, but handle gracefully)
        det = prev->detection;
      }
    }

    if( det )
    {
      kv::track_sptr trk = kv::track::create();
      trk->set_id( trk_id );

      kv::track_state_sptr ots =
        std::make_shared< kv::object_track_state >(
          timestamp, det );

      trk->append( ots );
      output_tracks.push_back( trk );
    }
  }

  kv::object_track_set_sptr output =
    std::make_shared< kv::object_track_set >( output_tracks );

  push_to_port_using_trait( object_track_set, output );
}

} // end namespace core

} // end namespace viame
