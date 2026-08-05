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
#include <vital/algo/read_object_track_set.h>

#include <sprokit/processes/kwiver_type_traits.h>
#include <sprokit/pipeline/process_exception.h>

#include <algorithm>
#include <map>
#include <vector>

namespace kv = kwiver::vital;
namespace algo = kwiver::vital::algo;

namespace viame
{

namespace core
{

create_config_trait( track_file, std::string, "",
  "Path to input track file" );
create_config_trait( input_rate, unsigned, "1",
  "The downsample rate the input tracks were generated at "
  "(e.g. 5 = every 5th frame)" );
create_config_trait( output_rate, unsigned, "1",
  "The desired output downsample rate "
  "(e.g. 2 = every 2nd frame)" );

create_algorithm_name_config_trait( reader );

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

  void load_tracks();

  kv::detected_object_sptr interpolate(
    const track_entry& e1,
    const track_entry& e2,
    kv::frame_id_t target_frame ) const;

  // Configuration settings
  std::string m_track_file;
  unsigned m_input_rate;
  unsigned m_output_rate;

  // Algorithm reader
  algo::read_object_track_set_sptr m_reader;

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
::load_tracks()
{
  m_reader->open( m_track_file );

  kv::object_track_set_sptr track_set;

  while( m_reader->read_set( track_set ) )
  {
    if( !track_set )
    {
      continue;
    }

    for( auto const& trk : track_set->tracks() )
    {
      int trk_id = static_cast< int >( trk->id() );

      for( auto const& state_sptr : *trk )
      {
        auto ots = std::dynamic_pointer_cast<
          kv::object_track_state >( state_sptr );

        if( !ots || !ots->detection() )
        {
          continue;
        }

        track_entry entry;
        entry.frame_id = ots->frame();
        entry.detection = ots->detection();

        m_tracks[ trk_id ].push_back( entry );
      }
    }
  }

  m_reader->close();

  // Sort each track's entries by frame id
  for( auto& pair : m_tracks )
  {
    std::sort( pair.second.begin(), pair.second.end(),
      []( const track_entry& a, const track_entry& b )
      {
        return a.frame_id < b.frame_id;
      } );
  }

  LOG_INFO( m_logger, "Loaded " << m_tracks.size()
            << " tracks from " << m_track_file );
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
  declare_config_using_trait( reader );
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

  kv::config_block_sptr algo_config = get_config();

  if( !check_nested_algo_configuration_using_trait(
        reader, algo_config, d->m_reader ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception,
                 name(), "Reader algorithm configuration check failed." );
  }

  set_nested_algo_configuration_using_trait(
    reader, algo_config, d->m_reader );

  if( !d->m_reader )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception,
                 name(), "Unable to create track reader." );
  }

  d->load_tracks();
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
