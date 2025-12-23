/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Split an object track set into a feature_track_set and a landmark_map
 */

#include "split_tracks_to_feature_landmarks_process.h"

#include <vital/vital_types.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/object_track_set.h>
#include <vital/types/feature_track_set.h>
#include <vital/types/landmark_map.h>

#include <sprokit/processes/kwiver_type_traits.h>

#include "convert_notes_to_attributes.h"


namespace kv = kwiver::vital;


create_type_trait( landmark_map, "kwiver:landmark_map", kwiver::vital::landmark_map_sptr );
create_port_trait( landmark_map, landmark_map, "Landmarks.");

namespace viame
{

namespace core
{

// =============================================================================
// Private implementation class
class split_tracks_to_feature_landmarks_process::priv
{
public:
  explicit priv( split_tracks_to_feature_landmarks_process* parent );
  ~priv();

  // Other variables
  split_tracks_to_feature_landmarks_process* parent;
  
  std::pair<std::string, float>
  get_attribut_value(const std::string& note)
  {
    // read formated notes in detection "(trk) :name=value"
    std::size_t pos = note.find_first_of( ':' );
    std::size_t pos2 = note.find_first_of( '=' );
    
    std::string attr_name = "";
    float value = 0.;
    if( pos == std::string::npos || pos2 == std::string::npos || pos2 == pos + 1 )
    {
      return std::make_pair( attr_name, value );
    }
    
    attr_name = note.substr( pos+1, pos2-1 );
    value = std::stof( note.substr( pos2 + 1 ) );
    
    return std::make_pair( attr_name, value );
  }
};


// -----------------------------------------------------------------------------
split_tracks_to_feature_landmarks_process::priv
::priv( split_tracks_to_feature_landmarks_process* ptr )
  : parent( ptr )
{
}


split_tracks_to_feature_landmarks_process::priv
::~priv()
{
}


// =============================================================================
split_tracks_to_feature_landmarks_process
::split_tracks_to_feature_landmarks_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new split_tracks_to_feature_landmarks_process::priv( this ) )
{
  make_ports();
  make_config();
}


split_tracks_to_feature_landmarks_process
::~split_tracks_to_feature_landmarks_process()
{
}

// -----------------------------------------------------------------------------
void
split_tracks_to_feature_landmarks_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- inputs --
  declare_input_port_using_trait( object_track_set, required );
  
  // -- outputs --
  declare_output_port_using_trait( feature_track_set, optional );
  declare_output_port_using_trait( landmark_map, optional);
}

// -----------------------------------------------------------------------------
void
split_tracks_to_feature_landmarks_process
::make_config()
{
}

// -----------------------------------------------------------------------------
void
split_tracks_to_feature_landmarks_process
::_configure()
{
}

// -----------------------------------------------------------------------------
void
split_tracks_to_feature_landmarks_process
::_step()
{
  kv::object_track_set_sptr object_track;
  kv::feature_track_set_sptr features;
  kv::landmark_map::map_landmark_t landmarks;

  object_track = grab_from_port_using_trait( object_track_set );
  std::vector< kv::track_sptr > all_tracks = object_track->tracks();
  features = std::make_shared< kv::feature_track_set >( all_tracks );

  for( auto track : all_tracks )
  {
    for( auto state : *track | kv::as_object_track )
    {
      if( !state->detection()->notes().size() )
        break;

      // get xyz attributes from detection notes
      std::map< std::string, double > attrs;
      for( auto note : state->detection()->notes() )
      {
        attrs.insert( d->get_attribut_value( note ) );
      }

      // Only keep image and world points for which xyz values has been found in notes
      if( attrs.count( "x" ) && attrs.count( "y" ) && attrs.count( "z" ) )
      {
        kv::vector_3d pt = kv::vector_3d( attrs["x"], attrs["y"], attrs["z"] );
        kv::landmark_sptr landmark = kv::landmark_sptr( new kv::landmark_d( pt ) );
        landmarks[track->id()] = landmark;
      }
    }
  }

  push_to_port_using_trait( feature_track_set, features );
  push_to_port_using_trait( landmark_map, kv::landmark_map_sptr( new kv::simple_landmark_map( landmarks ) ) );
}

} // end namespace core

} // end namespace viame
