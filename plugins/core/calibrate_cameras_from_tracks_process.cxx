/**
 * \file
 * \brief Calibrate two cameras from two objects track set
 */

#include "calibrate_cameras_from_tracks_process.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/object_track_set.h>
#include <vital/types/feature_track_set.h>
#include <vital/types/landmark_map.h>
#include <vital/types/camera_perspective_map.h>
#include <vital/types/camera_intrinsics.h>
#include <vital/algo/resection_camera.h>
#include <vital/algo/optimize_cameras.h>
#include <vital/range/transform.h>
#include <vital/range/iota.h>
#include <vital/io/camera_io.h>
#include <vital/exceptions.h>

#include <sprokit/processes/kwiver_type_traits.h>
#include <arrows/ocv/camera_intrinsics.h>


#include "read_object_track_set_viame_csv.h"

namespace kv = kwiver::vital;


namespace viame
{

namespace core
{


create_config_trait( output_cameras_directory, std::string, "",
                     "The calibrated cameras files directory" );
create_config_trait( tracks_left, std::string, "",
                     "Object track set of camera1.");
create_config_trait( tracks_right, std::string, "",
                     "Object track set of camera2.");
create_config_trait( image_width, unsigned, "",
                     "Camera image width in pixels.");
create_config_trait( image_height, unsigned, "",
                     "Camera image height in pixels.");

// Create custom camera map type to push camera_map to the output port
create_type_trait( camera_map, "kwiver:camera_map", kv::camera_map_sptr );
create_port_trait( camera_map, camera_map, "Calibrated cameras.");

// =============================================================================
// Private implementation class
class calibrate_cameras_from_tracks_process::priv
{
public:
  explicit priv( calibrate_cameras_from_tracks_process* parent );
  ~priv();

  // Configuration settings
  std::string m_output_cameras_directory;
  std::string m_track_set_left;
  std::string m_track_set_right;
  unsigned m_image_width;
  unsigned m_image_height;
  
  // Other variables
  calibrate_cameras_from_tracks_process* parent;
  
  kv::logger_handle_t m_logger;
  
  std::pair<std::string, float>
  get_attribut_value(const std::string& note);
  
  kv::feature_track_set_sptr
  merge_features_track(kv::feature_track_set_sptr feature_track1,
                       kv::feature_track_set_sptr feature_track2);
  
  kv::landmark_map_sptr
  merge_landmarks_map( kv::landmark_map_sptr landmarks_map1,
                       kv::landmark_map_sptr landmarks_map2);
  
  std::pair<kv::feature_track_set_sptr, kv::landmark_map_sptr>
  split_object_track(kv::object_track_set_sptr object_track);
};


std::pair<std::string, float>
calibrate_cameras_from_tracks_process::priv
::get_attribut_value(const std::string& note)
{
  // read a formated notes in detection "(trk) :name=value"
  std::size_t pos = note.find_first_of( ':' );
  std::size_t pos2 = note.find_first_of( '=' );
  
  std::string attr_name = "";
  float value = 0.;
  if( pos == std::string::npos || pos2 == std::string::npos || pos2 == pos + 1 )
  {
    return std::make_pair(attr_name, value);
  }
  
  attr_name = note.substr( pos+1, pos2-1 );
  value = std::stof(note.substr( pos2 + 1 ));
  
  return std::make_pair(attr_name, value);
}


kv::feature_track_set_sptr
calibrate_cameras_from_tracks_process::priv
::merge_features_track(kv::feature_track_set_sptr feature_track1,
                     kv::feature_track_set_sptr feature_track2)
{
  std::vector<kv::track_id_t> all_track_ids;
  std::vector<kv::track_sptr> all_tracks1 = feature_track1->tracks();
  std::vector<kv::track_sptr> all_tracks2 = feature_track2->tracks();
  for (auto track : all_tracks1 )
    all_track_ids.push_back(track->id());
  
  kv::track_id_t max_track1_id = *std::max_element(all_track_ids.begin(), all_track_ids.end());
  for (auto track : all_tracks2 )
    all_track_ids.push_back(track->id() + max_track1_id + 1);
  
  std::vector<kv::track_sptr> all_tracks;
  all_tracks.insert( all_tracks.end(), all_tracks1.begin(), all_tracks1.end() );
  all_tracks.insert( all_tracks.end(), all_tracks2.begin(), all_tracks2.end() );
  
  std::vector<kv::track_sptr> tracks;
  for (unsigned i = 0 ; i < all_tracks.size() ; i++ )
  {
    kv::track_sptr t = kv::track::create();
    t->set_id( all_track_ids[i] );
    tracks.push_back( t );
    
    for ( auto state : *all_tracks[i] | kv::as_feature_track )
    {
      auto fts = std::make_shared<kv::feature_track_state>(state->frame());
      fts->feature = std::make_shared<kv::feature_d>( state->feature->loc() );
      fts->inlier = true;
      t->append( fts );
    }
  }
  
  return std::make_shared<kv::feature_track_set>( tracks );
}

kv::landmark_map_sptr
calibrate_cameras_from_tracks_process::priv
::merge_landmarks_map( kv::landmark_map_sptr landmarks_map1,
                     kv::landmark_map_sptr landmarks_map2)
{
  kv::landmark_map::map_landmark_t ldms_1 = landmarks_map1->landmarks();
  kv::landmark_map::map_landmark_t ldms_2 = landmarks_map2->landmarks();
  kv::landmark_map::map_landmark_t all_ldms;
  
  std::vector<kv::track_id_t> all_landmarks_ids;
  for(auto ldms: ldms_1)
    all_ldms[ldms.first] = ldms.second;
  
  kv::track_id_t max_landmark1_id = (all_ldms.rbegin())->first;
  
  for(auto ldms: ldms_2)
    all_ldms[ldms.first + max_landmark1_id+1] = ldms.second;
  
  return kv::landmark_map_sptr(new kv::simple_landmark_map( all_ldms ));
}


std::pair<kv::feature_track_set_sptr, kv::landmark_map_sptr>
calibrate_cameras_from_tracks_process::priv
::split_object_track(kv::object_track_set_sptr object_track)
{
  std::map<kv::track_id_t, bool> erase_id_list;
  kv::landmark_map::map_landmark_t landmarks;
  kv::feature_track_set_sptr features;
  std::vector<kv::track_sptr> tracks;
  std::vector<kv::track_sptr> all_tracks = object_track->tracks();
  
  // get landmarks coordinate from detection notes and init landmark_map
  for(auto track : all_tracks)
  {
    for(auto state : *track | kv::as_object_track)
    {
      if(!state->detection()->notes().size())
        continue;
      
      std::map<std::string, double> attrs;
      for(auto note : state->detection()->notes())
      {
        attrs.insert(get_attribut_value(note));
      }
       
      // Only keep image and world points for which xyz values has been found in notes
      if(attrs.count("x") && attrs.count("y") && attrs.count("z"))
      {
        kv::vector_3d pt = kv::vector_3d(attrs["x"], attrs["y"], attrs["z"]);
        if(!landmarks.count(track->id()))
        {
           kv::landmark_sptr landmark = kv::landmark_sptr(new kv::landmark_d( pt ));
           landmarks[track->id()] = landmark;
        }
        // we will erase the landmark if we have different pt coordinate for the same track_id
        else if(landmarks[track->id()]->loc() != pt)
        {
          erase_id_list[track->id()] = true;
        }
      }
    }
  }
  
  // erase invalid landmarks
  for(const auto& elem : erase_id_list)
  {
    landmarks.erase(elem.first);
  }
  
  // get detected target corner pt
  // only push features with valid landmarks into features set
  for (auto track : all_tracks )
  {
    if(!erase_id_list[track->id()])
    {
      kv::track_sptr t = kv::track::create();
      t->set_id( track->id() );
      tracks.push_back( t );
      
      for ( auto state : *track | kv::as_object_track )
      {
        auto center = state->detection()->bounding_box().center();
        
        auto fts = std::make_shared<kv::feature_track_state>(state->frame());
        fts->feature = std::make_shared<kv::feature_d>( center );
        fts->inlier = true;
        t->append( fts );
      }
    }
  }
  features = std::make_shared<kv::feature_track_set>( tracks );
  
  if(!features->size())
  {
    VITAL_THROW( kv::invalid_data, "Features empty after object track splitting" );
  }
  
  if(!landmarks.size())
  {
    VITAL_THROW( kv::invalid_data, "Landmarks empty after object track splitting" );
  }
  
  LOG_DEBUG(m_logger, "Features number splitted from object track : " << features->size());
  LOG_DEBUG(m_logger, "Landmarks number splitted from object track : " << features->size());
  
  return std::make_pair(features, kv::landmark_map_sptr(new kv::simple_landmark_map( landmarks )));
}




// -----------------------------------------------------------------------------
calibrate_cameras_from_tracks_process::priv
::priv( calibrate_cameras_from_tracks_process* ptr )
  : m_output_cameras_directory( "" )
  , m_track_set_left( "" )
  , m_track_set_right( "" )
  , parent( ptr )
{
}


calibrate_cameras_from_tracks_process::priv
::~priv()
{
}


// =============================================================================
calibrate_cameras_from_tracks_process
::calibrate_cameras_from_tracks_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new calibrate_cameras_from_tracks_process::priv( this ) )
{
  make_ports();
  make_config();
  
  d->m_logger = logger();  
}


calibrate_cameras_from_tracks_process
::~calibrate_cameras_from_tracks_process()
{
}


// -----------------------------------------------------------------------------
void
calibrate_cameras_from_tracks_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- outputs --
  declare_output_port_using_trait(camera_map, optional);
}


// -----------------------------------------------------------------------------
void
calibrate_cameras_from_tracks_process
::make_config()
{
  declare_config_using_trait( output_cameras_directory );
  declare_config_using_trait( tracks_left );
  declare_config_using_trait( tracks_right );  
  declare_config_using_trait( image_width );  
  declare_config_using_trait( image_height );
}


// -----------------------------------------------------------------------------
void
calibrate_cameras_from_tracks_process
::_configure()
{
  d->m_output_cameras_directory = config_value_using_trait( output_cameras_directory );
  d->m_track_set_left = config_value_using_trait( tracks_left );
  d->m_track_set_right = config_value_using_trait( tracks_right );
  d->m_image_width = config_value_using_trait( image_width );
  d->m_image_height = config_value_using_trait( image_height );
  
  if(d->m_track_set_left == "")
    VITAL_THROW( kv::invalid_data, "Track_set_left path missing" );
  
  if(d->m_track_set_right == "")
    VITAL_THROW( kv::invalid_data, "Track_set_right path missing" );
}

// -----------------------------------------------------------------------------
void
calibrate_cameras_from_tracks_process
::_step()
{
  auto config_optimizer = kv::config_block::empty_config();
  config_optimizer->set_value( "image_width", d->m_image_width );
  config_optimizer->set_value( "image_height", d->m_image_height );
  
  kv::camera_map::map_camera_t cameras;
  kv::camera_map_sptr cameras_map;
  cameras[0] = kv::simple_camera_perspective_sptr(new kv::simple_camera_perspective());
  cameras[1] = kv::simple_camera_perspective_sptr(new kv::simple_camera_perspective());
  cameras_map = kv::camera_map_sptr( new kv::simple_camera_map( cameras ) );
  
  // load all tracks
  kv::object_track_set_sptr object_track_set1, object_track_set2;
  auto config_reader = kv::config_block::empty_config();
  config_reader->set_value( "batch_load", true );
  
  kv::algo::read_object_track_set_sptr reader;
  reader = kv::algo::read_object_track_set::create("viame_csv");
  reader->set_configuration(config_reader);
  reader->open(d->m_track_set_left);
  reader->read_set(object_track_set1);
  reader->close();

  reader->open(d->m_track_set_right);
  reader->read_set(object_track_set2);
  reader->close();
  
  // split object track set then merge features and landmarks
  std::pair<kv::feature_track_set_sptr, kv::landmark_map_sptr> split1, split2;
  split1 = d->split_object_track(object_track_set1);
  split2 = d->split_object_track(object_track_set2);
  
  kv::feature_track_set_sptr features = d->merge_features_track(split1.first, split2.first);
  kv::landmark_map_sptr landmarks = d->merge_landmarks_map(split1.second, split2.second);
  
  // get camera optimizer and compute cameras calibration
  kv::algo::optimize_cameras_sptr camera_optimizer;
  camera_optimizer = kv::algo::optimize_cameras::create("ocv_optimize_stereo_cameras");
  camera_optimizer->set_configuration(config_optimizer);
  camera_optimizer->optimize(cameras_map, features, landmarks);
  
  kv::camera_map::map_camera_t cams = cameras_map->cameras();
  unsigned cam_id = 1;
  for(auto cam: cams)
  {
    kv::simple_camera_perspective_sptr camera;
    camera = std::dynamic_pointer_cast<kv::simple_camera_perspective>(cam.second);
    if(!camera)
    {
      LOG_WARN(d->m_logger, "Unable to get the camera.");
      break;
    }

    std::string out_fname1 = d->m_output_cameras_directory + "/camera" + std::to_string(cam_id) + ".krtd";
    kv::write_krtd_file( *camera, out_fname1 );
    cam_id++;
  }
  
  push_to_port_using_trait( camera_map, cameras_map );
  mark_process_as_complete();
}


} // end namespace core

} // end namespace viame
