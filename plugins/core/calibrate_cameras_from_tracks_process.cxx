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

#include <memory>
#include <arrows/ocv/camera_intrinsics.h>


#include "read_object_track_set_viame_csv.h"

namespace kv = kwiver::vital;


namespace viame {

namespace core {


create_config_trait( output_cameras_directory, std::string, "",
  "The calibrated cameras files directory" );
create_config_trait( output_json_file, std::string, "",
  "Output JSON calibration file path (camera_rig_io compatible)" );
create_config_trait( frame_count_threshold, unsigned, "0",
  "Maximum number of frames to use during calibration. 0 to use all available frames." );
create_config_trait( square_size, double, "1.0",
  "Calibration pattern square size in world units (e.g., mm)" );

create_type_trait( integer, "kwiver:integer", int64_t );
create_port_trait( tracks_left, object_track_set, "Object track set of camera1." );
create_port_trait( tracks_right, object_track_set, "Object track set of camera2." );
create_port_trait( image_width, integer, "Width of the input images." );
create_port_trait( image_height, integer, "Height of the input images." );

// Create custom camera map type to push camera_map to the output port
create_type_trait( camera_map, "kwiver:camera_map", kv::camera_map_sptr );
create_port_trait( camera_map, camera_map, "Calibrated cameras." );

// =============================================================================
// Private implementation class
class calibrate_cameras_from_tracks_process::priv
{
public:
  explicit priv( calibrate_cameras_from_tracks_process* parent );
  ~priv();

  // Configuration settings
  std::string m_output_cameras_directory;
  std::string m_output_json_file;
  unsigned m_frame_count_threshold;
  double m_square_size;

  // Other variables
  calibrate_cameras_from_tracks_process* parent;

  kv::logger_handle_t m_logger;

  static std::pair< std::string, float >
  get_attribute_value( const std::string& note );

  static kv::feature_track_set_sptr
  merge_features_track( const kv::feature_track_set_sptr& feature_track1,
                        const kv::feature_track_set_sptr& feature_track2 );

  static kv::landmark_map_sptr
  merge_landmarks_map( const kv::landmark_map_sptr& landmarks_map1,
                       const kv::landmark_map_sptr& landmarks_map2 );

  std::pair< kv::feature_track_set_sptr, kv::landmark_map_sptr >
  split_object_track( const kv::object_track_set_sptr& object_track );
};


std::pair< std::string, float >
calibrate_cameras_from_tracks_process::priv
::get_attribute_value( const std::string& note )
{
  // read a formatted notes in detection "(trk) :name=value"
  std::size_t pos = note.find_first_of( ':' );
  std::size_t pos2 = note.find_first_of( '=' );

  std::string attr_name = "";
  float value = 0.;

  if( pos == std::string::npos || pos2 == std::string::npos || pos2 == pos + 1 )
  {
    return std::make_pair( attr_name, value );
  }

  attr_name = note.substr( pos + 1, pos2 - 1 );
  value = std::stof( note.substr( pos2 + 1 ) );

  return std::make_pair( attr_name, value );
}


kv::feature_track_set_sptr
calibrate_cameras_from_tracks_process::priv
::merge_features_track( const kv::feature_track_set_sptr& feature_track1,
                        const kv::feature_track_set_sptr& feature_track2 )
{
  auto all_tracks1 = feature_track1->tracks();
  auto all_tracks2 = feature_track2->tracks();
  std::vector< kv::track_sptr > all_tracks;

  all_tracks.insert( all_tracks.end(), all_tracks1.begin(), all_tracks1.end() );
  all_tracks.insert( all_tracks.end(), all_tracks2.begin(), all_tracks2.end() );

  int i_track{};
  for( auto& track : all_tracks )
  {
    track->set_id( i_track );
    i_track++;
  }

  return std::make_shared< kv::feature_track_set >( all_tracks );
}

kv::landmark_map_sptr
calibrate_cameras_from_tracks_process::priv
::merge_landmarks_map( const kv::landmark_map_sptr& landmarks_map1,
                       const kv::landmark_map_sptr& landmarks_map2 )
{
  kv::landmark_map::map_landmark_t ldms_1 = landmarks_map1->landmarks();
  kv::landmark_map::map_landmark_t ldms_2 = landmarks_map2->landmarks();
  kv::landmark_map::map_landmark_t all_ldms;

  std::vector< kv::track_id_t > all_landmarks_ids;

  for( const auto& ldms : ldms_1 )
  {
    all_ldms[ldms.first] = ldms.second;
  }

  kv::track_id_t max_landmark1_id = ( all_ldms.rbegin() )->first;

  for( const auto& ldms : ldms_2 )
  {
    all_ldms[ldms.first + max_landmark1_id + 1] = ldms.second;
  }

  return kv::landmark_map_sptr( new kv::simple_landmark_map( all_ldms ) );
}


std::pair< kv::feature_track_set_sptr, kv::landmark_map_sptr >
calibrate_cameras_from_tracks_process::priv
::split_object_track( const kv::object_track_set_sptr& object_track )
{
  std::set< kv::track_id_t > erase_id_set;
  kv::landmark_map::map_landmark_t landmarks;
  kv::feature_track_set_sptr features;
  std::vector< kv::track_sptr > tracks;
  std::vector< kv::track_sptr > all_tracks = object_track->tracks();

  // get landmarks coordinate from detection notes and init landmark_map
  for( const auto& track : object_track->tracks() )
  {
    for( auto state : *track | kv::as_object_track )
    {
      if( state->detection()->notes().empty() )
      {
        continue;
      }

      std::map< std::string, double > attrs;

      for( const auto& note : state->detection()->notes() )
      {
        attrs.insert( get_attribute_value( note ) );
      }

      // Only keep image and world points for which xyz values has been found
      if( attrs.count( "stereo3d_x" ) &&
          attrs.count( "stereo3d_y" ) &&
          attrs.count( "stereo3d_z" ) )
      {
        kv::vector_3d pt = kv::vector_3d(
          attrs["stereo3d_x"], attrs["stereo3d_y"], attrs["stereo3d_z"] );

        if( !landmarks.count( track->id() ) )
        {
          landmarks[track->id()] = kv::landmark_sptr( new kv::landmark_d( pt ) );
        }
        // we will erase the landmark if we have different pt coordinate
        // for the same track_id
        else if( landmarks[track->id()]->loc() != pt )
        {
          erase_id_set.insert( track->id() );
        }
      }
    }
  }

  // erase invalid landmarks
  for( const auto& track_id : erase_id_set )
  {
    landmarks.erase( track_id );
  }

  // get detected target corner pt
  // only push features with valid landmarks into features set
  auto is_erased = [&erase_id_set]( size_t track_id )
  {
    return erase_id_set.find( track_id ) != std::end( erase_id_set );
  };

  for( const auto& track : all_tracks )
  {
    if( is_erased( track->id() ) )
    {
      continue;
    }

    kv::track_sptr t = kv::track::create();
    t->set_id( track->id() );
    tracks.push_back( t );

    for( auto state1 : *track | kv::as_object_track )
    {
      auto center = state1->detection()->bounding_box().center();

      auto fts = std::make_shared< kv::feature_track_state >( state1->frame() );
      fts->feature = std::make_shared< kv::feature_d >( center );
      fts->inlier = true;
      t->append( fts );
    }
  }

  features = std::make_shared< kv::feature_track_set >( tracks );

  if( features->empty() )
  {
    VITAL_THROW( kv::invalid_data, "Features empty after object track splitting" );
  }

  if( landmarks.empty() )
  {
    VITAL_THROW( kv::invalid_data, "Landmarks empty after object track splitting" );
  }

  LOG_DEBUG( m_logger,
    "Features number split from object track: " << features->size() );
  LOG_DEBUG( m_logger,
    "Landmarks number split from object track: " << landmarks.size() );

  return std::make_pair( features,
    kv::landmark_map_sptr( new kv::simple_landmark_map( landmarks ) ) );
}


// -----------------------------------------------------------------------------
calibrate_cameras_from_tracks_process::priv
::priv( calibrate_cameras_from_tracks_process* ptr )
  : m_output_cameras_directory( "" ),
    m_output_json_file( "" ),
    m_frame_count_threshold( 0 ),
    m_square_size( 1.0 ),
    parent( ptr )
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

  // -- input --
  declare_input_port_using_trait( tracks_left, required );
  declare_input_port_using_trait( tracks_right, required );
  declare_input_port_using_trait( image_width, required );
  declare_input_port_using_trait( image_height, required );

  // -- outputs --
  declare_output_port_using_trait( camera_map, optional );
}


// -----------------------------------------------------------------------------
void
calibrate_cameras_from_tracks_process
::make_config()
{
  declare_config_using_trait( output_cameras_directory );
  declare_config_using_trait( output_json_file );
  declare_config_using_trait( frame_count_threshold );
  declare_config_using_trait( square_size );
}


// -----------------------------------------------------------------------------
void
calibrate_cameras_from_tracks_process
::_configure()
{
  d->m_output_cameras_directory = config_value_using_trait( output_cameras_directory );
  d->m_output_json_file = config_value_using_trait( output_json_file );
  d->m_frame_count_threshold = config_value_using_trait( frame_count_threshold );
  d->m_square_size = config_value_using_trait( square_size );
}

// -----------------------------------------------------------------------------
void
calibrate_cameras_from_tracks_process
::_step()
{
  kv::object_track_set_sptr object_track_set1, object_track_set2;

  object_track_set1 = grab_from_port_using_trait( tracks_left );
  object_track_set2 = grab_from_port_using_trait( tracks_right );
  int64_t input_image_width = grab_from_port_using_trait( image_width );
  int64_t input_image_height = grab_from_port_using_trait( image_height );

  if( !object_track_set1 || !object_track_set2 )
  {
    return;
  }

  LOG_DEBUG( d->m_logger,
    "Received image size: " << input_image_width << "x" << input_image_height );

  auto config_optimizer = kv::config_block::empty_config();
  config_optimizer->set_value( "image_width",
    static_cast< unsigned >( input_image_width ) );
  config_optimizer->set_value( "image_height",
    static_cast< unsigned >( input_image_height ) );
  config_optimizer->set_value( "frame_count_threshold", d->m_frame_count_threshold );
  config_optimizer->set_value( "output_calibration_directory",
    d->m_output_cameras_directory );
  config_optimizer->set_value( "output_json_file", d->m_output_json_file );
  config_optimizer->set_value( "square_size", d->m_square_size );

  kv::camera_map::map_camera_t cameras;
  cameras[0] = std::make_shared< kv::simple_camera_perspective >();
  cameras[1] = std::make_shared< kv::simple_camera_perspective >();
  kv::camera_map_sptr cameras_map =
    std::make_shared< kv::simple_camera_map >( cameras );

  // split object track set then merge features and landmarks
  std::pair< kv::feature_track_set_sptr, kv::landmark_map_sptr > split1, split2;
  split1 = d->split_object_track( object_track_set1 );
  split2 = d->split_object_track( object_track_set2 );

  // Sanity check on left and right tracks number after feature / landmark split
  auto t1_size = split1.first->tracks().size();
  auto t2_size = split2.first->tracks().size();

  if( t1_size != t2_size )
  {
    std::stringstream ss;
    ss << "Track size incoherent between left and right camera: "
       << t1_size << " - " << t2_size;
    VITAL_THROW( kv::invalid_data, ss.str() );
  }

  kv::feature_track_set_sptr features =
    d->merge_features_track( split1.first, split2.first );
  kv::landmark_map_sptr landmarks =
    d->merge_landmarks_map( split1.second, split2.second );

  // get camera optimizer and compute cameras calibration
  kv::algo::optimize_cameras_sptr camera_optimizer;
  camera_optimizer = kv::algo::optimize_cameras::create( "ocv_optimize_stereo_cameras" );
  camera_optimizer->set_configuration( config_optimizer );
  camera_optimizer->optimize( cameras_map, features, landmarks );

  kv::camera_map::map_camera_t cams = cameras_map->cameras();

  // Use current directory if output directory not specified
  std::string output_dir =
    d->m_output_cameras_directory.empty() ? "." : d->m_output_cameras_directory;

  unsigned cam_id = 1;

  for( const auto& cam : cams )
  {
    kv::simple_camera_perspective_sptr camera;
    camera = std::dynamic_pointer_cast< kv::simple_camera_perspective >( cam.second );

    if( !camera )
    {
      LOG_WARN( d->m_logger, "Unable to get the camera." );
      break;
    }

    std::string out_fname1 =
      output_dir + "/camera" + std::to_string( cam_id ) + ".krtd";
    kv::write_krtd_file( *camera, out_fname1 );
    cam_id++;
  }

  push_to_port_using_trait( camera_map, cameras_map );
  mark_process_as_complete();
}


} // end namespace core

} // end namespace viame
