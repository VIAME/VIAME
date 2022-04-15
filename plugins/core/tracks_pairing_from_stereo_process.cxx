/**
 * \file
 * \brief Compute object tracks pair from stereo depth map information
 */

#include "tracks_pairing_from_stereo_process.h"

#include <vital/vital_types.h>
#include <vital/types/image_container.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/object_track_set.h>
#include <vital/types/camera_perspective_map.h>
#include <vital/io/camera_io.h>
#include <vital/algo/image_io.h>
#include <vital/types/bounding_box.h>

#include <Eigen/Core>

#include <arrows/mvg/triangulate.h>

#include <sprokit/processes/kwiver_type_traits.h>


namespace kv = kwiver::vital;

namespace viame
{

namespace core
{

create_config_trait( cameras_directory, std::string, "", 
                     "The calibrated cameras files directory" );

create_port_trait( object_track_set1, object_track_set, "Set of object tracks1." );
create_port_trait( object_track_set2, object_track_set, "Set of object tracks2." );
create_port_trait( filtered_object_track_set1, object_track_set, "The stereo filtered object tracks1." );
create_port_trait( filtered_object_track_set2, object_track_set, "The stereo filtered object tracks2." );


// =============================================================================
// Private implementation class
class tracks_pairing_from_stereo_process::priv
{
public:
  explicit priv( tracks_pairing_from_stereo_process* parent );
  ~priv();

  // Configuration settings
  std::string m_cameras_directory;
  kv::camera_map::map_camera_t m_cameras;

  kv::logger_handle_t m_logger;  
  
  // Other variables
  tracks_pairing_from_stereo_process* parent;
  
  kv::camera_map::map_camera_t
  load_camera_map(std::string const& camera1_name,
                  std::string const& camera2_name,
                  std::string const& cameras_dir)
  {
    kv::camera_map::map_camera_t cameras;

    try
    {
      cameras[0] = kv::read_krtd_file( camera1_name, cameras_dir );
      cameras[1] = kv::read_krtd_file( camera2_name, cameras_dir );
    }
    catch ( const kv::file_not_found_exception& )
    {
      VITAL_THROW( kv::invalid_data, "Calibration file not found" );
    }
    
    if ( cameras.empty() )
    {
      VITAL_THROW( kv::invalid_data, "No krtd files found" );
    }

    return cameras;
  }
  
  int
  nearest_index(int max, double value)
  {
    if (abs(value - 0.0) < 1e-6)
    {
      return 0;
    }
    else if (abs(value - (double)max) < 1e-6)
    {
      return max - 1;
    }
    else
    {
      return (int)round(value - 0.5);
    }
  }
  
  bool
  check_stereo_depth_consistency
  (kv::camera_map::map_camera_t cameras,
   kv::image_container_sptr depth_map,
   kv::bounding_box_d bbox1, 
   kv::bounding_box_d bbox2)
  {
    if(cameras.size() != 2)
    {
      LOG_WARN(m_logger, "Only works with two cameras as inputs.");
      return false;
    }
    
    kv::simple_camera_perspective_sptr cam1, cam2;
    cam1 = std::dynamic_pointer_cast<kv::simple_camera_perspective>(cameras[0]);
    cam2 = std::dynamic_pointer_cast<kv::simple_camera_perspective>(cameras[1]);
    
    kv::vector_2d const img_pt;
    kv::vector_2d npt_ = cam1->intrinsics()->unmap(img_pt);
    auto npt = kv::vector_3d(npt_(0), npt_(1), 1.0);
  
    kv::matrix_3x3d M = cam1->rotation().matrix().transpose();
    kv::vector_3d cam_pos = cam1->center();
  
    kv::vector_3d Mp = M * npt;
  
    kv::image dm_data = depth_map->get_image();
    auto dm_width = (int)dm_data.width();
    auto dm_height = (int)dm_data.height();
  
    int img_pt_x = nearest_index(dm_width, img_pt(0));
    int img_pt_y = nearest_index(dm_height, img_pt(1));
  
    if (img_pt_x < 0 || img_pt_y < 0 ||
        img_pt_x >= dm_width ||
        img_pt_y >= dm_height)
    {
      throw std::invalid_argument("Provided image point is outside of image "
                                  "bounds");
    }
  
    float depth = dm_data.at<float>(img_pt_x, img_pt_y);
  
    kv::vector_3d world_pos = cam_pos + (Mp * depth);
  
    return true;
  }
};


// -----------------------------------------------------------------------------
tracks_pairing_from_stereo_process::priv
::priv( tracks_pairing_from_stereo_process* ptr )
  : m_cameras_directory( 0 )
  , parent( ptr )
{
}


tracks_pairing_from_stereo_process::priv
::~priv()
{
}


// =============================================================================
tracks_pairing_from_stereo_process
::tracks_pairing_from_stereo_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new tracks_pairing_from_stereo_process::priv( this ) )
{
  make_ports();
  make_config();
}


tracks_pairing_from_stereo_process
::~tracks_pairing_from_stereo_process()
{
}


// -----------------------------------------------------------------------------
void
tracks_pairing_from_stereo_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- inputs --
  declare_input_port_using_trait( image, optional );
  declare_input_port_using_trait( timestamp, optional );
  declare_input_port_using_trait( object_track_set1, required );
  declare_input_port_using_trait( object_track_set2, required );
  declare_input_port_using_trait( depth_map, required );

  // -- outputs --
  declare_output_port_using_trait( timestamp, optional );
  declare_output_port_using_trait( object_track_set, optional );
  declare_input_port_using_trait( filtered_object_track_set1, required );
  declare_input_port_using_trait( filtered_object_track_set2, required );
}

// -----------------------------------------------------------------------------
void
tracks_pairing_from_stereo_process
::make_config()
{
  declare_config_using_trait( cameras_directory );
}

// -----------------------------------------------------------------------------
void
tracks_pairing_from_stereo_process
::_configure()
{
  d->m_cameras_directory = config_value_using_trait( cameras_directory );
  d->m_cameras = d->load_camera_map("camera1", "camera2", d->m_cameras_directory);  
}

// -----------------------------------------------------------------------------
void
tracks_pairing_from_stereo_process
::_step()
{
  if(d->m_cameras.size() != 2)
  {
    LOG_WARN(d->m_logger, "Only works with two cameras as inputs.");
    return;
  }
  
  kv::simple_camera_perspective_sptr cam1, cam2;
  cam1 = std::dynamic_pointer_cast<kv::simple_camera_perspective>(d->m_cameras[0]);
  cam2 = std::dynamic_pointer_cast<kv::simple_camera_perspective>(d->m_cameras[1]);
  
  kv::object_track_set_sptr input_tracks1, input_tracks2;
  kv::image_container_sptr depth_map;
  kv::timestamp timestamp;

  input_tracks1 = grab_from_port_using_trait( object_track_set1 );
  input_tracks2 = grab_from_port_using_trait( object_track_set2 );

  if( has_input_port_edge_using_trait( timestamp ) )
  {
    timestamp = grab_from_port_using_trait( timestamp );
  }
  if( has_input_port_edge_using_trait( depth_map ) )
  {
    depth_map = grab_from_port_using_trait( depth_map );
  }

  std::vector< kv::track_sptr > filtered_tracks1, filtered_tracks2;
  
  for(auto track1 : input_tracks1->tracks())
  {
    for(auto state1 : *track1 | kv::as_object_track)
    {
      for(auto track2 : input_tracks2->tracks())
      {
        for(auto state2 : *track2 | kv::as_object_track)
        {
          // Only check depth consistency for track at the same frame id
          if(state1->frame() != state2->frame())
            continue;
          
          kv::bounding_box_d bbox1 = state1->detection()->bounding_box();
          kv::bounding_box_d bbox2 = state2->detection()->bounding_box();
          
          //TODO  Rectify bbox center TODO///
          
          if(d->check_stereo_depth_consistency
             (d->m_cameras, depth_map, bbox1, bbox2))
          {
//            kv::vector_3d pt3d = 
            kv::vector_3d pt3d(0,0,0);
//            =
//                kwiver::arrows::mvg::triangulate_fast_two_view( *cam1, 
//                                                                *cam2, 
//                                                                bbox1.center(), 
//                                                                bbox2.center());
            
            // Add 3d estimations
            state1->detection()->add_note(":x=" + std::to_string( pt3d[0] ));
            state1->detection()->add_note(":y=" + std::to_string( pt3d[1] ));
            state1->detection()->add_note(":z=" + std::to_string( pt3d[2] ));
            
            state2->detection()->add_note(":x=" + std::to_string( pt3d[0] ));
            state2->detection()->add_note(":y=" + std::to_string( pt3d[1] ));
            state2->detection()->add_note(":z=" + std::to_string( pt3d[2] ));
            
            filtered_tracks1.push_back( track1 );
            filtered_tracks2.push_back( track2 );
          }
        }
      }
    }
  }
  
  kv::object_track_set_sptr output1(
    new kv::object_track_set( filtered_tracks1 ) );
  kv::object_track_set_sptr output2(
    new kv::object_track_set( filtered_tracks2 ) );

  push_to_port_using_trait( timestamp, timestamp );
  push_to_port_using_trait( filtered_object_track_set1, output1 );
  push_to_port_using_trait( filtered_object_track_set2, output2 );
}

} // end namespace core

} // end namespace viame
