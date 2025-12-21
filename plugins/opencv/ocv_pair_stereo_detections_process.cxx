/**
 * \file
 * \brief Compute object detections pair from stereo depth map information
 */

#include "ocv_pair_stereo_detections_process.h"
#include "ocv_pair_stereo_detections.h"

#include <vital/vital_types.h>
#include <vital/types/detected_object_set.h>

#include <memory>

#include <arrows/ocv/image_container.h>
#include <sprokit/processes/kwiver_type_traits.h>

namespace kv = kwiver::vital;

namespace viame {

create_config_trait( cameras_directory, std::string, "",
  "The calibrated cameras files directory" )
create_config_trait( pairing_method, std::string, "PAIRING_3D",
  "One of PAIRING_3D, PAIRING_IOU, PAIRING_RECTIFIED_IOU" )
create_config_trait( iou_pair_threshold, double, "0.1",
  "Used with IOU pairing_method. Minimum IOU threshold below which left and right "
  "detections will not be paired." )
create_config_trait( verbose, bool, "false",
  "If true, will print debug information to the pipeline console." )
create_port_trait( detected_object_set1, detected_object_set, "Set of object detections1." )
create_port_trait( detected_object_set2, detected_object_set, "Set of object detections2." )
create_port_trait( detected_object_set_out1, detected_object_set,
  "The stereo filtered object detections1." )
create_port_trait( detected_object_set_out2, detected_object_set,
  "The stereo filtered object detections2." )

// =============================================================================
ocv_pair_stereo_detections_process
::ocv_pair_stereo_detections_process( kv::config_block_sptr const& config )
    : process( config ), d( new ocv_pair_stereo_detections() )
{
  make_ports();
  make_config();
}


ocv_pair_stereo_detections_process
::~ocv_pair_stereo_detections_process() = default;


// -----------------------------------------------------------------------------
void
ocv_pair_stereo_detections_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- inputs --
  declare_input_port_using_trait( depth_map, required );
  declare_input_port_using_trait( detected_object_set1, required );
  declare_input_port_using_trait( detected_object_set2, required );

  // -- outputs --
  declare_output_port_using_trait( detected_object_set_out1, optional );
  declare_output_port_using_trait( detected_object_set_out2, optional );
}

// -----------------------------------------------------------------------------
void
ocv_pair_stereo_detections_process
::make_config()
{
  declare_config_using_trait( cameras_directory );
  declare_config_using_trait( pairing_method );
  declare_config_using_trait( iou_pair_threshold );
  declare_config_using_trait( verbose );
}

// -----------------------------------------------------------------------------
void
ocv_pair_stereo_detections_process
::_configure()
{
  d->m_cameras_directory = config_value_using_trait( cameras_directory );
  d->m_pairing_method = config_value_using_trait( pairing_method );
  d->m_iou_pair_threshold = config_value_using_trait( iou_pair_threshold );
  d->m_verbose = config_value_using_trait( verbose );
  d->load_camera_calibration();
}

// -----------------------------------------------------------------------------
void
ocv_pair_stereo_detections_process
::_step()
{
  // Grab inputs from previous process
  auto left_detected_object_set = grab_from_port_using_trait( detected_object_set1 );
  auto right_detected_object_set = grab_from_port_using_trait( detected_object_set2 );
  auto depth_map = grab_from_port_using_trait( depth_map );

  // Split input disparity into left / right disparity maps
  auto cv_disparity_left = kwiver::arrows::ocv::image_container::vital_to_ocv(
    depth_map->get_image(),
    kwiver::arrows::ocv::image_container::BGR_COLOR );

  // Format detection sets as detection object vectors
  std::vector< kwiver::vital::detected_object_sptr > left_detections, right_detections;
  for( const auto& left_detection : *left_detected_object_set )
  {
    left_detections.emplace_back( left_detection );
  }

  for( const auto& right_detection : *right_detected_object_set )
  {
    right_detections.emplace_back( right_detection );
  }

  // Estimate 3D positions in left image with disparity
  auto left_3d_pos = d->update_left_detections_3d_positions( left_detections, cv_disparity_left );

  // Pair right and left tracks
  auto pairings = d->pair_left_right_detections( left_detections, left_3d_pos, right_detections );

  // Modify right_detection pairing id
  for( const auto& pairing : pairings )
  {
    right_detections[pairing[1]]->set_index( left_detections[pairing[0]]->index() );
  }

  // Return modified detections
  push_to_port_using_trait( detected_object_set_out1, left_detected_object_set );
  push_to_port_using_trait( detected_object_set_out2, right_detected_object_set );
}

} // end namespace viame
