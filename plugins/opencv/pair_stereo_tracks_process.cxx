/**
 * \file
 * \brief Compute object tracks pair from stereo depth map information
 */

#include "pair_stereo_tracks_process.h"
#include "pair_stereo_tracks.h"
#include "pair_stereo_detections.h"

#include <vital/vital_types.h>
#include <vital/types/image_container.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/object_track_set.h>

#include <memory>
#include <opencv2/core/core.hpp>

#include <arrows/ocv/image_container.h>
#include <sprokit/processes/kwiver_type_traits.h>

namespace kv = kwiver::vital;

namespace viame {

create_config_trait( cameras_directory, std::string, "",
  "The calibrated cameras files directory" )
create_config_trait( pairing_method, std::string, "PAIRING_3D",
  "One of PAIRING_3D, PAIRING_IOU, PAIRING_RECTIFIED_IOU" )
create_config_trait( min_detection_number_threshold, int, "0",
  "Filters out tracks with less detections than this threshold." )
create_config_trait( max_detection_number_threshold, int, "std::numeric_limits<int>::max()",
  "Filters out tracks with more detections than this threshold." )
create_config_trait( min_detection_surface_threshold_pix, int, "0",
  "Filters out tracks with less average mask area than this threshold." )
create_config_trait( max_detection_surface_threshold_pix, int, "std::numeric_limits<int>::max()",
  "Filters out tracks with more average mask area than this threshold." )
create_config_trait( iou_pair_threshold, double, "0.1",
  "Used with IOU pairing_method. Minimum IOU threshold below which left and right "
  "tracks will not be paired." )
create_config_trait( do_split_detections, bool, "true",
  "If true, will split tracks with inconsistent detection pairings across frames." )
create_config_trait( detection_split_threshold, int, "3",
  "Number of detections pairings required before split. Used when do_split_detections "
  "is set to true." )
create_config_trait( verbose, bool, "false",
  "If true, will print debug information to the pipeline console." )

create_port_trait( object_track_set1, object_track_set, "Set of object tracks1." )
create_port_trait( object_track_set2, object_track_set, "Set of object tracks2." )
create_port_trait( filtered_object_track_set1, object_track_set, "The stereo filtered object tracks1." )
create_port_trait( filtered_object_track_set2, object_track_set, "The stereo filtered object tracks2." )


// =============================================================================
pair_stereo_tracks_process
::pair_stereo_tracks_process( kv::config_block_sptr const& config )
    : process( config ), d( new pair_stereo_tracks() )
{
  make_ports();
  make_config();
}


pair_stereo_tracks_process
::~pair_stereo_tracks_process() = default;


// -----------------------------------------------------------------------------
void
pair_stereo_tracks_process
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
  // declare_output_port_using_trait( object_track_set, optional );
  declare_output_port_using_trait( filtered_object_track_set1, optional );
  declare_output_port_using_trait( filtered_object_track_set2, optional );
}

// -----------------------------------------------------------------------------
void
pair_stereo_tracks_process
::make_config()
{
  declare_config_using_trait( cameras_directory );
  declare_config_using_trait( min_detection_number_threshold );
  declare_config_using_trait( max_detection_number_threshold );
  declare_config_using_trait( min_detection_surface_threshold_pix );
  declare_config_using_trait( max_detection_surface_threshold_pix );
  declare_config_using_trait( pairing_method );
  declare_config_using_trait( iou_pair_threshold );
  declare_config_using_trait( do_split_detections );
  declare_config_using_trait( detection_split_threshold );
  declare_config_using_trait( verbose );
}

// -----------------------------------------------------------------------------
void
pair_stereo_tracks_process
::_configure()
{
  d->m_cameras_directory = config_value_using_trait( cameras_directory );
  d->m_min_detection_number_threshold = config_value_using_trait( min_detection_number_threshold );
  d->m_max_detection_number_threshold = config_value_using_trait( max_detection_number_threshold );
  d->m_min_detection_surface_threshold_pix = config_value_using_trait( min_detection_surface_threshold_pix );
  d->m_max_detection_surface_threshold_pix = config_value_using_trait( max_detection_surface_threshold_pix );
  d->m_pairing_method = config_value_using_trait( pairing_method );
  d->m_iou_pair_threshold = config_value_using_trait( iou_pair_threshold );
  d->m_do_split_detections = config_value_using_trait( do_split_detections );
  d->m_detection_split_threshold = config_value_using_trait( detection_split_threshold );
  d->m_verbose = config_value_using_trait( verbose );
  d->load_camera_calibration();
}

// -----------------------------------------------------------------------------
void
pair_stereo_tracks_process
::_step()
{
  // Grab inputs from previous process
  kv::object_track_set_sptr input_tracks1 = grab_from_port_using_trait( object_track_set1 );
  kv::object_track_set_sptr input_tracks2 = grab_from_port_using_trait( object_track_set2 );
  kv::timestamp timestamp = grab_from_port_using_trait( timestamp );
  kv::image_container_sptr depth_map = grab_from_port_using_trait( depth_map );

  // Split input disparity into left / right disparity maps
  cv::Mat cv_disparity_left = kwiver::arrows::ocv::image_container::vital_to_ocv(
    depth_map->get_image(),
    kwiver::arrows::ocv::image_container::BGR_COLOR );

  // Estimate 3D positions in left image with disparity
  auto tracks_and_pos1 = d->update_left_tracks_3d_position( input_tracks1->tracks(), cv_disparity_left, timestamp );
  auto left_tracks = std::get< 0 >( tracks_and_pos1 );
  auto left_3d_pos = std::get< 1 >( tracks_and_pos1 );

  // Filter tracks visible only in current timestamp
  auto right_tracks = d->keep_right_tracks_in_current_frame( input_tracks2->tracks(), timestamp );

  // Pair right and left tracks
  d->pair_left_right_tracks( left_tracks, left_3d_pos, right_tracks, timestamp );

  auto port_info = peek_at_port_using_trait( object_track_set1 );
  auto is_input_complete = port_info.datum->type() == sprokit::datum::complete;
  if( is_input_complete )
  {
    auto left_right_tracks = d->get_left_right_tracks_with_pairing();
    auto output1 = std::make_shared< kv::object_track_set >( std::get< 0 >( left_right_tracks ) );
    auto output2 = std::make_shared< kv::object_track_set >( std::get< 1 >( left_right_tracks ) );

    push_to_port_using_trait( timestamp, timestamp );
    push_to_port_using_trait( filtered_object_track_set1, output1 );
    push_to_port_using_trait( filtered_object_track_set2, output2 );

    mark_process_as_complete();
    const auto complete_dat = sprokit::datum::complete_datum();
    push_datum_to_port_using_trait( timestamp, complete_dat );
    push_datum_to_port_using_trait( filtered_object_track_set1, complete_dat );
    push_datum_to_port_using_trait( filtered_object_track_set2, complete_dat );
  }
  else
  {
    auto no_tracks = std::make_shared< kv::object_track_set >( std::vector< kv::track_sptr >{} );
    push_to_port_using_trait( timestamp, timestamp );
    push_to_port_using_trait( filtered_object_track_set1, no_tracks );
    push_to_port_using_trait( filtered_object_track_set2, no_tracks );
  }
}

} // end namespace viame
