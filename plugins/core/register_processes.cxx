/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "viame_processes_core_export.h"
#include <sprokit/pipeline/process_factory.h>
#include <vital/plugin_loader/plugin_loader.h>

#include "accumulate_image_statistics_process.h"
#include "align_multimodal_imagery_process.h"
#include "extract_desc_ids_for_training_process.h"
#include "filter_frame_process.h"
#include "filter_object_tracks_process.h"
#include "stack_frames_process.h"
#include "detect_shot_breaks_process.h"
#include "measure_objects_process.h"
#include "read_habcam_metadata_process.h"
#include "refine_measurements_process.h"
#include "track_conductor_process.h"
#include "write_homography_list_process.h"
#include "accumulate_object_tracks_process.h"
#include "filter_frame_index_process.h"
#include "calibrate_cameras_from_tracks_process.h"
#include "split_tracks_to_feature_landmarks_process.h"
#include "pair_stereo_detections_process.h"
#include "query_results_to_tracks_process.h"
#include "create_database_query_process.h"
#include "image_to_image_set_process.h"

// -----------------------------------------------------------------------------
/*! \brief Registers processes
 *
 */
extern "C"
VIAME_PROCESSES_CORE_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name =
    kwiver::vital::plugin_manager::module_t( "viame_processes_core" );

  if( sprokit::is_process_module_loaded( vpm, module_name ) )
  {
    return;
  }

  // ---------------------------------------------------------------------------
  auto fact = vpm.ADD_PROCESS( viame::core::accumulate_image_statistics_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "accumulate_image_statistics" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Accumulate image statistics (frame count, dimensions) over a stream" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  fact = vpm.ADD_PROCESS( viame::core::align_multimodal_imagery_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "align_multimodal_imagery" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Align multimodal images that may be out of sync" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  fact = vpm.ADD_PROCESS( viame::core::extract_desc_ids_for_training_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "extract_desc_ids_for_training" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Extract descriptor IDs overlapping with groundtruth" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  fact = vpm.ADD_PROCESS( viame::core::filter_frame_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "filter_frames" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Filter frames based on some property" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  fact = vpm.ADD_PROCESS( viame::core::filter_object_tracks_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "filter_object_tracks" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Filter object tracks based on different filters" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  fact = vpm.ADD_PROCESS( viame::core::stack_frames_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "stack_frames" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Stack multiple frames on top of each in the same image" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  fact = vpm.ADD_PROCESS( viame::core::detect_shot_breaks_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "detect_shot_breaks" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Detect shot breaks and create tracks for each shot" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;
  
  fact = vpm.ADD_PROCESS( viame::core::filter_frame_index_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "filter_frame_index" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Pass frame in min max index limits" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;
  
  fact = vpm.ADD_PROCESS( viame::core::accumulate_object_tracks_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "accumulate_object_tracks" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Accumulate detected objects into an object track set" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  fact = vpm.ADD_PROCESS( viame::core::measure_objects_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "compute_measurements" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Compute stereo measurements from track data" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  fact = vpm.ADD_PROCESS( viame::core::split_tracks_to_feature_landmarks_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "split_tracks_to_feature_landmarks" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Split an object track set into a feature_track_set and a landmark_map" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;
  
  fact = vpm.ADD_PROCESS( viame::core::calibrate_cameras_from_tracks_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "calibrate_cameras_from_tracks" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Calibrate stereo cameras from object track sets" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  fact = vpm.ADD_PROCESS( viame::core::pair_stereo_detections_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "pair_stereo_detections" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Match detections across stereo views using IOU and class labels, output tracks with aligned IDs" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  fact = vpm.ADD_PROCESS( viame::core::read_habcam_metadata_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "read_habcam_metadata" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Read HabCam metadata from input files" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  fact = vpm.ADD_PROCESS( viame::core::refine_measurements_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "refine_measurements" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Refine measurements using either local or global GSDs" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  fact = vpm.ADD_PROCESS( viame::core::track_conductor_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "track_conductor" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Consolidate and control multiple other trackers" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  fact = vpm.ADD_PROCESS( viame::core::write_homography_list_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "write_homography_list" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Write a homography list out to some file" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  fact = vpm.ADD_PROCESS( viame::core::query_results_to_tracks_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "query_results_to_tracks" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Convert query results to object track set with NN scores as confidence" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  fact = vpm.ADD_PROCESS( viame::core::create_database_query_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "create_database_query" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Create database query from track descriptors for use with perform_query" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  fact = vpm.ADD_PROCESS( viame::core::image_to_image_set_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "image_to_image_set" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Convert single image to image_set" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  sprokit::mark_process_module_as_loaded( vpm, module_name );
}
