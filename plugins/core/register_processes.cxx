/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "viame_processes_core_export.h"
#include <sprokit/pipeline/process_factory.h>
#include <vital/plugin_loader/plugin_loader.h>

#include "accumulate_image_statistics_process.h"
#include "align_multimodal_imagery_process.h"
#include "extract_desc_ids_for_training_process.h"
#include "fetch_descriptors_process.h"
#include "filter_frame_process.h"
#include "ingest_descriptors_process.h"
#include "filter_object_tracks_process.h"
#include "object_track_descriptors_process.h"
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
#include "write_query_results_as_tracks_process.h"
#include "create_database_query_process.h"
#include "select_database_query_process.h"
#include "image_to_image_set_process.h"
#include "resample_object_tracks_process.h"

// -----------------------------------------------------------------------------
/*! \brief Registers processes
 *
 */
extern "C"
VIAME_PROCESSES_CORE_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  using namespace sprokit;
  static auto const module_name = kwiver::vital::plugin_manager::module_t( "viame_processes_core" );
  kwiver::vital::plugin_factory_handle_t fact_handle;
    if( sprokit::is_process_module_loaded( vpm, module_name ) )
  {
    return;
  }

  // ---------------------------------------------------------------------------
  using kvpf = kwiver::vital::plugin_factory;

  kwiver::vital::plugin_factory* fact = new sprokit::cpp_process_factory(
    typeid( viame::core::accumulate_image_statistics_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::core::accumulate_image_statistics_process > );
  fact->add_attribute( kvpf::PLUGIN_NAME, "accumulate_image_statistics" )
    .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );
  vpm.add_factory( fact );

  fact = new sprokit::cpp_process_factory(
    typeid( viame::core::align_multimodal_imagery_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::core::align_multimodal_imagery_process > );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "align_multimodal_imagery" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Align multimodal images that may be out of sync" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );

  fact = new sprokit::cpp_process_factory(
    typeid( viame::core::extract_desc_ids_for_training_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::core::extract_desc_ids_for_training_process > );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "extract_desc_ids_for_training" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Extract descriptor IDs overlapping with groundtruth" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );

  fact = new sprokit::cpp_process_factory(
    typeid( viame::core::ingest_descriptors_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::core::ingest_descriptors_process > );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "ingest_descriptors" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Ingest descriptors with UIDs and write to file" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );

  fact = new sprokit::cpp_process_factory(
    typeid( viame::core::fetch_descriptors_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::core::fetch_descriptors_process > );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "fetch_descriptors" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Fetch descriptors from file given UIDs" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );

  fact = new sprokit::cpp_process_factory(
    typeid( viame::core::object_track_descriptors_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::core::object_track_descriptors_process > );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "object_track_descriptors" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Attach descriptors to object track states from file" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );

  fact = new sprokit::cpp_process_factory(
    typeid( viame::core::filter_frame_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::core::filter_frame_process > );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "filter_frames" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Filter frames based on some property" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );

  fact = new sprokit::cpp_process_factory(
    typeid( viame::core::filter_object_tracks_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::core::filter_object_tracks_process > );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "filter_object_tracks" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Filter object tracks based on different filters" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );

  fact = new sprokit::cpp_process_factory(
    typeid( viame::core::stack_frames_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::core::stack_frames_process > );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "stack_frames" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Stack multiple frames on top of each in the same image" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );

  fact = new sprokit::cpp_process_factory(
    typeid( viame::core::detect_shot_breaks_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::core::detect_shot_breaks_process > );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "detect_shot_breaks" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Detect shot breaks and create tracks for each shot" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );
  
  fact = new sprokit::cpp_process_factory(
    typeid( viame::core::filter_frame_index_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::core::filter_frame_index_process > );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "filter_frame_index" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Pass frame in min max index limits" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );
  
  fact = new sprokit::cpp_process_factory(
    typeid( viame::core::accumulate_object_tracks_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::core::accumulate_object_tracks_process > );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "accumulate_object_tracks" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Accumulate detected objects into an object track set" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );

  fact = new sprokit::cpp_process_factory(
    typeid( viame::core::measure_objects_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::core::measure_objects_process > );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "compute_measurements" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Compute stereo measurements from track data" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );

  fact = new sprokit::cpp_process_factory(
    typeid( viame::core::split_tracks_to_feature_landmarks_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::core::split_tracks_to_feature_landmarks_process > );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "split_tracks_to_feature_landmarks" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Split an object track set into a feature_track_set and a landmark_map" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );
  
  fact = new sprokit::cpp_process_factory(
    typeid( viame::core::calibrate_cameras_from_tracks_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::core::calibrate_cameras_from_tracks_process > );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "calibrate_cameras_from_tracks" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Calibrate stereo cameras from object track sets" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );

  fact = new sprokit::cpp_process_factory(
    typeid( viame::core::pair_stereo_detections_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::core::pair_stereo_detections_process > );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "pair_stereo_detections" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Match detections across stereo views using IOU and class labels, output tracks with aligned IDs" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );

  fact = new sprokit::cpp_process_factory(
    typeid( viame::core::read_habcam_metadata_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::core::read_habcam_metadata_process > );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "read_habcam_metadata" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Read HabCam metadata from input files" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );

  fact = new sprokit::cpp_process_factory(
    typeid( viame::core::refine_measurements_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::core::refine_measurements_process > );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "refine_measurements" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Refine measurements using either local or global GSDs" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );

  fact = new sprokit::cpp_process_factory(
    typeid( viame::core::track_conductor_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::core::track_conductor_process > );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "track_conductor" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Consolidate and control multiple other trackers" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );

  fact = new sprokit::cpp_process_factory(
    typeid( viame::core::write_homography_list_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::core::write_homography_list_process > );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "write_homography_list" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Write a homography list out to some file" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );

  fact = new sprokit::cpp_process_factory(
    typeid( viame::core::write_query_results_as_tracks_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::core::write_query_results_as_tracks_process > );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "write_query_results_as_tracks" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Write query results as object track CSV with NN scores as confidence" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );

  fact = new sprokit::cpp_process_factory(
    typeid( viame::core::create_database_query_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::core::create_database_query_process > );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "create_database_query" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Create database query from track descriptors for use with perform_query" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );

  fact = new sprokit::cpp_process_factory(
    typeid( viame::core::select_database_query_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::core::select_database_query_process > );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "select_database_query" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Select between two database query inputs (primary if non-null, otherwise fallback)" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );

  fact = new sprokit::cpp_process_factory(
    typeid( viame::core::image_to_image_set_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::core::image_to_image_set_process > );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "image_to_image_set" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Convert single image to image_set" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );

  fact = new sprokit::cpp_process_factory(
    typeid( viame::core::resample_object_tracks_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::core::resample_object_tracks_process > );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "resample_object_tracks" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Resample object tracks from one downsample rate to another" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  sprokit::mark_process_module_as_loaded( vpm, module_name );
}
