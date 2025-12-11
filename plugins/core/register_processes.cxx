/*ckwg +29
 * Copyright 2023 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <plugins/core/viame_processes_core_export.h>
#include <sprokit/pipeline/process_factory.h>
#include <vital/plugin_loader/plugin_loader.h>

#include "align_multimodal_imagery_process.h"
#include "extract_desc_ids_for_training_process.h"
#include "filter_frame_process.h"
#include "filter_object_tracks_process.h"
#include "frame_stacker_process.h"
#include "full_frame_tracker_process.h"
#include "measurement_process.h"
#include "read_habcam_metadata_process.h"
#include "refine_measurements_process.h"
#include "track_conductor_process.h"
#include "write_homography_list_process.h"
#include "append_detections_to_tracks_process.h"
#include "filter_frame_index_process.h"
#include "calibrate_cameras_from_tracks_process.h"
#include "split_object_track_to_feature_landmark_process.h"
#include "tracks_pairing_from_stereo_process.h"
#include "detections_pairing_from_stereo_process.h"

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
  auto fact = vpm.ADD_PROCESS( viame::core::align_multimodal_imagery_process );
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

  fact = vpm.ADD_PROCESS( viame::core::frame_stacker_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "frame_stacker_process" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Stack multiple frames on top of each in the same image" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  fact = vpm.ADD_PROCESS( viame::core::full_frame_tracker_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "full_frame_tracker" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Generate tracks covering entire input frames" )
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
  
  fact = vpm.ADD_PROCESS( viame::core::append_detections_to_tracks_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "append_detections_to_tracks" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Append consistent detected object set to an object track set" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  fact = vpm.ADD_PROCESS( viame::core::measurement_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "measurement" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Compute stereo measurements from track data" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  fact = vpm.ADD_PROCESS( viame::core::split_object_track_to_feature_landmark_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "split_object_track_to_feature_landmark" )
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

  fact = vpm.ADD_PROCESS( viame::core::tracks_pairing_from_stereo_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "tracks_pairing_from_stereo" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Compute object tracks pair from stereo depth map information" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  fact = vpm.ADD_PROCESS( viame::core::detections_pairing_from_stereo_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "detections_pairing_from_stereo" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Compute object detections pair from stereo depth map information" )
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

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  sprokit::mark_process_module_as_loaded( vpm, module_name );
}
