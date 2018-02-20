/*ckwg +29
 * Copyright 2014-2017 by Kitware, Inc.
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

#include <sprokit/processes/core/kwiver_processes_export.h>

#include <sprokit/pipeline/process_factory.h>
#include <vital/plugin_loader/plugin_loader.h>

// -- list processes to register --
#include "associate_detections_to_tracks_process.h"
#include "compute_association_matrix_process.h"
#include "compute_homography_process.h"
#include "compute_stereo_depth_map_process.h"
#include "compute_track_descriptors_process.h"
#include "detect_features_process.h"
#include "detected_object_filter_process.h"
#include "detected_object_input_process.h"
#include "detected_object_output_process.h"
#include "downsample_process.h"
#include "draw_detected_object_set_process.h"
#include "draw_tracks_process.h"
#include "extract_descriptors_process.h"
#include "frame_list_process.h"
#include "handle_descriptor_request_process.h"
#include "image_file_reader_process.h"
#include "image_filter_process.h"
#include "image_object_detector_process.h"
#include "image_writer_process.h"
#include "initialize_object_tracks_process.h"
#include "matcher_process.h"
#include "perform_query_process.h"
#include "read_descriptor_process.h"
#include "read_object_track_process.h"
#include "read_track_descriptor_process.h"
#include "refine_detections_process.h"
#include "split_image_process.h"
#include "stabilize_image_process.h"
#include "video_input_process.h"
#include "write_object_track_process.h"
#include "write_track_descriptor_process.h"

// ---------------------------------------------------------------------------------------
/*! \brief Regsiter processes
 *
 *
 */
extern "C"
KWIVER_PROCESSES_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name = kwiver::vital::plugin_manager::module_t( "kwiver_processes_core" );

  if ( sprokit::is_process_module_loaded( vpm, module_name ) )
  {
    return;
  }

  // -------------------------------------------------------------------------------------
  auto fact = vpm.ADD_PROCESS( kwiver::frame_list_process );

  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "frame_list_input" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Reads a list of image file names and generates stream of images and associated time stamps." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;


  fact = vpm.ADD_PROCESS( kwiver::stabilize_image_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "stabilize_image" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Generate current-to-reference image homographies." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;


  fact = vpm.ADD_PROCESS( kwiver::detect_features_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "detect_features" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Detect features in an image that will be used for stabilization." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;


  fact = vpm.ADD_PROCESS( kwiver::extract_descriptors_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "extract_descriptors" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Extract descriptors from detected features." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;


  fact = vpm.ADD_PROCESS( kwiver::matcher_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "feature_matcher" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Match extracted descriptors and detected features." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;


  fact = vpm.ADD_PROCESS( kwiver::compute_homography_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "compute_homography" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Compute a frame to frame homography based on tracks." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;


  fact = vpm.ADD_PROCESS( kwiver::compute_stereo_depth_map_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "compute_stereo_depth_map" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Compute a stereo depth map given two frames." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;


  fact = vpm.ADD_PROCESS( kwiver::draw_tracks_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "draw_tracks" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Draw feature tracks on image." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;


  fact = vpm.ADD_PROCESS( kwiver::read_descriptor_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "read_d_vector" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Read vector of doubles," )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;


  fact = vpm.ADD_PROCESS( kwiver::refine_detections_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "refine_detections" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Refines detections for a given frame," )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;


  fact = vpm.ADD_PROCESS( kwiver::image_object_detector_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "image_object_detector" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Apply selected image object detector algorithm to incoming images." )
  .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
     ;


  fact = vpm.ADD_PROCESS( kwiver::image_filter_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "image_filter" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Apply selected image filter algorithm to incoming images." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;


  fact = vpm.ADD_PROCESS( kwiver::image_writer_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "image_writer" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "Write image to disk." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;


  fact = vpm.ADD_PROCESS( kwiver::image_file_reader_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "image_file_reader" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "Reads an image file given the file name." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;


  fact = vpm.ADD_PROCESS( kwiver::detected_object_input_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "detected_object_input" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Reads detected object sets from an input file.\n\n"
                    "Detections read from the input file are grouped into sets for each "
                    "image and individually returned." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;


  fact = vpm.ADD_PROCESS( kwiver::detected_object_output_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "detected_object_output" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Writes detected object sets to an output file.\n\n"
                    "All detections are written to the same file." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;


  fact = vpm.ADD_PROCESS( kwiver::detected_object_filter_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "detected_object_filter" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Filters sets of detected objects using the detected_object_filter algorithm." )
  .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
     ;


  fact = vpm.ADD_PROCESS( kwiver::video_input_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "video_input" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Reads video files and produces sequential images with metadata per frame." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;


  fact = vpm.ADD_PROCESS( kwiver::draw_detected_object_set_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "draw_detected_object_set" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Draws border around detected objects in the set using the selected algorithm.\n\n"
                    "This process is a wrapper around a draw_detected_object_set algorithm.")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;


  fact = vpm.ADD_PROCESS( kwiver::split_image_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "split_image" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Split a image into multiple smaller images." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;


  fact = vpm.ADD_PROCESS( kwiver::read_track_descriptor_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "read_track_descriptor" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Reads track descriptor sets from an input file." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;


  fact = vpm.ADD_PROCESS( kwiver::write_track_descriptor_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "write_track_descriptor" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Writes track descriptor sets to an output file. All descriptors are written to the same file." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;


  fact = vpm.ADD_PROCESS( kwiver::read_object_track_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "read_object_track" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Reads object track sets from an input file." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;


  fact = vpm.ADD_PROCESS( kwiver::write_object_track_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "write_object_track" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Writes object track sets to an output file. All descriptors are written to the same file." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;


  fact = vpm.ADD_PROCESS( kwiver::associate_detections_to_tracks_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "associate_detections_to_tracks" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Associates new detections to existing tracks given a cost matrix." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;


  fact = vpm.ADD_PROCESS( kwiver::compute_association_matrix_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "compute_association_matrix" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Computes cost matrix for adding new detections to existing tracks." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;


  fact = vpm.ADD_PROCESS( kwiver::initialize_object_tracks_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "initialize_object_tracks" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Initialize new object tracks given detections for the current frame." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;


  fact = vpm.ADD_PROCESS( kwiver::handle_descriptor_request_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "handle_descriptor_request" );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
    "Handle a new descriptor request, producing desired descriptors on the input." );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  fact = vpm.ADD_PROCESS( kwiver::compute_track_descriptors_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "compute_track_descriptors" );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
    "Compute track descriptors on the input tracks or detections." );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  fact = vpm.ADD_PROCESS( kwiver::perform_query_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "perform_query" );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
    "Perform a query." );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  fact = vpm.ADD_PROCESS( kwiver::downsample_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "downsample" );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
    "Downsample an input stream." );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );


  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  sprokit::mark_process_module_as_loaded( vpm, module_name );
} // register_processes
