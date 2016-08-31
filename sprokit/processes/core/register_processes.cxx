/*ckwg +29
 * Copyright 2014-2016 by Kitware, Inc.
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

#include <sprokit/pipeline/process_registry.h>

// -- list processes to register --
#include "compute_homography_process.h"
#include "detect_features_process.h"
#include "draw_tracks_process.h"
#include "extract_descriptors_process.h"
#include "frame_list_process.h"
#include "matcher_process.h"
#include "read_descriptor_process.h"
#include "stabilize_image_process.h"
#include "image_object_detector_process.h"
#include "image_filter_process.h"
#include "image_writer_process.h"


extern "C"
KWIVER_PROCESSES_EXPORT void register_processes();


// ----------------------------------------------------------------
/*! \brief Regsiter processes
 *
 *
 */
void register_processes()
{
  static sprokit::process_registry::module_t const module_name =
    sprokit::process_registry::module_t( "kwiver_processes_core" );

  sprokit::process_registry_t const registry( sprokit::process_registry::self() );

  if ( registry->is_module_loaded( module_name ) )
  {
    return;
  }

  // ----------------------------------------------------------------
  registry->register_process(
    "frame_list_input",
    "Reads a list of image file names and generates stream "
    "of images and associated time stamps",
    sprokit::create_process< kwiver::frame_list_process > );

  registry->register_process(
    "stabilize_image", "Generate current-to-reference image homographies",
    sprokit::create_process< kwiver::stabilize_image_process > );

  registry->register_process(
    "detect_features", "Detect features in an image that will be used for stabilization",
    sprokit::create_process< kwiver::detect_features_process > );

  registry->register_process(
    "extract_descriptors", "Extract descriptors from detected features",
    sprokit::create_process< kwiver::extract_descriptors_process > );

  registry->register_process(
    "feature_matcher", "Match extracted descriptors and detected features",
    sprokit::create_process< kwiver::matcher_process > );

  registry->register_process(
    "compute_homography", "Compute a frame to frame homography based on tracks",
    sprokit::create_process< kwiver::compute_homography_process > );

  registry->register_process(
    "draw_tracks", "Draw feature tracks on image",
    sprokit::create_process< kwiver::draw_tracks_process > );

  registry->register_process(
    "read_d_vector", "Read vector of doubles",
    sprokit::create_process< kwiver::read_descriptor_process > );

  registry->register_process(
    "image_object_detector", "Apply selected image object detector algorithm to incoming images.",
    sprokit::create_process< kwiver::image_object_detector_process > );

  registry->register_process(
    "image_filter", "Apply selected image filter algorithm to incoming images.",
    sprokit::create_process< kwiver::image_filter_process > );

  registry->register_process(
    "image_writer", "Write image to disk.",
    sprokit::create_process< kwiver::image_writer_process > );

  // - - - - - - - - - - - - - - - - - - - - - - -
  registry->mark_module_as_loaded( module_name );
}
