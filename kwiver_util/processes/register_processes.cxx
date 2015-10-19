/*ckwg +5
 * Copyright 2014 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <sprokit/pipeline/process_registry.h>

// -- list processes to register --
#include "kw_archive_writer_process.h"
#include "frame_list_process.h"
#include "stabilize_image_process.h"
#include "matcher_process.h"
#include "compute_homography_process.h"

#include "detect_features_process.h"
#include "extract_descriptors_process.h"
#include "matcher_process.h"
#include "compute_homography_process.h"
#include "view_image_process.h"

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
    sprokit::process_registry::module_t( "kwiver_processes" );

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
    "kw_archive_writer", "Write kw archives",
    sprokit::create_process< kwiver::kw_archive_writer_process > );


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
    "compute_homography_process", "Compute a frame to frame homography based on tracks",
    sprokit::create_process< kwiver::compute_homography_process > );


  // - - - - - - - - - - - - - - - - - - - - - - -
  registry->mark_module_as_loaded( module_name );
}
