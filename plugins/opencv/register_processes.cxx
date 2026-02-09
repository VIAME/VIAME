/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "viame_processes_opencv_export.h"
#include <sprokit/pipeline/process_factory.h>
#include <vital/plugin_loader/plugin_loader.h>

#include "measure_objects_process.h"
#include "calibrate_single_camera_process.h"
#include "pair_stereo_detections_process.h"
#include "pair_stereo_tracks_process.h"
#include "detect_in_subregions_process.h"
#include "process_query_process_adaboost.h"

// -----------------------------------------------------------------------------
/*! \brief Registers processes
 *
 */
extern "C"
VIAME_PROCESSES_OPENCV_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name =
    kwiver::vital::plugin_manager::module_t( "viame_processes_opencv" );

  if( sprokit::is_process_module_loaded( vpm, module_name ) )
  {
    return;
  }

  // ---------------------------------------------------------------------------
  auto fact = vpm.ADD_PROCESS( viame::measure_objects_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME,
                       "measure_using_stereo" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Stereo measurement process that matches detections between "
                    "left and right cameras and computes fish length measurements "
                    "using triangulation" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  fact = vpm.ADD_PROCESS( viame::calibrate_single_camera_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME,
                       "ocv_calibrate_single_camera" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Calibrate a single camera from object track set" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  fact = vpm.ADD_PROCESS( viame::pair_stereo_detections_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME,
                       "ocv_pair_stereo_detections" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Compute object detections pair from stereo depth map information" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  fact = vpm.ADD_PROCESS( viame::pair_stereo_tracks_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME,
                       "ocv_pair_stereo_tracks" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Compute object tracks pair from stereo depth map information" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  fact = vpm.ADD_PROCESS( viame::detect_in_subregions_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME,
                       "detect_in_subregions" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Run a detection algorithm on all of the chips represented "
                    "by an incoming detected_object_set" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  // ---------------------------------------------------------------------------
  fact = vpm.ADD_PROCESS( viame::process_query_process_adaboost );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME,
                       "process_query_adaboost" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Process query descriptors using IQR and AdaBoost ranking" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  sprokit::mark_process_module_as_loaded( vpm, module_name );
}
