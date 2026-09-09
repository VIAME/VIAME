/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Register VIAME tool applets into a plugin
 */

#include "viame_tools_applets_export.h"

#include <vital/plugin_management/plugin_loader.h>
#include <vital/applets/applet_registrar.h>

#include <applet_attributes.h>
#include "csv.h"
#include "get_configs.h"
#include "json.h"
#include "resample.h"
#include "run.h"
#include "score.h"
#include "train.h"

#ifdef VIAME_TOOLS_ENABLE_PYTHON
#include <python_script_applet.h>
#endif

namespace viame {
namespace tools {

#ifdef VIAME_TOOLS_ENABLE_PYTHON

VIAME_PYTHON_SCRIPT_APPLET( add_ons_applet, "add-ons", "add_ons.py",
  "List installed add-on model packs and download new ones." )

VIAME_PYTHON_SCRIPT_APPLET( add_segmentations_applet, "add-segmentations",
  "add_segmentations.py",
  "Add SAM2 segmentation polygons to an existing box-level annotation set." )

VIAME_PYTHON_SCRIPT_APPLET( convert_cam_applet, "convert-cam",
  "convert_cam.py",
  "Convert stereo camera calibration files between different formats." )

VIAME_PYTHON_SCRIPT_APPLET( convert_itk_applet,
  "convert-itk", "convert_itk.py",
  "Convert an ITK HDF5 transform into a DIVE camera registration json." )

VIAME_PYTHON_SCRIPT_APPLET( database_applet, "database", "database.py",
  "Initialize, start, stop and index the descriptor database." )

VIAME_PYTHON_SCRIPT_APPLET( extract_frames_applet, "extract-frames",
  "extract_frames.py", "Extract frames from video files" )

VIAME_PYTHON_SCRIPT_APPLET( plot_detections_applet, "plot-detections",
  "plot_detections.py", "Plot detection counts per frame" )

VIAME_PYTHON_SCRIPT_APPLET( generate_nn_index_applet, "generate-nn-index",
  "generate_nn_index.py",
  "Build ITQ LSH index for efficient nearest neighbor search" )

VIAME_PYTHON_SCRIPT_APPLET( launch_annotator_applet, "launch-annotator",
  "launch_annotator.py", "Launch annotation GUI" )

VIAME_PYTHON_SCRIPT_APPLET( launch_search_applet, "launch-search",
  "launch_search.py", "Launch Query GUI" )

VIAME_PYTHON_SCRIPT_APPLET( pipeline_applet, "pipeline", "pipeline.py",
  "Generate, inspect, validate and modify .pipe files" )

VIAME_PYTHON_SCRIPT_APPLET( plot_eval_applet, "plot-eval",
  "plot_eval.py", "Generate plots from VIAME evaluation results" )

VIAME_PYTHON_SCRIPT_APPLET( metadata_applet, "metadata",
  "metadata.py",
  "Dump unified per-image survey metadata for a site folder" )

VIAME_PYTHON_SCRIPT_APPLET( train_fusion_applet, "train-fusion",
  "train_fusion.py",
  "Learn detection fusion parameters for the nms_fusion merger" )

#ifdef VIAME_TOOLS_HAVE_OPENCV

VIAME_PYTHON_SCRIPT_APPLET( calibrate_cameras_applet, "calibrate-cameras",
  "calibrate_cameras.py",
  "Estimate stereo calibration from calibration target images." )

VIAME_PYTHON_SCRIPT_APPLET( compute_depth_applet, "compute-depth",
  "compute_depth.py", "Estimate depth from a pair of rectified images" )

VIAME_PYTHON_SCRIPT_APPLET( compute_disparity_applet, "compute-disparity",
  "compute_disparity.py",
  "Estimate disparity between a pair of rectified images" )

VIAME_PYTHON_SCRIPT_APPLET( create_mosaic_applet, "create-mosaic",
  "create_mosaic.py", "Stitch a mosaic from images and their homographies" )

VIAME_PYTHON_SCRIPT_APPLET( detect_prior_coverage_applet,
  "detect-prior-coverage", "detect_prior_coverage.py",
  "Detect previously-observed regions in survey imagery" )

VIAME_PYTHON_SCRIPT_APPLET( reconstruct_3d_applet, "reconstruct-3d",
  "reconstruct_3d.py", "Build a 3D model from UAS imagery" )

VIAME_PYTHON_SCRIPT_APPLET( rectify_applet, "rectify",
  "rectify.py",
  "Rectify a stereo image pair using calibration parameters" )

#endif

#ifdef VIAME_TOOLS_HAVE_PYTORCH

VIAME_PYTHON_SCRIPT_APPLET( check_gpu_applet, "check-gpu",
  "check_gpu.py", "Check GPU properties of the system" )

#endif

#endif

// ----------------------------------------------------------------------------
/// Register an applet that needs no plugins loaded on its behalf.
template < typename applet_t >
static void
register_standalone_tool( kwiver::applet_registrar& reg )
{
  reg.register_tool< applet_t >()->add_attribute( SKIP_PLUGIN_PRELOAD, "true" );
}

// ----------------------------------------------------------------------------
/// Register an applet that forwards its whole command line to a script.
template < typename applet_t >
static void
register_script_tool( kwiver::applet_registrar& reg )
{
  reg.register_tool< applet_t >()
    ->add_attribute( SKIP_PLUGIN_PRELOAD, "true" )
    .add_attribute( FORWARDS_HELP, "true" );
}

// ----------------------------------------------------------------------------
extern "C"
VIAME_TOOLS_APPLETS_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  kwiver::applet_registrar reg( vpm, "viame.tools.applets" );

  if( reg.is_module_loaded() )
  {
    return;
  }

  // -- register applets --
  register_standalone_tool< csv_applet >( reg );
  register_standalone_tool< get_configs_applet >( reg );
  register_standalone_tool< json_applet >( reg );
  register_standalone_tool< resample_applet >( reg );
  register_standalone_tool< score_applet >( reg );
  register_standalone_tool< train_applet >( reg );
  register_script_tool< run_applet >( reg );

#ifdef VIAME_TOOLS_ENABLE_PYTHON
  register_script_tool< add_ons_applet >( reg );
  register_script_tool< add_segmentations_applet >( reg );
  register_script_tool< convert_cam_applet >( reg );
  register_script_tool< convert_itk_applet >( reg );
  register_script_tool< database_applet >( reg );
  register_script_tool< extract_frames_applet >( reg );
  register_script_tool< plot_detections_applet >( reg );
  register_script_tool< generate_nn_index_applet >( reg );
  register_script_tool< launch_annotator_applet >( reg );
  register_script_tool< launch_search_applet >( reg );
  register_script_tool< pipeline_applet >( reg );
  register_script_tool< plot_eval_applet >( reg );
  register_script_tool< metadata_applet >( reg );
  register_script_tool< train_fusion_applet >( reg );

#ifdef VIAME_TOOLS_HAVE_OPENCV
  register_script_tool< calibrate_cameras_applet >( reg );
  register_script_tool< compute_depth_applet >( reg );
  register_script_tool< compute_disparity_applet >( reg );
  register_script_tool< create_mosaic_applet >( reg );
  register_script_tool< detect_prior_coverage_applet >( reg );
  register_script_tool< reconstruct_3d_applet >( reg );
  register_script_tool< rectify_applet >( reg );
#endif

#ifdef VIAME_TOOLS_HAVE_PYTORCH
  register_script_tool< check_gpu_applet >( reg );
#endif
#endif

  reg.mark_module_as_loaded();
}

} // namespace tools
} // namespace viame
