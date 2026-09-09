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
#include "configs.h"
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

VIAME_PYTHON_SCRIPT_APPLET( segment_applet, "segment", "segment.py",
  "Add SAM2 segmentation polygons to an existing box-level annotation set." )

VIAME_PYTHON_SCRIPT_APPLET( convert_applet, "convert", "convert.py",
  "Convert stereo calibration files between formats, or an ITK HDF5 "
  "transform into a DIVE camera registration json." )

VIAME_PYTHON_SCRIPT_APPLET( database_applet, "database", "database.py",
  "Initialize, start, stop and index the descriptor database." )

// The script is not named inspect.py: a tool run from the configs folder
// would otherwise shadow the standard library inspect module for its imports.
VIAME_PYTHON_SCRIPT_APPLET( inspect_applet, "inspect", "inspect_file.py",
  "Identify a file, check it is intact, and say how VIAME can use it." )

VIAME_PYTHON_SCRIPT_APPLET( extract_applet, "extract",
  "extract.py", "Extract frames from video files" )

VIAME_PYTHON_SCRIPT_APPLET( index_applet, "index", "index.py",
  "Build and manage the video search index: add, remove, build, list, "
  "status, hash." )

VIAME_PYTHON_SCRIPT_APPLET( view_applet, "view",
  "view.py", "Launch the annotation and viewing GUI" )

VIAME_PYTHON_SCRIPT_APPLET( search_applet, "search",
  "search.py", "Launch the video search (query) GUI" )

VIAME_PYTHON_SCRIPT_APPLET( pipeline_applet, "pipeline", "pipeline.py",
  "Generate, inspect, validate and modify .pipe files" )

VIAME_PYTHON_SCRIPT_APPLET( plot_applet, "plot", "plot.py",
  "Plot detection counts per frame, or evaluation results." )

VIAME_PYTHON_SCRIPT_APPLET( metadata_applet, "metadata",
  "metadata.py",
  "Dump unified per-image survey metadata for a site folder" )

VIAME_PYTHON_SCRIPT_APPLET( train_fusion_applet, "train-fusion",
  "train_fusion.py",
  "Learn detection fusion parameters for the nms_fusion merger" )

#ifdef VIAME_TOOLS_HAVE_OPENCV

VIAME_PYTHON_SCRIPT_APPLET( calibrate_applet, "calibrate", "calibrate.py",
  "Estimate stereo calibration from calibration target images." )

VIAME_PYTHON_SCRIPT_APPLET( depth_applet, "depth",
  "depth.py", "Estimate depth from a pair of rectified images" )

VIAME_PYTHON_SCRIPT_APPLET( disparity_applet, "disparity",
  "disparity.py", "Estimate disparity between a pair of rectified images" )

VIAME_PYTHON_SCRIPT_APPLET( mosaic_applet, "mosaic",
  "mosaic.py", "Stitch a mosaic from images and their homographies" )

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

VIAME_PYTHON_SCRIPT_APPLET( gpu_applet, "gpu",
  "gpu.py", "Check GPU properties of the system" )

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
  register_standalone_tool< configs_applet >( reg );
  register_standalone_tool< json_applet >( reg );
  register_standalone_tool< resample_applet >( reg );
  register_standalone_tool< score_applet >( reg );
  register_standalone_tool< train_applet >( reg );
  register_script_tool< run_applet >( reg );

#ifdef VIAME_TOOLS_ENABLE_PYTHON
  register_script_tool< add_ons_applet >( reg );
  register_script_tool< segment_applet >( reg );
  register_script_tool< convert_applet >( reg );
  register_script_tool< database_applet >( reg );
  register_script_tool< inspect_applet >( reg );
  register_script_tool< extract_applet >( reg );
  register_script_tool< index_applet >( reg );
  register_script_tool< view_applet >( reg );
  register_script_tool< search_applet >( reg );
  register_script_tool< pipeline_applet >( reg );
  register_script_tool< plot_applet >( reg );
  register_script_tool< metadata_applet >( reg );
  register_script_tool< train_fusion_applet >( reg );

#ifdef VIAME_TOOLS_HAVE_OPENCV
  register_script_tool< calibrate_applet >( reg );
  register_script_tool< depth_applet >( reg );
  register_script_tool< disparity_applet >( reg );
  register_script_tool< mosaic_applet >( reg );
  register_script_tool< detect_prior_coverage_applet >( reg );
  register_script_tool< reconstruct_3d_applet >( reg );
  register_script_tool< rectify_applet >( reg );
#endif

#ifdef VIAME_TOOLS_HAVE_PYTORCH
  register_script_tool< gpu_applet >( reg );
#endif
#endif

  reg.mark_module_as_loaded();
}

} // namespace tools
} // namespace viame
