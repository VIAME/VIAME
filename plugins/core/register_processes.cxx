/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
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
#include "full_frame_tracker_process.h"
#include "track_conductor_process.h"
#include "write_homography_list_process.h"

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

  fact = vpm.ADD_PROCESS( viame::core::full_frame_tracker_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "full_frame_tracker" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Generate tracks covering entire input frames" )
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
