/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "viame_processes_core_export.h"
#include <viame/pipeline_framework/process_factory.h>
#include <viame/algorithm_framework/plugin/registry.h>

#include "filter_frame_process.h"
#include "filter_frame_index_process.h"
#include "image_to_image_set_process.h"

// -----------------------------------------------------------------------------
/*! \brief Registers processes
 *
 */
extern "C"
VIAME_PROCESSES_CORE_EXPORT
void
register_factories( kwiver::vital::registry& vpm )
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


  // `process_query_adaboost` was registered by `viame_processes_opencv`
  // until P7-T09 took its session from `cv::ml::Boost` to scikit-learn.

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  sprokit::mark_process_module_as_loaded( vpm, module_name );
}
