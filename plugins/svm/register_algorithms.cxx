/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief SVM algorithm registration implementation
 */

#include "viame_svm_plugin_export.h"
#include <vital/algo/algorithm_factory.h>

#include "refine_detections_svm.h"
#include "train_detector_svm.h"

namespace viame {

extern "C"
VIAME_SVM_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name = std::string( "viame.svm" );
  if (vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  auto fact = vpm.ADD_ALGORITHM( "svm", viame::refine_detections_svm );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Apply svm to refine detection" )
      .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
      .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
      .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
      ;

  auto fact2 = vpm.ADD_ALGORITHM( "svm", viame::train_detector_svm );
  fact2->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                        "Train SVM models for object detection" )
      .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
      .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
      .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
      ;

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
