/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief SVM algorithm registration
 *
 * The SVM refiner and detector trainer, from `svm` in P2-T08. Their own
 * library, as darknet's detector is, because libsvm is optional.
 */

#include "viame_classifiers_svm_plugin_export.h"
#include <viame/algorithm_framework/plugin/registry.h>

#include <viame/algorithm_framework/algo/refine_detections.h>
#include <viame/algorithm_framework/algo/train_detector.h>

#include "refine_detections_svm.h"
#include "train_detector_svm.h"

namespace viame {

namespace kv = viame;

extern "C"
VIAME_CLASSIFIERS_SVM_PLUGIN_EXPORT
void
register_factories( kv::registry& vpm )
{
  using kvpf = kv::plugin_factory;
  const std::string module_name = "viame.classifiers.svm";

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  auto fact = vpm.add_factory< kv::algo::refine_detections, refine_detections_svm >(
    refine_detections_svm::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::train_detector, train_detector_svm >(
    train_detector_svm::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
