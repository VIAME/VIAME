/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Classifier and detection filter registration
 */

#include "viame_classifiers_plugin_export.h"

#include <viame/algorithm_framework/algo/detected_object_filter.h>
#include <viame/algorithm_framework/plugin/plugin_loader.h>

#include "class_probability_filter.h"


namespace viame {

namespace kv = kwiver::vital;

extern "C"
VIAME_CLASSIFIERS_PLUGIN_EXPORT
void
register_factories( kv::plugin_loader& vpm )
{
  using kvpf = kv::plugin_factory;
  const std::string module_name = "viame.classifiers";

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  // Imported from arrows/core in P5-T04, under the name it registered under
  // there.
#define VIAME_REGISTER_IMPORTED( interface, impl, plugin, blurb )        \
  {                                                                      \
    auto fact = vpm.add_factory< interface, impl >( plugin );            \
    fact->add_attribute( kvpf::PLUGIN_NAME, plugin )                     \
      .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )            \
      .add_attribute( kvpf::PLUGIN_DESCRIPTION, blurb );                 \
  }

  VIAME_REGISTER_IMPORTED(
    kv::algo::detected_object_filter,
    kwiver::arrows::core::class_probability_filter,
    "class_probability_filter",
    "Filter detections by the probability of their classes" )

#undef VIAME_REGISTER_IMPORTED

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
