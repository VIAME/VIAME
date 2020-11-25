// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <arrows/kpf/kwiver_algo_kpf_plugin_export.h>
#include <vital/algo/algorithm_factory.h>

#include <arrows/kpf/detected_object_set_input_kpf.h>
#include <arrows/kpf/detected_object_set_output_kpf.h>

namespace kwiver {
namespace arrows {
namespace kpf {

extern "C"
KWIVER_ALGO_KPF_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name = std::string( "arrows.kpf" );
  if (vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  // add factory               implementation-name       type-to-create
  auto fact = vpm.ADD_ALGORITHM( "kpf_input", kwiver::arrows::kpf::detected_object_set_input_kpf);
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Detected object set reader using kpf format." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;

  fact = vpm.ADD_ALGORITHM( "kpf_output", kwiver::arrows::kpf::detected_object_set_output_kpf);
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Detected object set writer using kpf format.t" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;

  vpm.mark_module_as_loaded( module_name );
}

} } } // end namespace
