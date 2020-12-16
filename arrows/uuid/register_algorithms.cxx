// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Defaults plugin algorithm registration interface impl
 */

#include <arrows/uuid/kwiver_algo_uuid_plugin_export.h>
#include <vital/algo/algorithm_factory.h>

#include <arrows/uuid/uuid_factory_uuid.h>

namespace kwiver {
namespace arrows {
namespace uuid {

extern "C"
KWIVER_ALGO_UUID_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name = std::string( "arrows.uuid" );
  if (vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  // add factory                  implementation-name       type-to-create
  auto fact = vpm.ADD_ALGORITHM( "uuid", kwiver::arrows::uuid::uuid_factory_uuid );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Global UUID generator using system library as source for UUID." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;

  vpm.mark_module_as_loaded( module_name );
}

} } } // end namespace
