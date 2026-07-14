/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief CppDB plugin algorithm registration interface impl
 */

#include "viame_cppdb_plugin_export.h"
#include <vital/plugin_management/plugin_loader.h>

#include <vital/algo/query_track_descriptor_set.h>
#include <vital/algo/read_object_track_set.h>
#include <vital/algo/write_object_track_set.h>
#include <vital/algo/write_track_descriptor_set.h>

#include "write_object_track_set_db.h"
#include "write_track_descriptor_set_db.h"
#include "query_track_descriptor_set_db.h"
#include "read_object_track_set_db.h"

namespace viame {

namespace kv = kwiver::vital;

extern "C"
VIAME_CPPDB_PLUGIN_EXPORT
void
register_factories( kv::plugin_loader& vpm )
{
  using kvpf = kv::plugin_factory;
  const std::string module_name = "viame.cppdb";

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  auto fact = vpm.add_factory< kv::algo::write_object_track_set, write_object_track_set_db >(
    "db" );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::write_track_descriptor_set, write_track_descriptor_set_db >(
    "db" );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::query_track_descriptor_set, query_track_descriptor_set_db >(
    "db" );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::read_object_track_set, read_object_track_set_db >(
    "db" );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
