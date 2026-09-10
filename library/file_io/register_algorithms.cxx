/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief kw18 detection and track reader and writer registration
 *
 * Imported from arrows/core in P5-T04, along with the feature and descriptor
 * reader and writer. The implementations are unchanged and
 * are still in kwiver's namespace; what moved is where they are built and
 * where they register.
 */

#include "viame_file_io_plugin_export.h"

#include <viame/algorithm_framework/algo/detected_object_set_input.h>
#include <viame/algorithm_framework/algo/detected_object_set_output.h>
#include <viame/algorithm_framework/algo/feature_descriptor_io.h>
#include <viame/algorithm_framework/algo/read_object_track_set.h>
#include <viame/algorithm_framework/algo/write_object_track_set.h>

#include <viame/algorithm_framework/plugin/plugin_loader.h>

#include "detected_object_set_input_kw18.h"
#include "detected_object_set_output_kw18.h"
#include "feature_descriptor_io.h"
#include "read_object_track_set_kw18.h"
#include "write_object_track_set_kw18.h"

namespace viame {

namespace kv = kwiver::vital;

extern "C"
VIAME_FILE_IO_PLUGIN_EXPORT
void
register_factories( kv::plugin_loader& vpm )
{
  using kvpf = kv::plugin_factory;
  const std::string module_name = "viame.file_io";

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

#define VIAME_REGISTER( interface, impl, plugin, blurb )             \
  {                                                                  \
    auto fact = vpm.add_factory< interface, impl >( plugin );        \
    fact->add_attribute( kvpf::PLUGIN_NAME, plugin )                 \
      .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )        \
      .add_attribute( kvpf::PLUGIN_DESCRIPTION, blurb );             \
  }

  VIAME_REGISTER( kv::algo::detected_object_set_input,
                  kwiver::arrows::core::detected_object_set_input_kw18,
                  "kw18", "Read detected object sets in kw18 format" )

  VIAME_REGISTER( kv::algo::detected_object_set_output,
                  kwiver::arrows::core::detected_object_set_output_kw18,
                  "kw18", "Write detected object sets in kw18 format" )

  VIAME_REGISTER( kv::algo::read_object_track_set,
                  kwiver::arrows::core::read_object_track_set_kw18,
                  "kw18", "Read object track sets in kw18 format" )

  VIAME_REGISTER( kv::algo::write_object_track_set,
                  kwiver::arrows::core::write_object_track_set_kw18,
                  "kw18", "Write object track sets in kw18 format" )

  VIAME_REGISTER( kv::algo::feature_descriptor_io,
                  kwiver::arrows::core::feature_descriptor_io,
                  "core", "Read and write features and descriptors with cereal" )

#undef VIAME_REGISTER

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
