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
 *
 * VIAME's own readers and writers -- viame_csv, dive, cvat, habcam and the
 * rest -- came from `plugins/core` in P2-T04. They are declared with
 * PLUGGABLE_IMPL and name and describe themselves, so they register through
 * the template rather than the macro.
 */

#include "viame_file_io_plugin_export.h"

#include <viame/algorithm_framework/algo/detected_object_set_input.h>
#include <viame/algorithm_framework/algo/detected_object_set_output.h>
#include <viame/algorithm_framework/algo/feature_descriptor_io.h>
#include <viame/algorithm_framework/algo/read_object_track_set.h>
#include <viame/algorithm_framework/algo/transform_2d_io.h>
#include <viame/algorithm_framework/algo/write_object_track_set.h>

#include <viame/algorithm_framework/plugin/registry.h>

#include "auto_detect_transform.h"
#include "detected_object_set_input_kw18.h"
#include "detected_object_set_output_kw18.h"
#include "feature_descriptor_io.h"
#include "read_detected_object_set_auto.h"
#include "read_detected_object_set_cvat.h"
#include "read_detected_object_set_dive.h"
#include "read_detected_object_set_fishnet.h"
#include "read_detected_object_set_habcam.h"
#include "read_detected_object_set_oceaneyes.h"
#include "read_detected_object_set_viame_csv.h"
#include "read_detected_object_set_yolo.h"
#include "read_object_track_set_auto.h"
#include "read_object_track_set_dive.h"
#include "read_object_track_set_kw18.h"
#include "read_object_track_set_viame_csv.h"
#include "read_transform_homography_json.h"
#include "write_detected_object_set_dive.h"
#include "write_detected_object_set_viame_csv.h"
#include "write_object_track_set_dive.h"
#include "write_object_track_set_kw18.h"
#include "write_object_track_set_viame_csv.h"

namespace viame {

namespace kv = kwiver::vital;

namespace {

// An algorithm declared with PLUGGABLE_IMPL, which names and describes
// itself.
template < typename interface_t, typename algorithm_t >
void register_algorithm( kv::registry& vpm, std::string const& module_name )
{
  using kvpf = kv::plugin_factory;

  auto fact = vpm.add_factory< interface_t, algorithm_t >(
    algorithm_t::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_NAME, algorithm_t::plugin_name() )
    .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kvpf::PLUGIN_DESCRIPTION,
                    algorithm_t::plugin_description() );
}

}

extern "C"
VIAME_FILE_IO_PLUGIN_EXPORT
void
register_factories( kv::registry& vpm )
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
                  "core", "Read and write features and descriptors as KWFD" )

#undef VIAME_REGISTER

  register_algorithm< kv::algo::transform_2d_io,
    auto_detect_transform_io >( vpm, module_name );
  register_algorithm< kv::algo::detected_object_set_input,
    read_detected_object_set_auto >( vpm, module_name );
  register_algorithm< kv::algo::detected_object_set_input,
    read_detected_object_set_cvat >( vpm, module_name );
  register_algorithm< kv::algo::detected_object_set_input,
    read_detected_object_set_dive >( vpm, module_name );
  register_algorithm< kv::algo::detected_object_set_input,
    read_detected_object_set_fishnet >( vpm, module_name );
  register_algorithm< kv::algo::detected_object_set_input,
    read_detected_object_set_habcam >( vpm, module_name );
  register_algorithm< kv::algo::detected_object_set_input,
    read_detected_object_set_oceaneyes >( vpm, module_name );
  register_algorithm< kv::algo::detected_object_set_input,
    read_detected_object_set_viame_csv >( vpm, module_name );
  register_algorithm< kv::algo::detected_object_set_input,
    read_detected_object_set_yolo >( vpm, module_name );
  register_algorithm< kv::algo::read_object_track_set,
    read_object_track_set_auto >( vpm, module_name );
  register_algorithm< kv::algo::read_object_track_set,
    read_object_track_set_dive >( vpm, module_name );
  register_algorithm< kv::algo::read_object_track_set,
    read_object_track_set_viame_csv >( vpm, module_name );
  register_algorithm< kv::algo::transform_2d_io,
    read_transform_homography_json >( vpm, module_name );
  register_algorithm< kv::algo::detected_object_set_output,
    write_detected_object_set_dive >( vpm, module_name );
  register_algorithm< kv::algo::detected_object_set_output,
    write_detected_object_set_viame_csv >( vpm, module_name );
  register_algorithm< kv::algo::write_object_track_set,
    write_object_track_set_dive >( vpm, module_name );
  register_algorithm< kv::algo::write_object_track_set,
    write_object_track_set_viame_csv >( vpm, module_name );

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
