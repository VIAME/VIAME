// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Default plugin algorithm registration interface implementation.

#include <arrows/serialize/json/kwiver_serialize_json_plugin_export.h>

#include <vital/algo/algorithm_factory.h>

#include "activity.h"
#include "activity_type.h"
#include "bounding_box.h"
#include "detected_object.h"
#include "detected_object_set.h"
#include "detected_object_type.h"
#include "image.h"
#include "metadata_map_io.h"
#include "object_track_set.h"
#include "object_track_state.h"
#include "string.h"
#include "timestamp.h"
#include "track.h"
#include "track_set.h"
#include "track_state.h"

namespace kwiver {

namespace arrows {

namespace serialize {

namespace json {

// ----------------------------------------------------------------------------
extern "C"
KWIVER_SERIALIZE_JSON_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  auto const module_name = std::string{ "arrows.serialize.json" };
  kwiver::vital::serializer_registrar sreg( vpm, module_name, "json" );
  kwiver::vital::algorithm_registrar areg( vpm, module_name );

  if( sreg.is_module_loaded() )
  {
    return;
  }

  using namespace kwiver::arrows::serialize::json;

  sreg.register_algorithm< activity >();
  sreg.register_algorithm< activity_type >();
  sreg.register_algorithm< bounding_box >();
  sreg.register_algorithm< detected_object >();
  sreg.register_algorithm< detected_object_set >();
  sreg.register_algorithm< detected_object_type >();
  sreg.register_algorithm< timestamp >();
  sreg.register_algorithm< image >();
  sreg.register_algorithm< image >( "kwiver:mask" );
  sreg.register_algorithm< string >();
  sreg.register_algorithm< track_state >();
  sreg.register_algorithm< object_track_state >();
  sreg.register_algorithm< track >();
  sreg.register_algorithm< track_set >();
  sreg.register_algorithm< object_track_set >();
  sreg.register_algorithm< string >( "kwiver:file_name" );
  sreg.register_algorithm< string >( "kwiver:image_name" );
  sreg.register_algorithm< string >( "kwiver:video_name" );

  areg.register_algorithm< metadata_map_io >();

  sreg.mark_module_as_loaded();
}

} // end namespace json

} // end namespace serialize

} // end namespace arrows

} // end namespace kwiver
