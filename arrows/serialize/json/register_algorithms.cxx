/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file
 * \brief Defaults plugin algorithm registration interface impl
 */

#include <arrows/serialize/json/kwiver_serialize_json_plugin_export.h>
#include <vital/algo/algorithm_factory.h>

#include "activity.h"
#include "activity_type.h"
#include "bounding_box.h"
#include "detected_object.h"
#include "detected_object_type.h"
#include "detected_object_set.h"
#include "timestamp.h"
#include "image.h"
#include "string.h"
#include "track_state.h"
#include "object_track_state.h"
#include "track.h"
#include "track_set.h"
#include "object_track_set.h"

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
  kwiver::vital::serializer_registrar reg( vpm, "arrows.serialize.json",
                                           "json" );

  if (reg.is_module_loaded())
  {
    return;
  }

  using namespace kwiver::arrows::serialize::json;

  reg.register_algorithm< activity >();
  reg.register_algorithm< activity_type >();
  reg.register_algorithm< bounding_box >();
  reg.register_algorithm< detected_object >();
  reg.register_algorithm< detected_object_type >();
  reg.register_algorithm< detected_object_set >();
  reg.register_algorithm< timestamp >();
  reg.register_algorithm< image >();
  reg.register_algorithm< image >( "kwiver:mask" );
  reg.register_algorithm< string >();
  reg.register_algorithm< track_state >();
  reg.register_algorithm< object_track_state >();
  reg.register_algorithm< track >();
  reg.register_algorithm< track_set >();
  reg.register_algorithm< object_track_set >();
  reg.register_algorithm< string >( "kwiver:file_name" );
  reg.register_algorithm< string >( "kwiver:image_name" );
  reg.register_algorithm< string >( "kwiver:video_name" );

  reg.mark_module_as_loaded();
}

} // end namespace json
} // end namespace serialize
} // end namespace arrows
} // end namespace kwiver
