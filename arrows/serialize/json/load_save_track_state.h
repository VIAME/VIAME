// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file: Contains the private internal implementation interface for
 * serializing tracks
 */

#ifndef SERIAL_JSON_LOAD_SAVE_TRACK_STATE_H
#define SERIAL_JSON_LOAD_SAVE_TRACK_STATE_H

#include <vital/types/track.h>
#include <vital/types/object_track_set.h>

#include <vital/internal/cereal/types/polymorphic.hpp>

CEREAL_REGISTER_TYPE(kwiver::vital::track_state);
CEREAL_REGISTER_TYPE(kwiver::vital::object_track_state);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kwiver::vital::track_state,
                                     kwiver::vital::object_track_state);

namespace kwiver {
namespace vital {
  class track_state;
  class object_track_state;
} } // end namespace

namespace cereal {
  class JSONOutputArchive;
  class JSONInputArchive;

KWIVER_SERIALIZE_JSON_EXPORT
void save( cereal::JSONOutputArchive& archive, const kwiver::vital::track_state& trk_state );
KWIVER_SERIALIZE_JSON_EXPORT
void load( cereal::JSONInputArchive& archive, kwiver::vital::track_state& trk_state );

KWIVER_SERIALIZE_JSON_EXPORT
void save( cereal::JSONOutputArchive& archive,
            const kwiver::vital::object_track_state& obj_trk_state );
KWIVER_SERIALIZE_JSON_EXPORT
void load( cereal::JSONInputArchive& archive,
            kwiver::vital::object_track_state& obj_trk_state );

}

#endif // SERIAL_JSON_LOAD_SAVE_TRACK_STATE_H
