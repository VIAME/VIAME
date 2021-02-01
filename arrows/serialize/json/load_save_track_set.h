// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file: Contains the private internal implementation interface for
 * serializing track set
 */

#ifndef SERIAL_JSON_LOAD_SAVE_TRACK_SET_H
#define SERIAL_JSON_LOAD_SAVE_TRACK_SET_H

#include <vital/types/track_set.h>
#include <vital/types/object_track_set.h>

#include <vital/internal/cereal/types/polymorphic.hpp>

CEREAL_REGISTER_TYPE(kwiver::vital::track_set);
CEREAL_REGISTER_TYPE(kwiver::vital::object_track_set);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kwiver::vital::track_set,
                                     kwiver::vital::object_track_set);

namespace kwiver {
namespace vital {
  class track_set;
  class object_track_set;
} } // end namespace

namespace cereal {
  class JSONOutputArchive;
  class JSONInputArchive;

KWIVER_SERIALIZE_JSON_EXPORT
void save( cereal::JSONOutputArchive& archive, const kwiver::vital::track_set& trk_set );
KWIVER_SERIALIZE_JSON_EXPORT
void load( cereal::JSONInputArchive& archive, kwiver::vital::track_set& trk_set );

KWIVER_SERIALIZE_JSON_EXPORT
void save( cereal::JSONOutputArchive& archive,
            const kwiver::vital::object_track_set& obj_trk_set );
KWIVER_SERIALIZE_JSON_EXPORT
void load( cereal::JSONInputArchive& archive,
            kwiver::vital::object_track_set& obj_trk_set );

}

#endif // SERIAL_JSON_LOAD_SAVE_TRACK_SET_H
