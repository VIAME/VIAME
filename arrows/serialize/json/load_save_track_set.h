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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
