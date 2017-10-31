/*ckwg +29
* Copyright 2013-2017 by Kitware, Inc.
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
* \brief Header file for an abstract \link kwiver::vital::track_set track_set
*        \endlink and a concrete \link kwiver::vital::simple_track_set
*        simple_track_set \endlink
*/

#ifndef VITAL_KEYFRAME_DATA_H_
#define VITAL_KEYFRAME_DATA_H_

#include "track.h"

#include <vital/vital_export.h>
#include <vital/vital_config.h>
#include <vital/vital_types.h>

#include <vector>
#include <set>
#include <memory>

namespace kwiver {
  namespace vital {



    class keyframe_data;
    /// Shared pointer for base keyframe_data type
    typedef std::shared_ptr< keyframe_data > keyframe_data_sptr;

    /// A collection of tracks
    /**
    * This class dispatches everything to an implementation class as in the
    * bridge design pattern.  This pattern allows multiple back end implementations that
    * store and index track data in different ways.  Each back end can be combined with
    * any of the derived track_set types like feature_track_set and object_track_set.
    */
    class VITAL_EXPORT keyframe_data
    {
    public: 
      keyframe_data();

      ~keyframe_data();

      virtual bool is_keyframe(frame_id_t frame) = 0;
    };
  }
} // end namespace vital

#endif // VITAL_KEYFRAME_DATA_H_
