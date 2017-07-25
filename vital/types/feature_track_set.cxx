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
 * \brief Implementation of \link kwiver::vital::track_set track_set \endlink
 *        member functions
 */

#include "feature_track_set.h"

#include <limits>

#include <vital/vital_foreach.h>

namespace kwiver {
namespace vital {


/// Return the set of features in tracks on the last frame
feature_set_sptr
feature_track_set
::last_frame_features() const
{
  return frame_features(this->last_frame());
}


/// Return the set of descriptors in tracks on the last frame
descriptor_set_sptr
feature_track_set
::last_frame_descriptors() const
{
  return frame_descriptors(this->last_frame());
}


/// Return the set of features in all tracks for the given frame.
feature_set_sptr
feature_track_set
::frame_features( frame_id_t offset ) const
{
  std::vector<feature_sptr> features;
  std::vector<track_state_sptr> fsd = this->frame_states(offset);
  for( auto const data : fsd )
  {
    feature_sptr f = nullptr;
    auto fdata = std::dynamic_pointer_cast<feature_track_state>(data);
    if( fdata )
    {
      f = fdata->feature;
    }
    features.push_back(f);
  }
  return feature_set_sptr(new simple_feature_set(features));
}


/// Return the set of descriptors in all tracks for the given frame.
descriptor_set_sptr
feature_track_set
::frame_descriptors(frame_id_t offset) const
{
  std::vector<descriptor_sptr> descriptors;
  std::vector<track_state_sptr> fsd = this->frame_states(offset);
  for( auto const data : fsd )
  {
    descriptor_sptr d = nullptr;
    auto fdata = std::dynamic_pointer_cast<feature_track_state>(data);
    if( fdata )
    {
      d = fdata->descriptor;
    }
    descriptors.push_back(d);
  }

  return descriptor_set_sptr(new simple_descriptor_set(descriptors));
}


} } // end namespace vital
