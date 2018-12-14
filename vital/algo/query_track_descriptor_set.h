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
 * \brief Header file for \link kwiver::vital::algo::query_track_descriptor_set
 *        query_track_descriptor_set \endlink
 */

#ifndef VITAL_QUERY_TRACK_DESCRIPTOR_SET_H_
#define VITAL_QUERY_TRACK_DESCRIPTOR_SET_H_

#include <vital/algo/algorithm.h>
#include <vital/vital_export.h>

#include <vital/types/track_descriptor.h>
#include <vital/types/track.h>

namespace kwiver {
namespace vital {
namespace algo {


// ------------------------------------------------------------------
/// Abstract interface for a collection of track descriptors that can be queried
class VITAL_ALGO_EXPORT query_track_descriptor_set
  : public kwiver::vital::algorithm_def<query_track_descriptor_set>
{
public:
  /// Return the name of this algorithm
  static std::string static_type_name() { return "query_track_descriptor_set"; }

  /// Tuple containing video name, descriptor, and tracks
  typedef std::tuple< std::string,
                      vital::track_descriptor_sptr,
                      std::vector< vital::track_sptr > > desc_tuple_t;

  /// Destructor
  virtual ~query_track_descriptor_set() = default;

  /// Set option to use object tracks for track descriptor history
  virtual void use_tracks_for_history( bool value ) = 0;

  /// Get a track descriptor by UID
  virtual bool get_track_descriptor( std::string const& uid,
    desc_tuple_t& result ) = 0;
};

/// Shared pointer for base queryable_track_set type
typedef std::shared_ptr< query_track_descriptor_set >
query_track_descriptor_set_sptr;


} } } // end namespace algo

#endif // VITAL_QUERY_TRACK_DESCRIPTOR_SET_H_
