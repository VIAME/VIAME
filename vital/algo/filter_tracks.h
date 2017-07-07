/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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

#ifndef VITAL_ALGO_FILTER_TRACKS_H_
#define VITAL_ALGO_FILTER_TRACKS_H_

#include <vital/vital_config.h>

#include <utility>
#include <vector>
#include <memory>

#include <vital/algo/algorithm.h>
#include <vital/types/track_set.h>

/**
 * \file
 * \brief Header defining abstract \link kwiver::vital::algo::filter_tracks
 *        filter tracks \endlink algorithm
 */

namespace kwiver {
namespace vital {
namespace algo {

/// \brief Abstract base class for track set filter algorithms.
class VITAL_ALGO_EXPORT filter_tracks
  : public kwiver::vital::algorithm_def<filter_tracks>
{
public:

  /// Return the name of this algorithm.
  static std::string static_type_name() { return "filter_tracks"; }

  /// Filter a track set and return a subset of the tracks
  /**
   * \param [in] input The track set to filter
   * \returns a filtered version of the track set (simple_track_set)
   */
  virtual kwiver::vital::track_set_sptr
  filter( kwiver::vital::track_set_sptr input ) const = 0;

protected:
  filter_tracks();

};

/// type definition for shared pointer to a filter_tracks algorithm
typedef std::shared_ptr<filter_tracks> filter_tracks_sptr;

} } } // end namespace

#endif // VITAL_ALGO_FILTER_TRACKS_H_
