/*ckwg +29
 * Copyright 2013-2015 by Kitware, Inc.
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
 * \brief detect_features algorithm definition
 */

#ifndef VITAL_ALGO_DETECT_LOOPS_H_
#define VITAL_ALGO_DETECT_LOOPS_H_

#include <vital/vital_config.h>
#include <vital/algo/algorithm.h>
#include <vital/types/feature_track_set.h>
#include <vital/types/image_container.h>

namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class for detecting loops in image sets
class VITAL_ALGO_EXPORT detect_loops
  : public kwiver::vital::algorithm_def<detect_loops>
{
public:
  /// Return the name of this algorithm
  static std::string static_type_name() { return "detect_loops"; }

  /// Find loops in a feature_track_set
  /**
   * \param[in] feat_tracks set of feature tracks on which to detect loops
   * \param[in] frame_number frame to detect loops with
   * \returns a feature track set with any found loops included
   */
  virtual kwiver::vital::feature_track_set_sptr
  detect(kwiver::vital::feature_track_set_sptr feat_tracks,
         frame_id_t frame_number) = 0;

  virtual ~detect_loops() {};

protected:
  detect_loops();

};


/// Shared pointer for detect_features algorithm definition class
typedef std::shared_ptr<detect_loops> detect_loops_sptr;


} } } // end namespace

#endif // VITAL_ALGO_DETECT_LOOPS_H_
