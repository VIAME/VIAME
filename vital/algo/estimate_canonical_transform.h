/*ckwg +29
 * Copyright 2013-2016 by Kitware, Inc.
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
 * \brief Definition for canonical transform estimation algorithm
 */

#ifndef VITAL_ALGO_ESTIMATE_CANONICAL_TRANSFORM_H_
#define VITAL_ALGO_ESTIMATE_CANONICAL_TRANSFORM_H_

#include <vital/vital_config.h>

#include <string>
#include <vector>

#include <vital/algo/algorithm.h>
#include <vital/types/camera_map.h>
#include <vital/types/landmark_map.h>
#include <vital/types/similarity.h>
#include <vital/types/vector.h>

#include <vital/config/config_block.h>

namespace kwiver {
namespace vital {
namespace algo {

/// Algorithm for estimating a canonical transform for cameras and landmarks
/**
 *  A canonical transform is a repeatable transformation that can be recovered
 *  from data.  In this case we assume at most a similarity transformation.
 *  If data sets P1 and P2 are equivalent up to a similarity transformation,
 *  then applying a canonical transform to P1 and separately a
 *  canonical transform to P2 should bring the data into the same coordinates.
 */
class VITAL_EXPORT estimate_canonical_transform
  : public kwiver::vital::algorithm_def<estimate_canonical_transform>
{
public:
  /// Name of this algo definition
  static std::string static_type_name() { return "estimate_canonical_transform"; }

  /// Estimate a canonical similarity transform for cameras and points
  /**
   * \param cameras The camera map containing all the cameras
   * \param landmarks The landmark map containing all the 3D landmarks
   * \throws algorithm_exception When the data is insufficient or degenerate.
   * \returns An estimated similarity transform mapping the data to the
   *          canonical space.
   * \note This algorithm does not apply the transformation, it only estimates it.
   */
  virtual kwiver::vital::similarity_d
  estimate_transform(kwiver::vital::camera_map_sptr const cameras,
                     kwiver::vital::landmark_map_sptr const landmarks) const=0;

protected:
    estimate_canonical_transform();

};


/// Shared pointer for similarity transformation algorithms
typedef std::shared_ptr<estimate_canonical_transform> estimate_canonical_transform_sptr;


} } } // end namespace

#endif // VITAL_ALGO_ESTIMATE_CANONICAL_TRANSFORM_H_
