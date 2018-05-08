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

/**
 * \file
 * \brief compute_association_matrix algorithm definition
 */

#ifndef VITAL_ALGO_COMPUTE_ASSOCIATION_MATRIX_H_
#define VITAL_ALGO_COMPUTE_ASSOCIATION_MATRIX_H_

#include <vital/vital_config.h>
#include <vital/algo/algorithm.h>

#include <vital/types/timestamp.h>
#include <vital/types/object_track_set.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/image_container.h>
#include <vital/types/matrix.h>

namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class for computing association cost matrices for tracking
class VITAL_ALGO_EXPORT compute_association_matrix
  : public kwiver::vital::algorithm_def<compute_association_matrix>
{
public:
  /// Return the name of this algorithm
  static std::string static_type_name() { return "compute_association_matrix"; }

  /// Compute an association matrix given detections and tracks
  /**
   * \param ts frame ID
   * \param image contains the input image for the current frame
   * \param tracks active track set from the last frame
   * \param detections input detected object sets from the current frame
   * \param matrix output matrix
   * \param considered output detections used in matrix
   * \return returns whether a matrix was successfully computed
   */
  virtual bool
  compute( kwiver::vital::timestamp ts,
           kwiver::vital::image_container_sptr image,
           kwiver::vital::object_track_set_sptr tracks,
           kwiver::vital::detected_object_set_sptr detections,
           kwiver::vital::matrix_d& matrix,
           kwiver::vital::detected_object_set_sptr& considered ) const = 0;

protected:
  compute_association_matrix();

};


/// Shared pointer for compute_association_matrix algorithm definition class
typedef std::shared_ptr<compute_association_matrix> compute_association_matrix_sptr;


} } } // end namespace

#endif // VITAL_ALGO_COMPUTE_ASSOCIATION_MATRIX_MAP_H_
