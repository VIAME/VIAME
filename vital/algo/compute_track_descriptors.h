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
 * \brief compute_track_descriptors algorithm definition
 */

#ifndef VITAL_ALGO_COMPUTE_TRACK_DESCRIPTORS_H_
#define VITAL_ALGO_COMPUTE_TRACK_DESCRIPTORS_H_

#include <vital/vital_config.h>

#include <vital/algo/algorithm.h>
#include <vital/types/image_container.h>
#include <vital/types/descriptor_set.h>
#include <vital/types/track_set.h>

namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class for extracting feature descriptors
class VITAL_ALGO_EXPORT compute_track_descriptors
  : public kwiver::vital::algorithm_def<compute_track_descriptors>
{
public:
  /// Return the name of this algorithm
  static std::string static_type_name() { return "compute_track_descriptors"; }

  /// Extract from the image a descriptor corresoponding to each feature
  /**
   * \param image_data contains the image data to process
   * \param features the feature locations at which descriptors are extracted
   * \param image_mask Mask image of the same dimensions as \p image_data where
   *                   positive values indicate regions of \p image_data to
   *                   consider.
   * \returns a set of feature descriptors
   */
  virtual kwiver::vital::descriptor_set_sptr
  extract( kwiver::vital::image_container_sptr image_data,
           kwiver::vital::track_set_sptr tracks ) const = 0;

protected:
  compute_track_descriptors();

};


/// Shared pointer for base compute_track_descriptors algorithm definition class
typedef std::shared_ptr<compute_track_descriptors> compute_track_descriptors_sptr;


} } } // end namespace

#endif // VITAL_ALGO_COMPUTE_TRACK_DESCRIPTORS_H_
