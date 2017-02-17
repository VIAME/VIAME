/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 */

#ifndef VITAL_ALGO_DETECTED_OBJECT_FILTER_H_
#define VITAL_ALGO_DETECTED_OBJECT_FILTER_H_

#include <vital/vital_config.h>
#include <vital/algo/algorithm.h>
#include <vital/types/image_container.h>
#include <vital/types/detected_object_set.h>

#include <vector>

namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class for filtering sets of detected objects
// ----------------------------------------------------------------
/**
 * A detected object filter accepts a set of detections and produces
 * another set of detections. The output set may be different from the
 * input set. It all depends on the actual implementation. In any
 * case, the input detection set shall be unmodified.
 */
class VITAL_ALGO_EXPORT detected_object_filter
  : public algorithm_def<detected_object_filter>
{
public:
  /// Return the name of this algorithm
  static std::string static_type_name() { return "detected_object_filter"; }

  /// Filter set of detected objects.
  /**
   * This method applies a filter to the input set to create an output
   * set. The input set of detections is unmodified.
   *
   * \param input_set Set of detections to be filtered.
   * \returns Filtered set of detections.
   */
  virtual detected_object_set_sptr
      filter( const detected_object_set_sptr input_set) const = 0;

protected:
  detected_object_filter();
};

/// Shared pointer for generic detected_object_filter definition type.
typedef std::shared_ptr<detected_object_filter> detected_object_filter_sptr;

} } } // end namespace

#endif
