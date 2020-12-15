/*ckwg +29
 * Copyright 2020 by Kitware, Inc.
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
 * \brief Header defining abstract activity detector
 */

#ifndef VITAL_ALGO_ACTIVITY_DETECTOR_H_
#define VITAL_ALGO_ACTIVITY_DETECTOR_H_

#include <vital/algo/algorithm.h>
#include <vital/types/image_container.h>
#include <vital/types/activity.h>

#include <vector>


namespace kwiver {
namespace vital {
namespace algo {


// ----------------------------------------------------------------
/**
 * @brief activity detector base class/
 *
 */
class VITAL_ALGO_EXPORT activity_detector
: public algorithm_def<activity_detector>
{
public:
  /// Return the name of this algorithm
  static std::string static_type_name() { return "activity_detector"; }

  virtual std::vector<kwiver::vital::activity>
      detect( image_container_sptr image) const = 0;

protected:
  activity_detector();
};

/// Shared pointer for generic activity_detector definition type.
typedef std::shared_ptr<activity_detector> activity_detector_sptr;

} } } // end namespace

#endif
