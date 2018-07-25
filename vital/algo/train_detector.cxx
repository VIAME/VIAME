/*ckwg +29
 * Copyright 2017-2018 by Kitware, Inc.
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
 * \brief train_detector algorithm definition instantiation
 */

#include "train_detector.h"

#include <vital/algo/algorithm.txx>

namespace kwiver {
namespace vital {
namespace algo {

train_detector
::train_detector()
{
  attach_logger( "algo.train_detector" );
}

void
train_detector
::train_from_disk(
  vital::category_hierarchy_sptr object_labels,
  std::vector< std::string > train_image_names,
  std::vector< kwiver::vital::detected_object_set_sptr > train_groundtruth,
  std::vector< std::string > test_image_names,
  std::vector< kwiver::vital::detected_object_set_sptr > test_groundtruth)
{
  throw std::runtime_error( "Method not implemented" );
}


void
train_detector
::train_from_memory(
  vital::category_hierarchy_sptr object_labels,
  std::vector< kwiver::vital::image_container_sptr > train_images,
  std::vector< kwiver::vital::detected_object_set_sptr > train_groundtruth,
  std::vector< kwiver::vital::image_container_sptr > test_images,
  std::vector< kwiver::vital::detected_object_set_sptr > test_groundtruth)
{
  throw std::runtime_error( "Method not implemented" );
}

} } }

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::train_detector);
/// \endcond
