/*ckwg +29
 * Copyright 2017-2019 by Kitware, Inc.
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
 * \brief train_detector algorithm definition
 */

#ifndef VITAL_ALGO_TRAIN_DETECTOR_H_
#define VITAL_ALGO_TRAIN_DETECTOR_H_

#include <vital/vital_config.h>
#include <vital/algo/algorithm.h>
#include <vital/types/category_hierarchy.h>
#include <vital/types/image_container.h>
#include <vital/types/detected_object_set.h>

namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class for training object detectors
class VITAL_ALGO_EXPORT train_detector
  : public kwiver::vital::algorithm_def<train_detector>
{
public:
  /// Return the name of this algorithm
  static std::string static_type_name() { return "train_detector"; }

  /// Train a detection model given a list of images and detections
  /**
   * This varient is geared towards offline training.
   *
   * \param object_labels object category labels for training
   * \param train_image_list list of train image filenames
   * \param train_groundtruth annotations loaded for each image
   * \param test_image_list list of test image filenames
   * \param test_groundtruth annotations loaded for each image
   */
  virtual void
  add_data_from_disk(vital::category_hierarchy_sptr object_labels,
    std::vector< std::string > train_image_names,
    std::vector< kwiver::vital::detected_object_set_sptr > train_groundtruth,
    std::vector< std::string > test_image_names = std::vector< std::string >(),
    std::vector< kwiver::vital::detected_object_set_sptr > test_groundtruth
     = std::vector< kwiver::vital::detected_object_set_sptr >());

  /// Train a detection model given images and detections
  /**
   * This varient is geared towards online training, and is not required
   * to be defined.
   *
   * \throws runtime_exception if not defined.
   *
   * \param object_labels object category labels for training
   * \param train_images vector of input train images
   * \param train_groundtruth annotations loaded for each train image
   * \param test_images optional vector of input test images
   * \param test_groundtruth optional annotations loaded for each test image
   */
  virtual void
  add_data_from_memory(vital::category_hierarchy_sptr object_labels,
    std::vector< kwiver::vital::image_container_sptr > train_images,
    std::vector< kwiver::vital::detected_object_set_sptr > train_groundtruth,
    std::vector< kwiver::vital::image_container_sptr > test_images
      = std::vector< kwiver::vital::image_container_sptr >(),
    std::vector< kwiver::vital::detected_object_set_sptr > test_groundtruth
      = std::vector< kwiver::vital::detected_object_set_sptr >());


  /// Train a detection model given all loaded data
  /**
   * This varient is geared towards either offline or online training
   * depending on the implementation.
   *
   * \throws runtime_exception if not defined or there's a data issue.
   *
   * \param object_labels object category labels for training
   */
  virtual void update_model() = 0;

protected:
  train_detector();
};


/// Shared pointer for train_detector algorithm definition class
typedef std::shared_ptr<train_detector> train_detector_sptr;


} } } // end namespace

#endif // VITAL_ALGO_TRAIN_DETECTOR_H_
