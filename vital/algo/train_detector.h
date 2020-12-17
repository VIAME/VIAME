// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
  train_from_disk(vital::category_hierarchy_sptr object_labels,
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
  train_from_memory(vital::category_hierarchy_sptr object_labels,
    std::vector< kwiver::vital::image_container_sptr > train_images,
    std::vector< kwiver::vital::detected_object_set_sptr > train_groundtruth,
    std::vector< kwiver::vital::image_container_sptr > test_images
      = std::vector< kwiver::vital::image_container_sptr >(),
    std::vector< kwiver::vital::detected_object_set_sptr > test_groundtruth
      = std::vector< kwiver::vital::detected_object_set_sptr >());

protected:
  train_detector();

};

/// Shared pointer for train_detector algorithm definition class
typedef std::shared_ptr<train_detector> train_detector_sptr;

} } } // end namespace

#endif // VITAL_ALGO_TRAIN_DETECTOR_H_
