// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief train_detector algorithm definition

#ifndef VITAL_ALGO_TRAIN_DETECTOR_H_
#define VITAL_ALGO_TRAIN_DETECTOR_H_

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/core_types/category_hierarchy.h>
#include <viame/core_types/detected_object_set.h>
#include <viame/core_types/image_container.h>
#include <viame/algorithm_framework/vital_config.h>

#include <map>
#include <string>

namespace kwiver {

namespace vital {

namespace algo {

/// An abstract base class for training object detectors
class VITAL_ALGO_EXPORT train_detector
  : public kwiver::vital::algorithm
{
public:
  train_detector();
  PLUGGABLE_INTERFACE( train_detector );
  /// Add training data from disk
  ///
  /// This varient is geared towards offline training.
  ///
  /// \param object_labels object category labels for training
  /// \param train_image_list list of train image filenames
  /// \param train_groundtruth annotations loaded for each image
  /// \param test_image_list list of test image filenames
  /// \param test_groundtruth annotations loaded for each image
  virtual void
  add_data_from_disk(
    vital::category_hierarchy_sptr object_labels,
    std::vector< std::string > train_image_names,
    std::vector< kwiver::vital::detected_object_set_sptr > train_groundtruth,
    std::vector< std::string > test_image_names = std::vector< std::string >( ),
    std::vector< kwiver::vital::detected_object_set_sptr > test_groundtruth
    = std::vector< kwiver::vital::detected_object_set_sptr >( ) );

  /// Add training data from memory
  ///
  /// This varient is geared towards online training, and is not required
  /// to be defined.
  ///
  /// \throws runtime_exception if not defined.
  ///
  /// \param object_labels object category labels for training
  /// \param train_images vector of input train images
  /// \param train_groundtruth annotations loaded for each train image
  /// \param test_images optional vector of input test images
  /// \param test_groundtruth optional annotations loaded for each test image
  virtual void
  add_data_from_memory(
    vital::category_hierarchy_sptr object_labels,
    std::vector< kwiver::vital::image_container_sptr > train_images,
    std::vector< kwiver::vital::detected_object_set_sptr > train_groundtruth,
    std::vector< kwiver::vital::image_container_sptr > test_images
    = std::vector< kwiver::vital::image_container_sptr >( ),
    std::vector< kwiver::vital::detected_object_set_sptr > test_groundtruth
    = std::vector< kwiver::vital::detected_object_set_sptr >( ) );

  /// Train a detection model given all loaded data
  ///
  /// This varient is geared towards either offline or online training
  /// depending on the implementation.
  ///
  /// \throws runtime_exception if not defined or there's a data issue.
  ///
  /// \returns Map containing locations of final model files or other
  ///          general configuration parameters for model inference.
  virtual std::map< std::string, std::string > update_model() = 0;
};

/// Shared pointer for train_detector algorithm definition class
typedef std::shared_ptr< train_detector > train_detector_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif // VITAL_ALGO_TRAIN_DETECTOR_H_
