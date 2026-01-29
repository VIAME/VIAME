/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Header for train_detector_svm algorithm
 */

#ifndef VIAME_SVM_TRAIN_DETECTOR_SVM_H
#define VIAME_SVM_TRAIN_DETECTOR_SVM_H

#include "viame_svm_export.h"

#include <vital/algo/train_detector.h>
#include "viame_algorithm_plugin_interface.h"

#include <map>
#include <memory>
#include <string>

namespace viame {

namespace kv = kwiver::vital;

/// Train SVM models for object detection using descriptor-based classification
///
/// This algorithm trains binary SVM classifiers for each object category using
/// libsvm. It reads descriptor vectors from a CSV index file and trains models
/// based on positive/negative label files.
class VIAME_SVM_EXPORT train_detector_svm
  : public kv::algo::train_detector
{
public:
  VIAME_ALGORITHM_PLUGIN_INTERFACE( train_detector_svm )

  PLUGIN_INFO( "svm",
               "Train SVM models for object detection" )

  /// Constructor
  train_detector_svm();

  /// Destructor
  virtual ~train_detector_svm();

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  virtual kv::config_block_sptr get_configuration() const;

  /// Set this algorithm's properties via a config block
  virtual void set_configuration( kv::config_block_sptr config );

  /// Check that the algorithm's configuration is valid
  virtual bool check_configuration( kv::config_block_sptr config ) const;

  /// Train a detection model given a list of images and detections
  virtual void
  add_data_from_disk(
    kv::category_hierarchy_sptr object_labels,
    std::vector< std::string > train_image_names,
    std::vector< kv::detected_object_set_sptr > train_groundtruth,
    std::vector< std::string > test_image_names = std::vector< std::string >(),
    std::vector< kv::detected_object_set_sptr > test_groundtruth
      = std::vector< kv::detected_object_set_sptr >() );

  /// Train all SVM models
  virtual std::map<std::string, std::string> update_model() override;

private:

  /// Private implementation class
  class priv;
  const std::unique_ptr< priv > d_;
};

} // end namespace viame

#endif // VIAME_SVM_TRAIN_DETECTOR_SVM_H
