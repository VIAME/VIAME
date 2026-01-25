/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_ADAPTIVE_DETECTOR_TRAINER_H
#define VIAME_CORE_ADAPTIVE_DETECTOR_TRAINER_H

#include "viame_core_export.h"

#include <vital/algo/train_detector.h>

#include <map>
#include <string>

namespace viame {

// -----------------------------------------------------------------------------
/**
 * @brief Adaptive detector trainer that analyzes data and runs appropriate training pipelines
 *
 * This algorithm analyzes training data characteristics (annotation counts,
 * object sizes) and runs multiple configured training pipelines sequentially.
 * Each trainer can have hard requirements (must be met to run) and soft
 * preferences (used for ranking when multiple trainers qualify).
 *
 * Use cases:
 * - Automatically running SVM for small datasets, deep learning for large
 * - Running tiled/windowed training for small objects
 * - Training multiple models with different characteristics
 */
class VIAME_CORE_EXPORT adaptive_detector_trainer
  : public kwiver::vital::algorithm_impl< adaptive_detector_trainer,
      kwiver::vital::algo::train_detector >
{
public:

  PLUGIN_INFO( "adaptive",
               "Analyzes training data and runs appropriate training pipelines" )

  adaptive_detector_trainer();
  virtual ~adaptive_detector_trainer();

  virtual kwiver::vital::config_block_sptr get_configuration() const;

  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual void
  add_data_from_disk( kwiver::vital::category_hierarchy_sptr object_labels,
    std::vector< std::string > train_image_names,
    std::vector< kwiver::vital::detected_object_set_sptr > train_groundtruth,
    std::vector< std::string > test_image_names,
    std::vector< kwiver::vital::detected_object_set_sptr > test_groundtruth );

  virtual void
  add_data_from_memory( kwiver::vital::category_hierarchy_sptr object_labels,
    std::vector< kwiver::vital::image_container_sptr > train_images,
    std::vector< kwiver::vital::detected_object_set_sptr > train_groundtruth,
    std::vector< kwiver::vital::image_container_sptr > test_images,
    std::vector< kwiver::vital::detected_object_set_sptr > test_groundtruth );

  virtual std::map<std::string, std::string> update_model() override;

private:

  class priv;
  const std::unique_ptr< priv > d;
};

} // end namespace viame

#endif /* VIAME_CORE_ADAPTIVE_DETECTOR_TRAINER_H */
