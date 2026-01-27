/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_WINDOWED_TRAINER_H
#define VIAME_CORE_WINDOWED_TRAINER_H

#include "viame_core_export.h"

#include <vital/algo/train_detector.h>

#include <map>
#include <string>

namespace viame {

// -----------------------------------------------------------------------------
/**
 * @brief Run training on arbitrary other trainers in a windowed fashion
 *
 * This algorithm wraps another trainer and pre-processes training images
 * by breaking them into smaller windows/chips before passing to the
 * underlying trainer.
 *
 * This is a pure vital::image implementation with no OpenCV dependency.
 */
class VIAME_CORE_EXPORT windowed_trainer
  : public kwiver::vital::algorithm_impl< windowed_trainer,
      kwiver::vital::algo::train_detector >
{
public:

  PLUGIN_INFO( "windowed",
               "Window some other arbitrary detector trainer across the image (no OpenCV)" )

  windowed_trainer();
  virtual ~windowed_trainer();

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

#endif /* VIAME_CORE_WINDOWED_TRAINER_H */
