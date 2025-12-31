/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_DARKNET_TRAINER_H
#define VIAME_DARKNET_TRAINER_H

#include "viame_darknet_export.h"

#include <vital/algo/train_detector.h>

namespace viame {

class VIAME_DARKNET_EXPORT darknet_trainer :
  public kwiver::vital::algo::train_detector
{
public:
  darknet_trainer();
  virtual ~darknet_trainer();

  PLUGIN_INFO( "darknet",
               "Training utility for darknet." )

  virtual kwiver::vital::config_block_sptr get_configuration() const;

  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual void add_data_from_disk(
    kwiver::vital::category_hierarchy_sptr object_labels,
    std::vector< std::string > train_image_names,
    std::vector< kwiver::vital::detected_object_set_sptr > train_groundtruth,
    std::vector< std::string > test_image_names,
    std::vector< kwiver::vital::detected_object_set_sptr > test_groundtruth );

  virtual void add_data_from_memory(
    kwiver::vital::category_hierarchy_sptr object_labels,
    std::vector< kwiver::vital::image_container_sptr > train_images,
    std::vector< kwiver::vital::detected_object_set_sptr > train_groundtruth,
    std::vector< kwiver::vital::image_container_sptr > test_images,
    std::vector< kwiver::vital::detected_object_set_sptr > test_groundtruth );

  virtual void update_model();

private:
  class priv;
  const std::unique_ptr< priv > d;
};

} // end namespace

#endif /* VIAME_DARKNET_TRAINER_H */
