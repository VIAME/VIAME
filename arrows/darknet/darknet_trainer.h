// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_DARKNET_TRAINER
#define KWIVER_ARROWS_DARKNET_TRAINER

#include <arrows/darknet/kwiver_algo_darknet_export.h>

#include <vital/algo/train_detector.h>

namespace kwiver {
namespace arrows {
namespace darknet {

// ----------------------------------------------------------------
/**
 * @brief Darknet Training Utility Class
 */
class KWIVER_ALGO_DARKNET_EXPORT darknet_trainer
  : public vital::algo::train_detector
{
public:
  darknet_trainer();
  virtual ~darknet_trainer();

  PLUGIN_INFO( "darknet",
               "Training utility for darknet." )

  vital::config_block_sptr get_configuration() const override;

  void set_configuration( vital::config_block_sptr config ) override;
  bool check_configuration( vital::config_block_sptr config ) const override;

  void train_from_disk( vital::category_hierarchy_sptr object_labels,
                        std::vector< std::string > train_image_names,
                        std::vector< vital::detected_object_set_sptr > train_groundtruth,
                        std::vector< std::string > test_image_names,
                        std::vector< vital::detected_object_set_sptr > test_groundtruth ) override;

private:
  class priv;

  const std::unique_ptr< priv > d;
};

} } }

#endif /* KWIVER_ARROWS_DARKNET_TRAINER */
