// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_DARKENT_DETECTOR
#define KWIVER_ARROWS_DARKENT_DETECTOR

#include <arrows/darknet/kwiver_algo_darknet_export.h>

#include <vital/algo/image_object_detector.h>

namespace kwiver {
namespace arrows {
namespace darknet {

// -----------------------------------------------------------------------------
/**
 * @brief
 *
 */
class KWIVER_ALGO_DARKNET_EXPORT darknet_detector
  : public vital::algo::image_object_detector
{
public:
  darknet_detector();
  virtual ~darknet_detector();

  PLUGIN_INFO( "darknet",
               "Image object detector using darknet." )

  vital::config_block_sptr get_configuration() const override;

  void set_configuration( vital::config_block_sptr config ) override;
  bool check_configuration( vital::config_block_sptr config ) const override;

  vital::detected_object_set_sptr detect(
    vital::image_container_sptr image_data ) const override;

private:
  class priv;
  const std::unique_ptr<priv> d;
};

} } }

#endif /* KWIVER_ARROWS_DARKENT_DETECTOR */
