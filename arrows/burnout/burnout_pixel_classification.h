// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_BURNOUT_PIXEL_CLASSIFICATION
#define KWIVER_ARROWS_BURNOUT_PIXEL_CLASSIFICATION

#include <arrows/burnout/kwiver_algo_burnout_export.h>

#include <vital/algo/image_filter.h>

namespace kwiver {
namespace arrows {
namespace burnout {

/**
 * @brief Burnout Image Filtering
 *
 * This method contains basic methods for image filtering on top of input
 * images via automatic white balancing and smoothing.
 */
class KWIVER_ALGO_BURNOUT_EXPORT burnout_pixel_classification
  : public vital::algo::image_filter
{
public:
  burnout_pixel_classification();
  virtual ~burnout_pixel_classification();

  PLUGIN_INFO( "burnout_classifier",
               "Pixel classification using burnout" )

  vital::config_block_sptr get_configuration() const override;

  void set_configuration( vital::config_block_sptr config ) override;
  bool check_configuration( vital::config_block_sptr config ) const override;

  kwiver::vital::image_container_sptr filter(
    kwiver::vital::image_container_sptr image_data ) override;

private:

  class priv;
  const std::unique_ptr<priv> d;
};

} } }

#endif /* KWIVER_ARROWS_BURNOUT_PIXEL_CLASSIFICATION */
