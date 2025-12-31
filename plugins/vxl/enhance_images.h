/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_VXL_ENHANCE_IMAGES_H
#define VIAME_VXL_ENHANCE_IMAGES_H

#include "viame_vxl_export.h"

#include <vital/algo/image_filter.h>

namespace viame {

namespace kv = kwiver::vital;

/// @brief VXL Image Enhancement
///
/// This method contains basic methods for image filtering on top of input
/// images via automatic white balancing, smoothing, and illumination
/// normalization, without depending on the burnout library.
class VIAME_VXL_EXPORT enhance_images
  : public kv::algorithm_impl< enhance_images,
      kv::algo::image_filter >
{
public:
  PLUGIN_INFO( "vxl_enhancer",
               "Image enhancement using VXL (smoothing, white balance, illumination)" )

  enhance_images();
  virtual ~enhance_images();

  /// Get this algorithm's configuration block
  virtual kv::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration( kv::config_block_sptr config );
  /// Check that the algorithm's configuration is valid
  virtual bool check_configuration( kv::config_block_sptr config ) const;

  /// Perform image enhancement
  virtual kv::image_container_sptr filter(
    kv::image_container_sptr image_data );

private:

  class priv;
  const std::unique_ptr<priv> d;
};

} // end namespace viame

#endif /* VIAME_VXL_ENHANCE_IMAGES_H */
