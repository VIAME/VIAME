/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_OPENCV_ENHANCE_IMAGES_H
#define VIAME_OPENCV_ENHANCE_IMAGES_H

#include "viame_opencv_export.h"

#include <vital/algo/image_filter.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

namespace viame {

class VIAME_OPENCV_EXPORT enhance_images
  : public kwiver::vital::algo::image_filter
{
public:
  PLUGGABLE_IMPL( enhance_images,
                  "Simple illumination normalization using Lab space and CLAHE",
    PARAM_DEFAULT( apply_smoothing, bool, "Apply smoothing to the input", false ),
    PARAM_DEFAULT( smoothing_kernel, unsigned, "Smoothing kernel size", 3 ),
    PARAM_DEFAULT( apply_denoising, bool, "Apply denoising to the input", false ),
    PARAM_DEFAULT( denoise_kernel, unsigned, "Denoising kernel size", 3 ),
    PARAM_DEFAULT( denoise_coeff, unsigned, "Denoising coefficient", 3 ),
    PARAM_DEFAULT( force_8bit, bool, "Force output to be 8 bit", false ),
    PARAM_DEFAULT( auto_balance, bool, "Perform automatic white balancing", false ),
    PARAM_DEFAULT( apply_clahe, bool, "Apply CLAHE illumination normalization", false ),
    PARAM_DEFAULT( clip_limit, unsigned, "Clip limit used during hist normalization", 4 ),
    PARAM_DEFAULT( saturation, float, "Saturation scale factor", 1.0 ),
    PARAM_DEFAULT( apply_sharpening, bool, "Apply sharpening to the input", false ),
    PARAM_DEFAULT( sharpening_kernel, unsigned, "Sharpening kernel size", 3 ),
    PARAM_DEFAULT( sharpening_weight, double, "Sharpening weight [0.0,1.0]", 0.5 )
  )

  virtual ~enhance_images() = default;

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  // Main filtering method
  virtual kwiver::vital::image_container_sptr filter(
    kwiver::vital::image_container_sptr image_data );
};

} // end namespace

#endif /* VIAME_OPENCV_ENHANCE_IMAGES_H */
