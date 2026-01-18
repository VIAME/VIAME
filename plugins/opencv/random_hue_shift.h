/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_OPENCV_RANDOM_HUE_SHIFT_H
#define VIAME_OPENCV_RANDOM_HUE_SHIFT_H

#include "viame_opencv_export.h"

#include <vital/algo/image_filter.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

namespace viame {

class VIAME_OPENCV_EXPORT random_hue_shift
  : public kwiver::vital::algo::image_filter
{
public:
  PLUGGABLE_IMPL( random_hue_shift,
                  "Add in a random hue shift to the imagery",
                  PARAM_DEFAULT( trigger_percent, double,
                                 "Trigger for other operations", 0.50 ),
                  PARAM_DEFAULT( hue_range, double,
                                 "Hue random adjustment range", 0.0 ),
                  PARAM_DEFAULT( sat_range, double,
                                 "Saturation random adjustment range", 0.0 ),
                  PARAM_DEFAULT( int_range, double,
                                 "Intensity random adjustment range", 0.0 ),
                  PARAM_DEFAULT( rgb_shift_range, double,
                                 "Random color shift range", 0.0 ) )

  virtual ~random_hue_shift() = default;

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  // Main filtering method
  virtual kwiver::vital::image_container_sptr filter(
    kwiver::vital::image_container_sptr image_data );
};

} // end namespace

#endif /* VIAME_OPENCV_RANDOM_HUE_SHIFT_H */
