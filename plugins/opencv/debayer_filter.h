/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_OPENCV_DEBAYER_FILTER_H
#define VIAME_OPENCV_DEBAYER_FILTER_H

#include "viame_opencv_export.h"

#include <vital/algo/image_filter.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

namespace viame {

class VIAME_OPENCV_EXPORT debayer_filter
  : public kwiver::vital::algo::image_filter
{
public:
  PLUGGABLE_IMPL(
    debayer_filter,
    "OpenCV debayer filter for converting to RGB or grayscale",
    PARAM_DEFAULT( pattern, std::string,
      "Bayer pattern, can either be: BG, GB, RG, or GR. The two letters indicate the particular pattern type.", "BG" ),
    PARAM_DEFAULT( force_8bit, bool,
      "Force output to be 8 bit", false )
  )

  virtual ~debayer_filter() = default;

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  // Main filtering method
  virtual kwiver::vital::image_container_sptr filter(
    kwiver::vital::image_container_sptr image_data );

private:
  mutable bool m_is_first = true;
};

} // end namespace

#endif /* VIAME_OPENCV_DEBAYER_FILTER_H */
