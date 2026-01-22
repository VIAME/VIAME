/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_FULL_FRAME_DETECTOR_H
#define VIAME_CORE_FULL_FRAME_DETECTOR_H

#include "viame_core_export.h"

#include <vital/algo/image_object_detector.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

namespace viame {

class VIAME_CORE_EXPORT full_frame_detector
  : public kwiver::vital::algo::image_object_detector
{
public:
  PLUGGABLE_IMPL(
    full_frame_detector,
    "Outputs a single fixed full-frame detection the same size as "
    "the input image size.",
    PARAM_DEFAULT(
      detection_type, std::string,
      "Object type to add to newly created detected objects",
      "generic_object_proposal" )
  )

  virtual ~full_frame_detector() = default;

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual kwiver::vital::detected_object_set_sptr detect(
    kwiver::vital::image_container_sptr image_data ) const;
};

} // end namespace viame

#endif // VIAME_CORE_FULL_FRAME_DETECTOR_H
