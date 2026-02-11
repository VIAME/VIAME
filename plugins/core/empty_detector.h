/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_EMPTY_DETECTOR_H
#define VIAME_CORE_EMPTY_DETECTOR_H

#include "viame_core_export.h"

#include <vital/algo/image_object_detector.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

namespace viame {

// The worst detector in the world, always produces empty detections for debug
// purposes, and to hack reseting into VIAME-web
class VIAME_CORE_EXPORT empty_detector :
  public kwiver::vital::algo::image_object_detector
{
public:
  PLUGGABLE_IMPL(
    empty_detector,
    "Produce empty detector output" )

  virtual ~empty_detector() = default;

  bool check_configuration( kwiver::vital::config_block_sptr config ) const override
  {
    return true;
  }

  // Main detection method
  virtual kwiver::vital::detected_object_set_sptr detect(
    kwiver::vital::image_container_sptr image_data ) const override;
};

} // end namespace

#endif /* VIAME_EMPTY_DETECTOR_H */
