/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_EMPTY_DETECTOR_H
#define VIAME_CORE_EMPTY_DETECTOR_H

#include "viame_core_export.h"

#include <vital/algo/image_object_detector.h>
#include "viame_algorithm_plugin_interface.h"

namespace viame {

// The worst detector in the world, always produces empty detections for debug
// purposes, and to hack reseting into VIAME-web
class VIAME_CORE_EXPORT empty_detector :
  public kwiver::vital::algo::image_object_detector
{
public:
  VIAME_ALGORITHM_PLUGIN_INTERFACE( empty_detector )
  static constexpr char const* name = "empty";

  static constexpr char const* description = "Produce empty detector output";

  empty_detector();
  virtual ~empty_detector();

  // Get the current configuration (parameters) for this detector
  virtual kwiver::vital::config_block_sptr get_configuration() const;

  // Set configurations automatically parsed from input pipeline and config files
  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  // Main detection method
  virtual kwiver::vital::detected_object_set_sptr detect(
    kwiver::vital::image_container_sptr image_data ) const;
};

} // end namespace

#endif /* VIAME_EMPTY_DETECTOR_H */
