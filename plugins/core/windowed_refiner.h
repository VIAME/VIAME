/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_WINDOWED_REFINER_H
#define VIAME_CORE_WINDOWED_REFINER_H

#include "viame_core_export.h"

#include <vital/algo/refine_detections.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

#include "windowed_utils.h"

namespace viame {

// -----------------------------------------------------------------------------
/**
 * @brief Window an arbitrary detection refiner over an image
 *
 * This algorithm applies a detection refinement algorithm across multiple
 * windowed regions of an image, scaling input detections to each region.
 *
 * This is a pure vital::image implementation with no OpenCV dependency.
 */
class VIAME_CORE_EXPORT windowed_refiner
  : public kwiver::vital::algo::refine_detections
{
public:
  PLUGGABLE_IMPL(
    windowed_refiner,
    "Window some other arbitrary refiner across the image (no OpenCV)",
    PARAM_DEFAULT(
      process_boundary_dets, bool,
      "Pass through detections touching tile boundaries unmodified in refiner",
      false ),
    PARAM_DEFAULT(
      overlapping_proc_once, bool,
      "Only refine each detection once if it appears in multiple tiles",
      true )
  )

  virtual ~windowed_refiner() = default;

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual kwiver::vital::detected_object_set_sptr refine(
    kwiver::vital::image_container_sptr image_data,
    kwiver::vital::detected_object_set_sptr detections ) const;

protected:
  void initialize() override;
  void set_configuration_internal( kwiver::vital::config_block_sptr config ) override;

private:
  window_settings m_settings;
  kwiver::vital::algo::refine_detections_sptr m_refiner;
  kwiver::vital::logger_handle_t m_logger;
};

} // end namespace viame

#endif /* VIAME_CORE_WINDOWED_REFINER_H */
