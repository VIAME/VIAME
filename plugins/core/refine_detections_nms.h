/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_REFINE_DETECTIONS_NMS_H
#define VIAME_CORE_REFINE_DETECTIONS_NMS_H

#include "viame_core_export.h"

#include <vital/algo/refine_detections.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

namespace viame {

// -----------------------------------------------------------------------------
/**
 * \class refine_detections_nms
 *
 * \brief Prunes overlapping detections
 *
 * \iports
 * \iport{detections}
 *
 * \oports
 * \oport{pruned_detections}
 */
class VIAME_CORE_EXPORT refine_detections_nms
  : public kwiver::vital::algo::refine_detections
{

public:
  PLUGGABLE_IMPL(
    refine_detections_nms,
    "Refines detections based on overlap.\n\n"
    "This algorithm sorts through detections, pruning detections "
    "that heavily overlap with higher confidence detections.",
    PARAM_DEFAULT(
      nms_scale_factor, double,
      "The factor by which the detections are scaled during NMS.",
      1.0 ),
    PARAM_DEFAULT(
      output_scale_factor, double,
      "The factor by which the refined final detections are scaled.",
      1.0 ),
    PARAM_DEFAULT(
      max_scale_difference, double,
      "If the ratio of the areas of two boxes are different by more "
      "than this amount [1.0,inf], then don't suppress them.",
      4.0 ),
    PARAM_DEFAULT(
      max_overlap, double,
      "The maximum percent a detection can overlap with another "
      "before it's discarded. Range [0.0,1.0].",
      0.5 )
  )

  virtual ~refine_detections_nms() = default;

  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration(kwiver::vital::config_block_sptr config) const;

  /// Refine all object detections on the provided image
  /**
   * This method analyzes the supplied image and and detections on it,
   * returning a refined set of detections.
   *
   * \param image_data the image pixels
   * \param detections detected objects
   * \returns vector of image objects refined
   */
  virtual kwiver::vital::detected_object_set_sptr
  refine( kwiver::vital::image_container_sptr image_data,
          kwiver::vital::detected_object_set_sptr detections ) const;

private:
  void initialize() override;

  /// Computed from max_scale_difference, not a config param
  double m_min_scale_difference;

 }; // end class refine_detections_nms


} // end namespace viame

#endif // VIAME_CORE_REFINE_DETECTIONS_NMS_H
