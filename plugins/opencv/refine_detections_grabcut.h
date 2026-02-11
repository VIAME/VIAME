// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header for OCV refine detections grabcut algorithm

#ifndef VIAME_OPENCV_REFINE_DETECTIONS_GRABCUT_H
#define VIAME_OPENCV_REFINE_DETECTIONS_GRABCUT_H

#include "viame_opencv_export.h"

#include <vital/algo/refine_detections.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

namespace viame {

/// A class for drawing various information about feature tracks
class VIAME_OPENCV_EXPORT refine_detections_grabcut
  : public kwiver::vital::algo::refine_detections
{
public:
  PLUGGABLE_IMPL(
    refine_detections_grabcut,
    "Estimate a segmentation using GrabCut",
    PARAM_DEFAULT( iter_count, int,
      "Number of iterations GrabCut should perform for each detection", 2 ),
    PARAM_DEFAULT( context_scale_factor, double,
      "Amount to scale the detection by to produce a context region", 2.0 ),
    PARAM_DEFAULT( seed_with_existing_masks, bool,
      "If true, use existing masks as \"certainly foreground\" seed regions", true ),
    PARAM_DEFAULT( foreground_scale_factor, double,
      "Amount to scale the detection by to produce a region considered certainly foreground", 0.0 )
  )

  virtual ~refine_detections_grabcut() = default;

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
};

} // end namespace viame

#endif
