// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header for OCV refine detections watershed algorithm

#ifndef VIAME_OPENCV_REFINE_DETECTIONS_WATERSHED_H
#define VIAME_OPENCV_REFINE_DETECTIONS_WATERSHED_H

#include "viame_opencv_export.h"

#include <vital/algo/refine_detections.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

namespace viame {

/// A class for drawing various information about feature tracks
class VIAME_OPENCV_EXPORT refine_detections_watershed
  : public kwiver::vital::algo::refine_detections
{
public:
  PLUGGABLE_IMPL( refine_detections_watershed,
                  "Estimate a segmentation using watershed",
    PARAM_DEFAULT( seed_with_existing_masks, bool,
                   "If true, use existing masks as seed regions",
                   true ),
    PARAM_DEFAULT( seed_scale_factor, double,
                   "Amount to scale the detection by to produce "
                   "a high-confidence seed region",
                   0.2 ),
    PARAM_DEFAULT( uncertain_scale_factor, double,
                   "Amount to scale the detection by to produce "
                   "a region that will be marked as uncertain",
                   1 )
  )

  virtual ~refine_detections_watershed() = default;

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
