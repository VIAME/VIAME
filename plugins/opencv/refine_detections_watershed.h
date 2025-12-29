// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header for OCV refine detections watershed algorithm

#ifndef VIAME_OPENCV_REFINE_DETECTIONS_WATERSHED_H
#define VIAME_OPENCV_REFINE_DETECTIONS_WATERSHED_H

#include <plugins/opencv/viame_opencv_export.h>

#include <vital/algo/refine_detections.h>

namespace viame {

/// A class for drawing various information about feature tracks
class VIAME_OPENCV_EXPORT refine_detections_watershed
: public kwiver::vital::algorithm_impl<refine_detections_watershed,
    kwiver::vital::algo::refine_detections>
{
public:
  PLUGIN_INFO( "ocv_watershed",
               "Estimate a segmentation using watershed" )

  /// Constructor
  refine_detections_watershed();

  /// Destructor
  virtual ~refine_detections_watershed();

  /// Get this algorithm's \link kwiver::vital::config_block configuration block \endlink
  virtual kwiver::vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(kwiver::vital::config_block_sptr config);
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

  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d_;
};

} // end namespace viame

#endif
