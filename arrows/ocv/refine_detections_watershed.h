// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header for OCV refine detections watershed algorithm

#ifndef KWIVER_ARROWS_OCV_REFINE_DETECTIONS_WATERSHED_H_
#define KWIVER_ARROWS_OCV_REFINE_DETECTIONS_WATERSHED_H_

#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <vital/algo/refine_detections.h>

namespace kwiver {
namespace arrows {
namespace ocv {

/// A class for drawing various information about feature tracks
class KWIVER_ALGO_OCV_EXPORT refine_detections_watershed
: public vital::algorithm_impl<refine_detections_watershed,
    vital::algo::refine_detections>
{
public:
  PLUGIN_INFO( "ocv_watershed",
               "Estimate a segmentation using watershed" )

  /// Constructor
  refine_detections_watershed();

  /// Destructor
  virtual ~refine_detections_watershed();

  /// Get this algorithm's \link kwiver::vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

  /// Refine all object detections on the provided image
  /**
   * This method analyzes the supplied image and and detections on it,
   * returning a refined set of detections.
   *
   * \param image_data the image pixels
   * \param detections detected objects
   * \returns vector of image objects refined
   */
  virtual vital::detected_object_set_sptr
  refine( vital::image_container_sptr image_data,
          vital::detected_object_set_sptr detections ) const;

private:

  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d_;
};

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif
