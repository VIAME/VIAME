// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV motion detection 3-frame difference algorithm impl interface
 */

#ifndef KWIVER_ARROWS_OCV_THREE_FRAME_DIFFERENCING_H_
#define KWIVER_ARROWS_OCV_THREE_FRAME_DIFFERENCING_H_

#include <memory>

#include <opencv2/opencv.hpp>

#include <vital/types/timestamp.h>
#include <vital/vital_config.h>
#include <vital/algo/detect_motion.h>
#include <vital/config/config_block.h>

#include <arrows/ocv/kwiver_algo_ocv_export.h>

namespace kwiver {
namespace arrows {
namespace ocv {

/// OCV implementation of detect_motion using three-frame differencing
class KWIVER_ALGO_OCV_EXPORT detect_motion_3frame_differencing
  : public vital::algo::detect_motion
{
public:
  PLUGIN_INFO( "ocv_3frame_differencing",
               "OCV implementation of detect_motion using three-frame differencing" )
  /// Constructor
  detect_motion_3frame_differencing();
  /// Destructor
  virtual ~detect_motion_3frame_differencing() noexcept;

  /// Get this algorithm's \link kwiver::vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);
  /// Check that the algorithm's configuration vital::config_block is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

  /// Detect motion from a sequence of images
  /**
   * This method detects motion of foreground objects within a
   * sequence of images in which the background remains stationary.
   * Sequential images are passed one at a time. Motion estimates
   * are returned for each image as a heat map with higher values
   * indicating greater confidence.
   *
   * \param ts Timestamp for the input image
   * \param image Image from a sequence
   * \param reset_model Indicates that the model should be reset, for example,
   * due to changes in lighting condition or
   * camera pose
   *
   * \returns A heat map image is returned indicating the confidence
   * that motion occurred at each pixel. Heat map image is single channel
   * and has the same width and height dimensions as the input image.
   */
  virtual kwiver::vital::image_container_sptr
    process_image( const kwiver::vital::timestamp& ts,
                   const kwiver::vital::image_container_sptr image,
                   bool reset_model );

private:
  // private implementation class
  class priv;
  std::unique_ptr<priv> d_;
};

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif
