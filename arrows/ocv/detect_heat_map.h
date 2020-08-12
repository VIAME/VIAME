/*ckwg +29
 * Copyright 2017, 2019-2020 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file
 * \brief OCV warp_image algorithm impl interface
 */

#ifndef KWIVER_ARROWS_OCV_HEAT_MAP_BOUNDING_BOXES_H_
#define KWIVER_ARROWS_OCV_HEAT_MAP_BOUNDING_BOXES_H_

#include <memory>

#include <opencv2/opencv.hpp>

#include <vital/types/timestamp.h>
#include <vital/vital_config.h>
#include <vital/algo/image_object_detector.h>
#include <vital/config/config_block.h>

#include <arrows/ocv/kwiver_algo_ocv_export.h>

namespace kwiver {
namespace arrows {
namespace ocv {

// ----------------------------------------------------------------
/**
 * @brief Generate bounding boxes from a heat map.
 *
 * This object detector algorithm implementation extracts a detected object set
 * from a heat map image. There are a number of different modes of operation. If
 * "threshold" is set to a positive value, the heat map is first thresholded to
 * a binary image, and the detected objects correspond to bounding boxes around
 * clusters of connected pixels. These detected objects can further be filtered
 * based on the cluster region properties (e.g., area, fill fraction, etc.).
 * If threshold is set to -1, the heat map will be processed using the full
 * pixel-value range.
 *
 * If a threshold is provided and force_bbox_width and force_bbox_height are not
 * set, then the thresholded binary image will be clustered into connected-
 * component regions, each becoming a detection with a bounding box.
 *
 * If force_bbox_width and force_bbox_height are set, a greedy algorithm will
 * attempt to put down bounding boxes of fixed size. The first bounding box is
 * chosen to cover the greatest sum-intensity in the heat map. This region is
 * masked out, and the next bounding box tries to captures the maximum remaining
 * intensity, and this process is repeated. The end result is not necassarily a
 * global optimum, as map cover problems are np hard.
 *
 */
class KWIVER_ALGO_OCV_EXPORT detect_heat_map
  : public vital::algo::image_object_detector
{
public:
  PLUGIN_INFO( "detect_heat_map",
               "OCV implementation to create detections from heatmaps" )
  /// Constructor
  detect_heat_map();
  /// Destructor
  virtual ~detect_heat_map() noexcept;

  /// Get this algorithm's \link kwiver::vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);
  /// Check that the algorithm's configuration vital::config_block is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

  /// Find all objects on the provided image
  /**
   * This method analyzes the supplied image and along with any saved
   * context, returns a vector of detected image objects.
   *
   * \param image_data the image pixels
   * \returns vector of image objects found
   */
  virtual
  kwiver::vital::detected_object_set_sptr
    detect(kwiver::vital::image_container_sptr image_data) const;

private:
  // private implementation class
  class priv;
  std::unique_ptr<priv> d_;
};

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif
