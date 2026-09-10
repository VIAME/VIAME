// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief OCV warp_image algorithm impl interface

#ifndef KWIVER_ARROWS_OCV_HEAT_MAP_BOUNDING_BOXES_H_
#define KWIVER_ARROWS_OCV_HEAT_MAP_BOUNDING_BOXES_H_

#include <memory>


#include <viame/algorithm_framework/algo/image_object_detector.h>
#include <viame/algorithm_framework/config/config_block.h>
#include <viame/core_types/timestamp.h>
#include <viame/algorithm_framework/vital_config.h>

#include "viame_object_detectors_export.h"

namespace kwiver {

namespace arrows {

namespace ocv {

// ----------------------------------------------------------------------------
/// @brief Generate bounding boxes from a heat map.
///
/// This object detector algorithm implementation extracts a detected object set
/// from a heat map image. There are a number of different modes of operation.
/// If
/// "threshold" is set to a positive value, the heat map is first thresholded to
/// a binary image, and the detected objects correspond to bounding boxes around
/// clusters of connected pixels. These detected objects can further be filtered
/// based on the cluster region properties (e.g., area, fill fraction, etc.).
/// If threshold is set to -1, the heat map will be processed using the full
/// pixel-value range.
///
/// If a threshold is provided and force_bbox_width and force_bbox_height are
/// not
/// set, then the thresholded binary image will be clustered into connected-
/// component regions, each becoming a detection with a bounding box.
///
/// If force_bbox_width and force_bbox_height are set, a greedy algorithm will
/// attempt to put down bounding boxes of fixed size. The first bounding box is
/// chosen to cover the greatest sum-intensity in the heat map. This region is
/// masked out, and the next bounding box tries to captures the maximum
/// remaining
/// intensity, and this process is repeated. The end result is not necassarily a
/// global optimum, as map cover problems are np hard.
///
class VIAME_OBJECT_DETECTORS_EXPORT detect_heat_map
  : public vital::algo::image_object_detector
{
public:
  // "detect_heat_map",
  PLUGGABLE_IMPL(
    detect_heat_map,
    "OCV implementation to create detections from heatmaps",

    PARAM_DEFAULT(
      threshold, double,
      "Threshold value applied to each pixel of the heat map to "
      "turn it into a binary mask. Any pixels with value "
      "strictly greater than this threshold will be turned on "
      "in the mask. Detection objects will be associated with "
      "connected-component regions of above-threshold pixels. "
      "The default threshold of -1 indicates that further "
      "processing will be done on the full-range heat map "
      "image. This mode of processing requires that ",
      -1.0 ),

    PARAM_DEFAULT(
      force_bbox_width, int,
      "Create bounding boxes of this fixed width.",
      -1 ),

    PARAM_DEFAULT(
      force_bbox_height, int,
      "Create bounding boxes of this fixed height.",
      -1 ),

    PARAM_DEFAULT(
      score_mode, std::string,
      "Mode in which a score is attributed to each detected"
      "object. A numerical value indicates that all detected"
      "objects will be assigned this fixed score. No other"
      "modes are defined at this time.",
      "1" ),

    PARAM_DEFAULT(
      bbox_buffer, int,
      "If a bounding box of fixed height and width is specified,"
      "the default bbox_buffer of 0 indicates that the bounding"
      "boxes will tightly crop features in the heat map, and "
      "multiple, non-overlapping bounding boxes will be created "
      "to cover large, extended heat-map features. With a value "
      "greater than 0, generated bounding boxes will tend to "
      "have that number of pixels of buffer from the heat-map "
      "features. Also, setting bbox_buffer causes the generated "
      "bounding boxes to tend to overlap by this number of "
      "pixels when multiple boxes are required to cover and "
      "extended heat-map feature.", 0 ),

    PARAM_DEFAULT(
      min_area, int,
      "Minimum area of above-threshold pixels in a connected "
      "cluster allowed. Area is approximately equal to the "
      "number of pixels in the cluster.",
      1 ),

    PARAM_DEFAULT(
      max_area, int,
      "Maximum area of above-threshold pixels in a connected "
      "cluster allowed. Area is approximately equal to the "
      "number of pixels in the cluster.",
      10000000 ),

    PARAM_DEFAULT(
      min_fill_fraction, double,
      "Fraction of the bounding box filled with above threshold "
      "pixels.",
      0.25 ),

    PARAM_DEFAULT(
      class_name, std::string,
      "Detection class name.",
      "unspecified" ),

    PARAM_DEFAULT(
      max_boxes, int,
      "Maximum number of "
      "bounding boxes to generate. If exceeded, the top "
      "'max_boxes' ones will be returned", 1000000 ),

    PARAM_DEFAULT(
      pyr_red_levels, int,
      "Levels of image "
      "pyramid reduction (decimation) on the heat map before "
      "box selection. This improves speed at the expense of "
      "coarseness of bounding box placement. ", 0 )
  );

  /// Destructor
  virtual ~detect_heat_map() noexcept;

  /// Check that the algorithm's configuration vital::config_block is valid
  bool check_configuration( vital::config_block_sptr config ) const override;

  /// Find all objects on the provided image
  ///
  /// This method analyzes the supplied image and along with any saved
  /// context, returns a vector of detected image objects.
  ///
  /// \param image_data the image pixels
  /// \returns vector of image objects found
  kwiver::vital::detected_object_set_sptr
  detect( kwiver::vital::image_container_sptr image_data ) const override;

private:
  void initialize()  override;
  void set_configuration_internal( vital::config_block_sptr config ) override;
  // private implementation class
  class priv;

  KWIVER_UNIQUE_PTR( priv, d_ );
};

} // end namespace ocv

} // end namespace arrows

} // end namespace kwiver

#endif
