// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief detect_features adaptor that applies a filter to the features

#ifndef KWIVER_ARROWS_CORE_DETECT_FEATURES_FILTERED_H_
#define KWIVER_ARROWS_CORE_DETECT_FEATURES_FILTERED_H_

#include <viame/algorithm_framework/algo/detect_features.h>
#include <viame/algorithm_framework/algo/filter_features.h>

#include "viame_image_processing_export.h"
#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/algorithm_framework/algo/algorithm.txx>

namespace kwiver {

namespace arrows {

namespace core {

/// A feature detector that applies a filter to results
class VIAME_IMAGE_PROCESSING_EXPORT detect_features_filtered
  : public kwiver::vital::algo::detect_features
{
public:
  PLUGGABLE_IMPL(
    detect_features_filtered,
    "Wrapper that runs a feature detector and "
    "applies a filter to the detector output",
    PARAM(
      detector, vital::algo::detect_features_sptr,
      "detector" ),
    PARAM(
      filter, vital::algo::filter_features_sptr,
      "filter" )
  )

  /// Destructor
  virtual ~detect_features_filtered();

  /// Check that the algorithm's configuration config_block is valid
  virtual bool check_configuration( vital::config_block_sptr config ) const;

  /// Extract a set of image features from the provided image
  ///
  /// A given mask image should be one-channel (mask->depth() == 1). If the
  /// given mask image has more than one channel, only the first will be
  /// considered.
  ///
  /// \param image_data contains the image data to process
  /// \param mask Mask image where regions of positive values (boolean true)
  ///             indicate regions to consider. Only the first channel will be
  ///             considered.
  /// \returns a set of image features
  virtual vital::feature_set_sptr
  detect(
    vital::image_container_sptr image_data,
    vital::image_container_sptr mask = vital::image_container_sptr() ) const;

private:
  void initialize() override;
  /// private implementation class
  class priv;
  KWIVER_UNIQUE_PTR( priv, d_ );
};

} // end namespace core

} // end namespace arrows

} // end namespace kwiver

#endif // KWIVER_ARROWS_CORE_DETECT_FEATURES_FILTERED_H_
