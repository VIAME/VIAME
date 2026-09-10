// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Arrows algorithm wrapping of OCV's DescriptorExtractor

#ifndef KWIVER_ARROWS_OCV_EXTRACT_DESCRIPTORS_H_
#define KWIVER_ARROWS_OCV_EXTRACT_DESCRIPTORS_H_

#include <viame/algorithm_framework/algo/extract_descriptors.h>

#include "viame_image_processing_export.h"

#include <opencv2/features2d/features2d.hpp>

namespace kwiver {

namespace arrows {

namespace ocv {

/// OCV specific definition for algorithms that describe feature points
///
/// This extended algorithm_def provides a common implementation for the extract
/// method.
class VIAME_IMAGE_PROCESSING_EXPORT extract_descriptors
  : public vital::algo::extract_descriptors
{
public:
  /// Extract from the image a descriptor corresponding to each feature
  ///
  /// \param image_data contains the image data to process
  /// \param features the feature locations at which descriptors are extracted
  /// \returns a set of feature descriptors
  virtual vital::descriptor_set_sptr
  extract(
    vital::image_container_sptr image_data,
    vital::feature_set_sptr& features,
    vital::image_container_sptr image_mask = vital::image_container_sptr() )
  const;

protected:
  // update the extractor based on the latest parameter value of the algorithm
  // which could be set either via config block or a parameter setter.
  virtual void update_extractor_parameters() const = 0;

  /// the descriptor extractor algorithm
  cv::Ptr< cv::DescriptorExtractor > extractor;
};

} // end namespace ocv

} // end namespace arrows

} // end namespace kwiver

#endif
