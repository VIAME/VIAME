// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief detect_features algorithm definition

#ifndef VITAL_ALGO_DETECT_FEATURES_H_
#define VITAL_ALGO_DETECT_FEATURES_H_

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/core_types/feature_set.h>
#include <viame/core_types/image_container.h>
#include <viame/algorithm_framework/vital_config.h>

namespace kwiver {

namespace vital {

namespace algo {

/// An abstract base class for detecting feature points
class VITAL_ALGO_EXPORT detect_features
  : public kwiver::vital::algorithm
{
public:
  detect_features();
  PLUGGABLE_INTERFACE( detect_features );
  /// Extract a set of image features from the provided image
  ///
  /// A given mask image should be one-channel (mask->depth() == 1). If the
  /// given mask image has more than one channel, only the first will be
  /// considered.
  ///
  /// \throws image_size_mismatch_exception
  ///    When the given non-zero mask image does not match the size of the
  ///    dimensions of the given image data.
  ///
  /// \param image_data contains the image data to process
  /// \param mask Mask image where regions of positive values (boolean true)
  ///             indicate regions to consider. Only the first channel will be
  ///             considered.
  /// \returns a set of image features
  virtual kwiver::vital::feature_set_sptr
  detect(
    kwiver::vital::image_container_sptr image_data,
    kwiver::vital::image_container_sptr mask = kwiver::vital::
    image_container_sptr() )
  const = 0;
};

/// Shared pointer for detect_features algorithm definition class
typedef std::shared_ptr< detect_features > detect_features_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif // VITAL_ALGO_DETECT_FEATURES_H_
