// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_ALGO_CONVERT_IMAGE_H_
#define VITAL_ALGO_CONVERT_IMAGE_H_

#include <viame/algorithm_framework/vital_config.h>

#include <memory>
#include <string>

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/core_types/image_container.h>

namespace kwiver {

namespace vital {

namespace algo {

/// An abstract base class for converting base image type.
///
/// Arrows that implement this interface convert the input image type
/// (e.g. BGR 16) to a different type (e.g. RGB 8). Concrete
/// implementations usually work with a single image representation,
/// such as VXL or OCV.
///
/// If you are lookling for an interface for an image transform that
/// will change the value of a pixel, then use the image_filter
/// interface.
class VITAL_ALGO_EXPORT convert_image
  : public kwiver::vital::algorithm
{
public:
  convert_image();
  PLUGGABLE_INTERFACE( convert_image );
  /// Set this algorithm's properties via a config block
  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration(
    kwiver::vital::config_block_sptr config ) const;

  /// Convert image base type
  virtual kwiver::vital::image_container_sptr convert(
    kwiver::vital::image_container_sptr img ) const = 0;
};

typedef std::shared_ptr< convert_image > convert_image_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif // VITAL_ALGO_CONVERT_IMAGE_H_
