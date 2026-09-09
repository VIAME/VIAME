// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief compute_stereo_depth_map algorithm definition

#ifndef VITAL_ALGO_COMPUTE_STEREO_DEPTH_MAP_H_
#define VITAL_ALGO_COMPUTE_STEREO_DEPTH_MAP_H_

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/core_types/image_container.h>
#include <viame/algorithm_framework/vital_config.h>

namespace kwiver {

namespace vital {

namespace algo {

/// An abstract base class for detecting feature points
class VITAL_ALGO_EXPORT compute_stereo_depth_map
  : public kwiver::vital::algorithm
{
public:
  compute_stereo_depth_map();
  PLUGGABLE_INTERFACE( compute_stereo_depth_map );
  /// Compute a stereo depth map given two images
  ///
  /// \throws image_size_mismatch_exception
  ///    When the given input image sizes do not match.
  ///
  /// \param left_image contains the first image to process
  /// \param right_image contains the second image to process
  /// \returns a depth map image
  virtual kwiver::vital::image_container_sptr
  compute(
    kwiver::vital::image_container_sptr left_image,
    kwiver::vital::image_container_sptr right_image ) const = 0;
};

/// Shared pointer for compute_stereo_depth_map algorithm definition class
typedef std::shared_ptr< compute_stereo_depth_map >
  compute_stereo_depth_map_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif // VITAL_ALGO_COMPUTE_STEREO_DEPTH_MAP_H_
