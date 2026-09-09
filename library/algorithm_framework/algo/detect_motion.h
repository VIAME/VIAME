// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Interface to algorithms for motion detection

#ifndef VITAL_ALGO_DETECT_MOTION_H
#define VITAL_ALGO_DETECT_MOTION_H

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/core_types/image_container.h>
#include <viame/core_types/timestamp.h>
#include <viame/algorithm_framework/vital_config.h>

namespace kwiver {

namespace vital {

namespace algo {

/// \brief Abstract base class for motion detection algorithms.
class VITAL_ALGO_EXPORT detect_motion
  : public kwiver::vital::algorithm
{
public:
  /// Return the name of this algorithm.
  detect_motion();
  PLUGGABLE_INTERFACE( detect_motion );
  /// Detect motion from a sequence of images
  ///
  /// This method detects motion of foreground objects within a
  /// sequence of images in which the background remains stationary.
  /// Sequential images are passed one at a time. Motion estimates
  /// are returned for each image as a heat map with higher values
  /// indicating greater confidence.
  ///
  /// \param ts Timestamp for the input image
  /// \param image Image from a sequence
  /// \param reset_model Indicates that the background model should
  /// be reset, for example, due to changes in lighting condition or
  /// camera pose
  ///
  /// \returns A heat map image is returned indicating the confidence
  /// that motion occurred at each pixel. Heat map image is single channel
  /// and has the same width and height dimensions as the input image.
  virtual image_container_sptr
  process_image(
    const timestamp& ts,
    const image_container_sptr image,
    bool reset_model ) = 0;
};

/// type definition for shared pointer to a detect_motion algorithm
typedef std::shared_ptr< detect_motion > detect_motion_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif // VITAL_ALGO_DETECT_MOTION_H
