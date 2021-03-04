// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief compute_track_descriptors algorithm definition
 */

#ifndef VITAL_ALGO_COMPUTE_TRACK_DESCRIPTORS_H_
#define VITAL_ALGO_COMPUTE_TRACK_DESCRIPTORS_H_

#include <vital/vital_config.h>

#include <vital/algo/algorithm.h>

#include <vital/types/timestamp.h>
#include <vital/types/object_track_set.h>
#include <vital/types/image_container.h>
#include <vital/types/track_descriptor_set.h>

namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class for computing track descriptors
class VITAL_ALGO_EXPORT compute_track_descriptors
  : public kwiver::vital::algorithm_def<compute_track_descriptors>
{
public:
  /// Return the name of this algorithm
  static std::string static_type_name() { return "compute_track_descriptors"; }

  /// Compute track descriptors given an image and tracks
  /**
   * \param ts timestamp for the current frame
   * \param image_data contains the image data to process
   * \param tracks the tracks to extract descriptors around
   *
   * \returns a set of track descriptors
   */
  virtual kwiver::vital::track_descriptor_set_sptr
  compute( kwiver::vital::timestamp ts,
           kwiver::vital::image_container_sptr image_data,
           kwiver::vital::object_track_set_sptr tracks ) = 0;

  /// Flush any remaining in-progress descriptors
  /**
   * This is typically called at the end of a video, in case
   * any temporal descriptors and currently in progress and
   * still need to be output.
   *
   * \returns a set of track descriptors
   */
  virtual kwiver::vital::track_descriptor_set_sptr flush() = 0;

protected:
  compute_track_descriptors();

};

/// Shared pointer for base compute_track_descriptors algorithm definition class
typedef std::shared_ptr<compute_track_descriptors> compute_track_descriptors_sptr;

} } } // end namespace

#endif // VITAL_ALGO_COMPUTE_TRACK_DESCRIPTORS_H_
