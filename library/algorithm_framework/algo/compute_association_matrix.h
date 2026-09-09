// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief compute_association_matrix algorithm definition

#ifndef VITAL_ALGO_COMPUTE_ASSOCIATION_MATRIX_H_
#define VITAL_ALGO_COMPUTE_ASSOCIATION_MATRIX_H_

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/algorithm_framework/vital_config.h>

#include <viame/core_types/detected_object_set.h>
#include <viame/core_types/image_container.h>
#include <viame/core_types/matrix.h>
#include <viame/core_types/object_track_set.h>
#include <viame/core_types/timestamp.h>

namespace kwiver {

namespace vital {

namespace algo {

/// An abstract base class for computing association cost matrices for tracking
class VITAL_ALGO_EXPORT compute_association_matrix
  : public kwiver::vital::algorithm
{
public:
  compute_association_matrix();
  PLUGGABLE_INTERFACE( compute_association_matrix );

  /// Compute an association matrix given detections and tracks
  ///
  /// \param ts frame ID
  /// \param image contains the input image for the current frame
  /// \param tracks active track set from the last frame
  /// \param detections input detected object sets from the current frame
  /// \param matrix output matrix
  /// \param considered output detections used in matrix
  /// \return returns whether a matrix was successfully computed
  virtual bool
  compute(
    kwiver::vital::timestamp ts,
    kwiver::vital::image_container_sptr image,
    kwiver::vital::object_track_set_sptr tracks,
    kwiver::vital::detected_object_set_sptr detections,
    kwiver::vital::matrix_d& matrix,
    kwiver::vital::detected_object_set_sptr& considered ) const = 0;
};

/// Shared pointer for compute_association_matrix algorithm definition class
typedef std::shared_ptr< compute_association_matrix >
  compute_association_matrix_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif // VITAL_ALGO_COMPUTE_ASSOCIATION_MATRIX_MAP_H_
