// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header for MVG triangulate_landmarks algorithm
 */

#ifndef KWIVER_ARROWS_MVG_TRIANGULATE_LANDMARKS_H_
#define KWIVER_ARROWS_MVG_TRIANGULATE_LANDMARKS_H_

#include <arrows/mvg/kwiver_algo_mvg_export.h>

#include <vital/algo/triangulate_landmarks.h>

namespace kwiver {
namespace arrows {
namespace mvg {

/// A class for triangulating landmarks from feature tracks and cameras using Eigen
class KWIVER_ALGO_MVG_EXPORT triangulate_landmarks
: public vital::algo::triangulate_landmarks
{
public:
  PLUGIN_INFO( "mvg",
               "Triangulate landmarks from tracks and cameras"
               " using a simple least squares solver." )

  /// Constructor
  triangulate_landmarks();

  /// Destructor
  virtual ~triangulate_landmarks();

  /// Copy Constructor
  triangulate_landmarks(const triangulate_landmarks& other);

  /// Get this alg's \link vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algo's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

  /// Triangulate the landmark locations given sets of cameras and feature tracks
  /**
   * \param [in] cameras the cameras viewing the landmarks
   * \param [in] tracks the feature tracks to use as constraints
   * \param [in,out] landmarks the landmarks to triangulate
   *
   * This function only triangulates the landmarks with indicies in the
   * landmark map and which have support in the tracks and cameras.  Note:
   * triangulate modifies the inlier/outlier flags in tracks. It also sets
   * the cosine of the maximum observation angle and number of observations
   * in the landmarks.
   */
  virtual void
  triangulate(vital::camera_map_sptr cameras,
              vital::feature_track_set_sptr tracks,
              vital::landmark_map_sptr& landmarks) const;

  /// Triangulate the landmark locations given sets of cameras and feature tracks
  /**
  * \param [in] cameras the cameras viewing the landmarks
  * \param [in] tracks the feature tracks to use as constraints in a map
  * \param [in,out] landmarks the landmarks to triangulate
  *
  * This function only triangulates the landmarks with indicies in the
  * landmark map and which have support in the tracks and cameras.  Note:
  * triangulate modifies the inlier/outlier flags in tracks. It also sets
  * the cosine of the maximum observation angle and number of observations
  * in the landmarks.
  */
  virtual void
  triangulate(vital::camera_map_sptr cameras,
              vital::track_map_t tracks,
              vital::landmark_map_sptr& landmarks) const;

private:
  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d_;
};

typedef std::shared_ptr<triangulate_landmarks> triangulate_landmarks_sptr;

} // end namespace mvg
} // end namespace arrows
} // end namespace kwiver

#endif
