// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
* \file
* \brief Header for utility functions for structure from motion
*/

#ifndef KWIVER_ARROWS_MVG_SFM_UTILS_H_
#define KWIVER_ARROWS_MVG_SFM_UTILS_H_

#include <vector>
#include <unordered_set>

#include <vital/vital_config.h>
#include <vital/vital_types.h>
#include <vital/types/bounding_box.h>
#include <arrows/mvg/kwiver_algo_mvg_export.h>

#include <vital/types/feature_track_set.h>
#include <vital/types/landmark_map.h>
#include <vital/types/camera_map.h>
#include <vital/types/camera_perspective.h>
#include <vital/types/camera_perspective_map.h>
#include <vital/types/feature_track_set.h>

namespace kwiver {
namespace arrows {
namespace mvg{

/// Generate camera based on crop
/**
 * Change the parameters of the camera appropriately to account for the cropping
 * \param [in] cam   The camera to crop.
 * \param [in] crop  The region of the image to use
 * \return           The cropped camera, with shifted principal point
 */
KWIVER_ALGO_MVG_EXPORT
kwiver::vital::camera_perspective_sptr
crop_camera(const kwiver::vital::camera_perspective_sptr& cam,
            vital::bounding_box<int> crop);

/// Detect tracks which remain stationary in the image
/**
 * For each track, compute its mean location in the image space and then
 * check that at least one track state is a distance more than \p threshold
 * pixels from the mean.  If all locations are close to the mean then
 * add this to the set of stationary tracks to return.  If a subset of tracks
 * are stationary it may indicate that these tracks lie on a heads-up display
 * or other feature fixed to the camera, rather than on the scene.
 * \param [in] tracks     The set of feature tracks to process.
 * \param [in] threshold  The threshold on pixel distance to the mean
 * \return                The set of stationary tracks
 */
KWIVER_ALGO_MVG_EXPORT
std::set<vital::track_sptr>
detect_stationary_tracks(vital::feature_track_set_sptr tracks,
                         double threshold = 10.0);

/// Select keyframes that are a good starting point for SfM
/**
 * Analyze a set of tracks and select a good subset of keyframe that would
 * be useful to initialize structure-from-motion.  These frames are well
 * distributed but still highly connected to each other.  The \p radius
 * controls how far apart the initial keyframes must be.  The \p ratio
 * then controls which key frame intervals are weak and must be sub-divided.
 * \param [in] tracks     The set of feature tracks to process
 * \param [in] radius     The number of adjacent frames to block
 *                        in non-maximum suppression
 * \param [in] ratio      The minimum ratio of number of tracks between
 *                        adjacent key frames and adjacent frames
 * \return                The set of selected frame numbers
 */
KWIVER_ALGO_MVG_EXPORT
std::set<vital::frame_id_t>
keyframes_for_sfm(vital::feature_track_set_sptr tracks,
                  const vital::frame_id_t radius = 10,
                  const double ratio = 0.75);

typedef std::pair<vital::frame_id_t, float> coverage_pair;
typedef std::vector<coverage_pair> frame_coverage_vec;

/// Calculate fraction of each image that is covered by landmark projections
/**
 * For each frame find landmarks that project into that frame.  Mark the
 * associated feature projection locations as occupied in a mask.  After all
 * masks have been accumulated for all frames calculate the fraction of each
 * mask that is occupied.  Return this fraction in a frame coverage vector.
 * \param [in] tracks the set of feature tracks
 * \param [in] lms landmarks to check coverage on
 * \param [in] cams the set of frames to have coverage calculated on
 * \return     the image coverages
 */
KWIVER_ALGO_MVG_EXPORT
frame_coverage_vec
image_coverages(
  std::vector<vital::track_sptr> const& trks,
  vital::landmark_map::map_landmark_t const& lms,
  vital::simple_camera_perspective_map::frame_to_T_sptr_map const& cams);

typedef std::vector<std::unordered_set<vital::frame_id_t>> camera_components;

/// find connected components of cameras
/**
 * Find connected components in the view graph.  Cameras that view the same
 * landmark are connected in the graph.
 * \param [in] cams the cameras in the view graph.
 * \param [in] lms the landmarks in the view graph.
 * \param [in] tracks the track set
 * \return camera_components vector.  Each set in the vector represents a
 * different connected component.
 */
KWIVER_ALGO_MVG_EXPORT
camera_components
connected_camera_components(
  vital::simple_camera_perspective_map::frame_to_T_sptr_map const& cams,
  vital::landmark_map::map_landmark_t const& lms,
  vital::feature_track_set_sptr tracks);

/// Detect critical tracks that connect disjoint components
/**
 * Compute the subset of all tracks which have track states in more than one
 * connected component.  These tracks have likely been marked as outliers,
 * which is why there are multiple connected components.  To avoid a
 * fragmented solution these critical tracks need to be reconsidered.
 * \param [in] cc      The connected camera components
 * \param [in] tracks  The set of tracks
 * \returns            A vector of all critical tracks
 */
KWIVER_ALGO_MVG_EXPORT
std::vector<vital::track_sptr>
detect_critical_tracks(camera_components const& cc,
                       vital::feature_track_set_sptr tracks);

/// Detect bad landmarks
/**
* Checks landmark reprojection errors, triangulation angles and whether or not
* landmark is adequately constrained (2+ rays).  If not it is returned as a
* landmark to be removed
* \param [in] cams the cameras in the view graph.
* \param [in] lms the landmarks in the view graph.
* \param [in] tracks the track set
* \param [in] triang_cos_ang_thresh features must have one pair of rays that
*             meet this minimum intersection angle to keep
* \param [in] error_tol reprojection error threshold
* \param [in] min_landmark_inliers minimum number of inlier measurements to keep a landmark.
              Set to -1 to ignore.
* \param [in] median_distance_multiple remove landmarks more than the median
*             landmark to camera distance * median_distance_multiple.  Set to 0
              to disable
* \return set of landmark ids (track_ids) that were bad and should be removed
*/
KWIVER_ALGO_MVG_EXPORT
std::set<vital::landmark_id_t>
detect_bad_landmarks(
  vital::simple_camera_perspective_map::frame_to_T_sptr_map const& cams,
  vital::landmark_map::map_landmark_t const& lms,
  vital::feature_track_set_sptr tracks,
  double triang_cos_ang_thresh,
  double error_tol = 5.0,
  int min_landmark_inliers = -1,
  double median_distance_multiple = 10);

/// Remove landmarks with IDs in the set
/**
 * Erases the landmarks with the associated ids from lms
 * \param [in] to_remove track ids for landmarks to set null
 * \param [in,out] lms landmark map to remove landmarks from
 */
KWIVER_ALGO_MVG_EXPORT
void
remove_landmarks(const std::set<vital::track_id_t>& to_remove,
  vital::landmark_map::map_landmark_t& lms);

/// Detect bad cameras in sfm solution
/**
 * Find cameras that don't meet the minimum required coverage and return them
 * \param [in] cams the cameras in the view graph.
 * \param [in] lms the landmarks in the view graph.
 * \param [in] tracks the track set
 * \param [in] coverage_thresh images that have less than this  fraction [0 - 1]
 *             of coverage are included in the return set
 * \return set of frames that do not meet the coverage requirement
 */
KWIVER_ALGO_MVG_EXPORT
std::set<vital::frame_id_t>
detect_bad_cameras(
  vital::simple_camera_perspective_map::frame_to_T_sptr_map const& cams,
  vital::landmark_map::map_landmark_t const& lms,
  vital::feature_track_set_sptr tracks,
  float coverage_thresh);

/// Clean structure from motion solution
/**
 * Clean up the structure from motion solution by checking landmark
 * reprojection errors, removing under-constrained landmarks, removing
 * under-constrained cameras and keeping only the largest connected component
 * of cameras.
 * \param [in,out] cams the cameras in the view graph.  Removed cameras are set to null.
 * \param [in,out] lms the landmarks in the view graph.  Removed landmarks are set to null.
 * \param [in] tracks the track set
 * \param [in] triang_cos_ang_thresh largest angle rays intersecting landmark must
 *             have less than this cos angle for landmkark to be kept.
 * \param [out] removed_cams frame ids of cameras that are removed while cleaning
 * \param [in] active_cams if non-empty only these cameras will be cleaned
 * \param [in] active_lms if non-empty only these landmarks will be cleaned
 * \param [in] image_coverage_threshold images must have this fraction [0 - 1] of
 *             coverage to be kept in the solution
 * \param [in] error_tol maximum reprojection error to keep features
 * \param [in] min_landmark_inliers minimum number of inlier measurements to keep a landmark.
 *             Set to -1 to ignore.
 */
KWIVER_ALGO_MVG_EXPORT
void
clean_cameras_and_landmarks(
  vital::simple_camera_perspective_map& cams,
  vital::landmark_map::map_landmark_t& lms,
  vital::feature_track_set_sptr tracks,
  double triang_cos_ang_thresh,
  std::vector<vital::frame_id_t> &removed_cams,
  const std::set<vital::frame_id_t> &active_cams,
  const std::set<vital::landmark_id_t> &active_lms,
  double image_coverage_threshold = 0.25,
  double error_tol = 5.0,
  int min_landmark_inliers = -1);

/// Return true if the camera is upright
/*
 * Test if "up" in the image aligns with "up" in the world for each camera.
 * The alignment in this case means that the dot product of these vectors
 * is postive, not that the vectors are equal.
 * Up in the image is the negative Y-axis.  Up in the world defaults to
 * the postive Z-axis, but this is configurable by specifying \p up.
 */
KWIVER_ALGO_MVG_EXPORT
bool
camera_upright(vital::camera_perspective const& camera,
               vital::vector_3d const& up = vital::vector_3d(0, 0, 1));

/// Return true if most cameras are upright
/*
 * \sa camera_upright
 */
KWIVER_ALGO_MVG_EXPORT
bool
majority_upright(
  vital::camera_perspective_map::frame_to_T_sptr_map const& cameras,
  vital::vector_3d const& up = vital::vector_3d(0,0,1));

/// Return a subset of cameras on the positive side of a plane
vital::camera_perspective_map::frame_to_T_sptr_map
cameras_above_plane(
  vital::camera_perspective_map::frame_to_T_sptr_map const& cameras,
  vital::vector_4d const& plane);

/// Compute the ground center of a collection of landmarks
/**
 * Compute the location of the center of the ground in a collection of
 * landmarks.  This function assumes that the landmarks are already oriented
 * with the ground normal vector aligned with the Z-axis.  It returns the
 * median location in X and Y, and a small percentile (5%) of the height in Z.
 */
KWIVER_ALGO_MVG_EXPORT
vital::vector_3d
landmarks_ground_center(vital::landmark_map const& landmarks,
                        double ground_frac = 0.05);
}
}
}

#endif
