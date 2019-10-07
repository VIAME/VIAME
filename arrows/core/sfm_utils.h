/*ckwg +29
* Copyright 2018-2019 by Kitware, Inc.
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
*  * Redistributions of source code must retain the above copyright notice,
*    this list of conditions and the following disclaimer.
*
*  * Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
*
*  * Neither name of Kitware, Inc. nor the names of any contributors may be used
*    to endorse or promote products derived from this software without specific
*    prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/**
* \file
* \brief Header for kwiver::arrows::sfm_utils utility functions for structure
* from motion
*/

#ifndef KWIVER_ARROWS_CORE_SFM_UTILS_H_
#define KWIVER_ARROWS_CORE_SFM_UTILS_H_

#include <vector>
#include <unordered_set>

#include <vital/vital_config.h>
#include <vital/vital_types.h>
#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/types/track_set.h>
#include <vital/types/landmark_map.h>
#include <vital/types/camera_map.h>
#include <vital/types/camera_perspective.h>
#include <vital/types/camera_perspective_map.h>
#include <vital/types/feature_track_set.h>

namespace kwiver {
namespace arrows {
namespace core{

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
KWIVER_ALGO_CORE_EXPORT
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
KWIVER_ALGO_CORE_EXPORT
camera_components
connected_camera_components(
  vital::simple_camera_perspective_map::frame_to_T_sptr_map const& cams,
  vital::landmark_map::map_landmark_t const& lms,
  vital::feature_track_set_sptr tracks);

/// detect bad landmarks
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
KWIVER_ALGO_CORE_EXPORT
std::set<vital::landmark_id_t>
detect_bad_landmarks(
  vital::simple_camera_perspective_map::frame_to_T_sptr_map const& cams,
  vital::landmark_map::map_landmark_t const& lms,
  vital::feature_track_set_sptr tracks,
  double triang_cos_ang_thresh,
  double error_tol = 5.0,
  int min_landmark_inliers = -1,
  double median_distance_multiple = 10);

/// remove landmarks with IDs in the set
/**
* Erases the landmarks with the associated ids from lms
* \param [in] to_remove track ids for landmarks to set null
* \param [in,out] lms landmark map to remove landmarks from
*/
KWIVER_ALGO_CORE_EXPORT
void
remove_landmarks(const std::set<vital::track_id_t>& to_remove,
  vital::landmark_map::map_landmark_t& lms);

/// detect bad cameras in sfm solution
/**
* Find cameras that don't meet the minimum required coverage and return them
* \param [in] cams the cameras in the view graph.
* \param [in] lms the landmarks in the view graph.
* \param [in] tracks the track set
* \param [in] coverage_thresh images that have less than this  fraction [0 - 1]
*             of coverage are included in the return set
* \return set of frames that do not meet the coverage requirement
*/
KWIVER_ALGO_CORE_EXPORT
std::set<vital::frame_id_t>
detect_bad_cameras(
  vital::simple_camera_perspective_map::frame_to_T_sptr_map const& cams,
  vital::landmark_map::map_landmark_t const& lms,
  vital::feature_track_set_sptr tracks,
  float coverage_thresh);

/// clean structure from motion solution
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
              Set to -1 to ignore.
*/
KWIVER_ALGO_CORE_EXPORT
void
clean_cameras_and_landmarks(
  vital::simple_camera_perspective_map& cams,
  vital::landmark_map::map_landmark_t& lms,
  vital::feature_track_set_sptr tracks,
  double triang_cos_ang_thresh,
  std::vector<vital::frame_id_t> &removed_cams,
  const std::set<vital::frame_id_t> &active_cams,
  const std::set<vital::landmark_id_t> &active_lms,
  float image_coverage_threshold = 0.25,
  double error_tol = 5.0,
  int min_landmark_inliers = -1);

}
}
}

#endif
