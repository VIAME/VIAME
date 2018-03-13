/*ckwg +29
* Copyright 2018 by Kitware, Inc.
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

#include <vital/vital_config.h>
#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/types/track_set.h>
#include <vital/types/landmark_map.h>
#include <vital/types/camera_map.h>

namespace kwiver {
namespace arrows {

  typedef std::pair<vital::frame_id_t, float> coverage_pair;
  typedef std::vector<coverage_pair> frame_coverage_vec;

  /// Calculate fraction of each image that is covered by landmark
  /// projections
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
    const vital::track_set_sptr tracks,
    const vital::landmark_map::map_landmark_t& lms,
    const vital::camera_map::map_camera_t& cams);

  /// clean structure from motion solution
  /**
  * Clean up the structure from motion soluiton by checking landmark
  * reprojection errors, removing under-constrained landmarks, removing
  * under-constrained cameras and keeping only the largest connected compoent
  * of cameras.
  * \param [in,out] cams the cameras in the view graph.  Removed cameras are set to null.
  * \param [in,out] lms the landmarks in the view graph.  Removed landmarks are set to null.
  * \param [in] tracks the track set
  * \param [in] triang_cos_ang_thresh largest angle rays intersecting landmark must
  *             have less than this cos angle for landmkark to be kept.
  * \param [out] removed_cams frame ids of cameras that are removed while cleaning
  * \param [in] logger the logger to write debug info to
  * \param [in] image_coverage_threshold images must have this fraction [0 - 1] of
  *             coverage to be kept in the solution
  * \param [in] error_tol maximum reprojection error to keep features
  * \return true on success
  */

  KWIVER_ALGO_CORE_EXPORT
  bool
  clean_cameras_and_landmarks(
    vital::camera_map::map_camera_t& cams,
    vital::landmark_map::map_landmark_t& lms,
    vital::track_set_sptr tracks,
    double triang_cos_ang_thresh,
    std::vector<vital::frame_id_t> &removed_cams,
    kwiver::vital::logger_handle_t logger,
    float image_coverage_threshold = 0.25,
    double error_tol = 5.0);

}
}

#endif