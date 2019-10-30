/*ckwg +29
 * Copyright 2014-2018 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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
 * \brief Header for evaluation metric functions.
 */

#ifndef ALGORITHMS_METRICS_H_
#define ALGORITHMS_METRICS_H_

#include <vital/vital_config.h>
#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/types/camera.h>
#include <vital/types/camera_map.h>
#include <vital/types/camera_perspective.h>
#include <vital/types/landmark.h>
#include <vital/types/landmark_map.h>
#include <vital/types/feature.h>
#include <vital/types/track.h>
#include <vector>
#include <map>
#include <cmath>

namespace kwiver {
namespace arrows {

/// Compute the reprojection error vector of lm projected by cam compared to f
/**
 * \param [in] cam is the camera used for projection
 * \param [in] lm is the landmark projected into the camera
 * \param [in] f is the measured feature point location
 * \returns the vector between the projected lm and f in image space
 */
KWIVER_ALGO_CORE_EXPORT
vital::vector_2d reprojection_error_vec(const vital::camera& cam,
                                        const vital::landmark& lm,
                                        const vital::feature& f);


/// Compute the square reprojection error of lm projected by cam compared to f
/**
 * \param [in] cam is the camera used for projection
 * \param [in] lm is the landmark projected into the camera
 * \param [in] f is the measured feature point location
 * \returns the squared distance between the projected lm and f in image space
 */
inline
double
reprojection_error_sqr(const vital::camera& cam,
                       const vital::landmark& lm,
                       const vital::feature& f)
{
  return reprojection_error_vec(cam, lm, f).squaredNorm();
}


/// Compute the maximum angle between the rays from X to each camera center
/**
 * \param [in] cameras is the set of cameras that view X
 * \param [in] X is the landmark projected into the cameras
 * \returns cos of the maximum angle pair of rays intersecting at lm from the
 *          cameras observing lm
 */
KWIVER_ALGO_CORE_EXPORT
double
bundle_angle_max(const std::vector<vital::simple_camera_perspective> &cameras,
                 const vital::vector_3d &X);


/// Check that at least one pair of rays has cos(angle) less than or equal to cos_ang_thresh
/**
 * \param[in] cameras is the set of cameras that view X
 * \param[in] X is the landmark projected into the cameras
 * \param[in] cos_ang_thresh cosine of the angle threshold
 * \returns true if at least one pair of rays has cos(angle) <= cos_ang_thresh
 */
KWIVER_ALGO_CORE_EXPORT
bool
bundle_angle_is_at_least(const std::vector<vital::simple_camera_perspective> &cameras,
                         const vital::vector_3d &X,
                         double cos_ang_thresh);


/// Compute the reprojection error of lm projected by cam compared to f
/**
 * \param [in] cam is the camera used for projection
 * \param [in] lm is the landmark projected into the camera
 * \param [in] f is the measured feature point location
 * \returns the distance between the projected lm and f in image space
 */
inline
double
reprojection_error(const vital::camera& cam,
                   const vital::landmark& lm,
                   const vital::feature& f)
{
  return std::sqrt(reprojection_error_sqr(cam, lm, f));
}


/// Compute a vector of all reprojection errors in the data
/**
 * \param [in] cameras is the map of frames/cameras used for projection
 * \param [in] landmarks is the map ids/landmarks projected into the cameras
 * \param [in] tracks is the set of tracks providing measurements
 * \returns a vector containing one reprojection error for each observation
 *          (i.e. track state) that has a corresponding camera and landmark
 */
KWIVER_ALGO_CORE_EXPORT
std::vector<double>
reprojection_errors(const std::map<vital::frame_id_t, vital::camera_sptr>& cameras,
                    const std::map<vital::landmark_id_t, vital::landmark_sptr>& landmarks,
                    const std::vector< vital::track_sptr>& tracks);


/// Compute a vector of all reprojection errors in the data
/**
 * \param [in] cameras is the map of frames/cameras used for projection
 * \param [in] landmarks is the map ids/landmarks projected into the cameras
 * \param [in] tracks is the set of tracks providing measurements
 * \returns a map containing one reprojection error rms value per camera mapped by the
 *          the cameras' frame ids
 */
KWIVER_ALGO_CORE_EXPORT
std::map<vital::frame_id_t, double>
reprojection_rmse_by_cam(const vital::camera_map::map_camera_t& cameras,
                         const vital::landmark_map::map_landmark_t& landmarks,
                         const std::vector<vital::track_sptr>& tracks);


/// Compute the Root-Mean-Square-Error (RMSE) of the reprojections
/**
 * \param [in] cameras is the map of frames/cameras used for projection
 * \param [in] landmarks is the map ids/landmarks projected into the cameras
 * \param [in] tracks is the set of tracks providing measurements
 * \returns the RMSE between all landmarks projected by all cameras that have
 *          corresponding image measurements provided by the tracks
 */
KWIVER_ALGO_CORE_EXPORT
double
reprojection_rmse(const std::map<vital::frame_id_t, vital::camera_sptr>& cameras,
                  const std::map<vital::landmark_id_t, vital::landmark_sptr>& landmarks,
                  const std::vector<vital::track_sptr>& tracks);


/// Compute the median of the reprojection errors
/**
 * \param [in] cameras is the map of frames/cameras used for projection
 * \param [in] landmarks is the map ids/landmarks projected into the cameras
 * \param [in] tracks is the set of tracks providing measurements
 * \returns the median reprojection error between all landmarks projected by
 *          all cameras that have corresponding image measurements provided
 *          by the tracks
 */
KWIVER_ALGO_CORE_EXPORT
double
reprojection_median_error(const std::map<vital::frame_id_t, vital::camera_sptr>& cameras,
                          const std::map<vital::landmark_id_t, vital::landmark_sptr>& landmarks,
                          const std::vector<vital::track_sptr>& tracks);


/// Compute the median of the reprojection errors
/**
 * \param [in] cameras is the map of frames/cameras used for projection
 * \param [in] landmarks is the map ids/landmarks projected into the cameras
 * \param [in] tracks is the set of tracks providing measurements
 * \returns the median reprojection error between all landmarks projected by
 *          all cameras that have corresponding image measurements provided
 *          by the tracks
 */
KWIVER_ALGO_CORE_EXPORT
double
reprojection_median_error(const std::map<vital::frame_id_t, vital::camera_sptr>& cameras,
                          const std::map<vital::landmark_id_t, vital::landmark_sptr>& landmarks,
                          const std::vector<vital::track_sptr>& tracks);


} // end namespace arrows
} // end namespace kwiver


#endif // ALGORITHMS_METRICS_H_
