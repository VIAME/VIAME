// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of MVG evaluation metric functions.
 */

#include "metrics.h"
#include <vital/types/feature_track_set.h>
#include <limits>

namespace kwiver {
namespace arrows {
namespace mvg {

using namespace kwiver::vital;

/// Compute the reprojection error vector of lm projected by cam compared to f
vector_2d
reprojection_error_vec(const camera& cam,
                       const landmark& lm,
                       const feature& f)
{
  vector_2d pt(0.0, 0.0);
  pt = cam.project(lm.loc());
  return pt - f.loc();
}

/// Compute the maximum angle between the rays from X to each camera center
double
bundle_angle_max(const std::vector<vital::simple_camera_perspective> &cameras,
                 const vital::vector_3d &X)
{
  double min_cos_ang = std::numeric_limits<double>::infinity();
  std::vector<vital::vector_3d> rays(cameras.size());

  for(size_t i = 0; i < cameras.size(); ++i)
  {
    auto const& cam_i = cameras[i];
    const vital::vector_3d c_i(cam_i.center().cast<double>());
    rays[i] = (c_i - X).normalized();

    for(size_t j = 0; j < i; ++j)
    {
      // the second camera
      double cos_ang = rays[i].dot(rays[j]);

      if (cos_ang <= min_cos_ang)
      {
        min_cos_ang = cos_ang;
      }
    }
  }
  return min_cos_ang;
}

/// Check that at least one pair of rays has cos(angle) less than or equal to cos_ang_thresh
bool
bundle_angle_is_at_least(const std::vector<vital::simple_camera_perspective> &cameras,
                         const vital::vector_3d &X,
                         double cos_ang_thresh)
{
  std::vector<vital::vector_3d> rays(cameras.size());

  for (size_t i = 0; i < cameras.size(); ++i)
  {
    auto const& cam_i = cameras[i];
    const vital::vector_3d c_i(cam_i.center().cast<double>());
    rays[i] = (c_i - X).normalized();

    for (size_t j = 0; j < i; ++j)
    {
      // the second camera
      double cos_ang = rays[i].dot(rays[j]);

      if (cos_ang <= cos_ang_thresh)
      {
        return true;
      }
    }
  }
  return false;
}

/// Compute a vector of all reprojection errors in the data
std::vector<double>
reprojection_errors(const std::map<frame_id_t, camera_sptr>& cameras,
                    const std::map<landmark_id_t, landmark_sptr>& landmarks,
                    const std::vector<track_sptr>& tracks)
{
  typedef std::map<landmark_id_t, landmark_sptr>::const_iterator lm_map_itr_t;
  typedef std::map<frame_id_t, camera_sptr>::const_iterator cam_map_itr_t;
  std::vector<double> errors;
  for(const track_sptr& t : tracks)
  {
    lm_map_itr_t lmi = landmarks.find(t->id());
    if (lmi == landmarks.end() || !lmi->second)
    {
      // no landmark corresponding to this track
      continue;
    }
    const landmark& lm = *lmi->second;
    for( track::history_const_itr tsi = t->begin(); tsi != t->end(); ++tsi)
    {
      auto fts = std::dynamic_pointer_cast<feature_track_state>(*tsi);
      if (!fts || !fts->feature)
      {
        // no feature for this track state.
        continue;
      }
      if (!fts->inlier)
      {
        continue; //feature is not marked as an inlier so skip it
      }
      const feature& feat = *fts->feature;
      cam_map_itr_t ci = cameras.find((*tsi)->frame());
      if (ci == cameras.end() || !ci->second)
      {
        // no camera corresponding to this track state
        continue;
      }
      const camera& cam = *ci->second;
      errors.push_back(reprojection_error(cam, lm, feat));
    }
  }
  return errors;
}

/// Compute the per camera Root-Mean-Square-Error (RMSE) of the reprojections
std::map<frame_id_t, double>
reprojection_rmse_by_cam(const vital::camera_map::map_camera_t& cameras,
                         const vital::landmark_map::map_landmark_t& landmarks,
                         const std::vector<track_sptr>& tracks)
{
  typedef std::map<landmark_id_t, landmark_sptr>::const_iterator lm_map_itr_t;
  typedef std::map<frame_id_t, camera_sptr>::const_iterator cam_map_itr_t;

  struct err_vals {
    unsigned int num_obs;
    double sum_error_sq;
    err_vals() :
      num_obs(0), sum_error_sq(0) {}

    err_vals(unsigned int num_obs_, double sum_error_sq_) :
      num_obs(num_obs_), sum_error_sq(sum_error_sq_) {}
  };

  std::map<frame_id_t, err_vals> cam_errors;

  for (const track_sptr& t : tracks)
  {
    lm_map_itr_t lmi = landmarks.find(t->id());
    if (lmi == landmarks.end() || !lmi->second)
    {
      // no landmark corresponding to this track
      continue;
    }
    const landmark& lm = *lmi->second;
    for (track::history_const_itr tsi = t->begin(); tsi != t->end(); ++tsi)
    {
      auto frame_num = (*tsi)->frame();

      auto fts = std::dynamic_pointer_cast<feature_track_state>(*tsi);
      if (!fts || !fts->feature)
      {
        // no feature for this track state.
        continue;
      }
      if (!fts->inlier)
      {
        continue; //feature is not marked as an inlier so skip it
      }

      const feature& feat = *fts->feature;
      cam_map_itr_t ci = cameras.find(frame_num);
      if (ci == cameras.end() || !ci->second)
      {
        // no camera corresponding to this track state
        continue;
      }
      const camera& cam = *ci->second;

      auto rpe = reprojection_error_sqr(cam, lm, feat);
      auto ce_it = cam_errors.find(ci->first);
      if (ce_it == cam_errors.end())
      {
        cam_errors[ci->first] = err_vals(1.0, rpe);
      }
      else
      {
        ce_it->second.num_obs += 1;
        ce_it->second.sum_error_sq += rpe;
      }
    }
  }

  std::map<frame_id_t, double> ret_errs;
  for (auto& err : cam_errors)
  {
    if (err.second.num_obs > 0)
    {
      ret_errs[err.first] = std::sqrt(err.second.sum_error_sq / static_cast<double>(err.second.num_obs));
    }
  }
  return ret_errs;
}

/// Compute the Root-Mean-Square-Error (RMSE) of the reprojections
double
reprojection_rmse(const std::map<frame_id_t, camera_sptr>& cameras,
                  const std::map<landmark_id_t, landmark_sptr>& landmarks,
                  const std::vector<track_sptr>& tracks)
{
  typedef std::map<landmark_id_t, landmark_sptr>::const_iterator lm_map_itr_t;
  typedef std::map<frame_id_t, camera_sptr>::const_iterator cam_map_itr_t;
  double error_sum = 0.0;
  unsigned num_obs = 0;
  for(const track_sptr& t : tracks)
  {
    lm_map_itr_t lmi = landmarks.find(t->id());
    if (lmi == landmarks.end() || !lmi->second)
    {
      // no landmark corresponding to this track
      continue;
    }
    const landmark& lm = *lmi->second;
    for( track::history_const_itr tsi = t->begin(); tsi != t->end(); ++tsi)
    {
      auto fts = std::dynamic_pointer_cast<feature_track_state>(*tsi);
      if (!fts || !fts->feature)
      {
        // no feature for this track state.
        continue;
      }
      if (!fts->inlier)
      {
        continue; //feature is not marked as an inlier so skip it
      }
      const feature& feat = *fts->feature;
      cam_map_itr_t ci = cameras.find((*tsi)->frame());
      if (ci == cameras.end() || !ci->second)
      {
        // no camera corresponding to this track state
        continue;
      }
      const camera& cam = *ci->second;
      error_sum += reprojection_error_sqr(cam, lm, feat);
      ++num_obs;
    }
  }
  return std::sqrt(error_sum / num_obs);
}

/// Compute the median of the reprojection errors
double
reprojection_median_error(const std::map<frame_id_t, camera_sptr>& cameras,
                          const std::map<landmark_id_t, landmark_sptr>& landmarks,
                          const std::vector<track_sptr>& tracks)
{
  std::vector<double> errors = reprojection_errors(cameras, landmarks, tracks);
  std::nth_element(errors.begin(),
                   errors.begin() + errors.size()/2,
                   errors.end());
  return errors[errors.size()/2];
}

} // end namespace mvg
} // end namespace arrows
} // end namespace kwiver
