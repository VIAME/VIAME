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
 * \brief Implementation of core triangulate landmarks algorithm
 */

#include "triangulate_landmarks.h"

#include <set>
#include <random>
#include <ctime>

#include <arrows/core/triangulate.h>
#include <arrows/core/metrics.h>

#include <vital/math_constants.h>

namespace kwiver {
namespace arrows {
namespace core {

// Private implementation class
class triangulate_landmarks::priv
{
public:
  // Constructor
  priv()
    : m_homogeneous(false),
      m_ransac(true),
      m_min_angle_deg(1.0f),
      m_inlier_threshold_pixels(2.0),
      m_inlier_threshold_pixels_sq(m_inlier_threshold_pixels*m_inlier_threshold_pixels),
      m_frac_track_inliers_to_keep_triangulated_point(0.5f),
      m_max_ransac_samples(20),
      m_conf_thresh(0.99)
  {
  }

  priv(const priv& other)
    : m_homogeneous(other.m_homogeneous),
      m_ransac(other.m_ransac),
      m_min_angle_deg(other.m_min_angle_deg),
      m_inlier_threshold_pixels(other.m_inlier_threshold_pixels),
      m_inlier_threshold_pixels_sq(other.m_inlier_threshold_pixels_sq),
      m_frac_track_inliers_to_keep_triangulated_point(
        other.m_frac_track_inliers_to_keep_triangulated_point),
      m_max_ransac_samples(other.m_max_ransac_samples)
  {
  }

  vital::vector_3d
  ransac_triangulation(const std::vector<vital::simple_camera_perspective> &lm_cams,
                       const std::vector<vital::vector_2d> &lm_image_pts,
                       int &best_inlier_count,
                       vital::vector_3d const* guess) const;

  bool
  triangulate(const std::vector<vital::simple_camera_perspective> &lm_cams,
              const std::vector<vital::vector_2d> &lm_image_pts,
              vital::vector_3d &pt3d) const;

  // use the homogeneous method for triangulation
  bool m_homogeneous;
  bool m_ransac;
  float m_min_angle_deg;
  float m_inlier_threshold_pixels;
  float m_inlier_threshold_pixels_sq;
  float m_frac_track_inliers_to_keep_triangulated_point;
  int m_max_ransac_samples;
  double m_conf_thresh;
};


// Constructor
triangulate_landmarks
::triangulate_landmarks()
: d_(new priv)
{
  attach_logger( "arrows.core.triangulate_landmarks" );
}


// Copy Constructor
triangulate_landmarks
::triangulate_landmarks(const triangulate_landmarks& other)
: d_(new priv(*other.d_))
{
}


// Destructor
triangulate_landmarks
::~triangulate_landmarks()
{
}


// Get this alg's \link vital::config_block configuration block \endlink
vital::config_block_sptr
triangulate_landmarks
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config
    = vital::algo::triangulate_landmarks::get_configuration();

  // Bad frame detection parameters
  config->set_value("homogeneous", d_->m_homogeneous,
                    "Use the homogeneous method for triangulating points. "
                    "The homogeneous method can triangulate points at or near "
                    "infinity and discard them.");

  config->set_value("ransac", d_->m_ransac,"Use RANSAC in triangulating the points");

  config->set_value("min_angle_deg", d_->m_min_angle_deg,
                    "minimum angle required to triangulate a point.");


  config->set_value("inlier_threshold_pixels", d_->m_inlier_threshold_pixels,
                   "reprojection error threshold in pixels.");

  config->set_value("frac_track_inliers_to_keep_triangulated_point",
                    d_->m_frac_track_inliers_to_keep_triangulated_point,
                    "fraction of measurements in track that must be inliers to "
                    "keep the triangulated point");

  config->set_value("max_ransac_samples",
                    d_->m_max_ransac_samples,
                    "maximum number of samples to take in RANSAC triangulation");

  config->set_value("ransac_confidence_threshold",
                    d_->m_conf_thresh,
                    "RANSAC sampling terminates when this confidences in the "
                    "solution is reached.");

  return config;
}


// Set this algorithm's properties via a config block
void
triangulate_landmarks
::set_configuration(vital::config_block_sptr in_config)
{
  // Starting with our generated config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);

  // Settings for bad frame detection
  d_->m_homogeneous = config->get_value<bool>("homogeneous", d_->m_homogeneous);

  d_->m_ransac = config->get_value<bool>("ransac", d_->m_ransac);

  d_->m_min_angle_deg = config->get_value<float>("min_angle_deg", d_->m_min_angle_deg);

  d_->m_inlier_threshold_pixels =
    config->get_value<float>("inlier_threshold_pixels",
                             d_->m_inlier_threshold_pixels);

  d_->m_inlier_threshold_pixels_sq = d_->m_inlier_threshold_pixels * d_->m_inlier_threshold_pixels;

  d_->m_frac_track_inliers_to_keep_triangulated_point =
    config->get_value<float>("frac_track_inliers_to_keep_triangulated_point",
                             d_->m_frac_track_inliers_to_keep_triangulated_point);

  d_->m_max_ransac_samples =
    config->get_value<int>("max_ransac_samples", d_->m_max_ransac_samples);

  d_->m_conf_thresh =
    config->get_value<double>("ransac_confidence_threshold", d_->m_conf_thresh);
}


// Check that the algorithm's currently configuration is valid
bool
triangulate_landmarks
::check_configuration(vital::config_block_sptr config) const
{
  return true;
}

bool
triangulate_landmarks::priv
::triangulate(const std::vector<vital::simple_camera_perspective> &lm_cams,
              const std::vector<vital::vector_2d> &lm_image_pts,
              vital::vector_3d &pt3d) const
{
  if (m_homogeneous)
  {
    vital::vector_4d pt4d = kwiver::arrows::triangulate_homog(lm_cams, lm_image_pts);
    if (std::abs(pt4d[3]) < 1e-6)
    {
      pt3d.setZero();
      return false;
    }
    pt3d = pt4d.segment(0, 3) / pt4d[3];
  }
  else
  {
    pt3d = kwiver::arrows::triangulate_inhomog(lm_cams, lm_image_pts);
  }

  return true;
}


/// Triangulate the landmark with RANSAC robust estimation
vital::vector_3d
triangulate_landmarks::priv
::ransac_triangulation(const std::vector<vital::simple_camera_perspective> &lm_cams,
                       const std::vector<vital::vector_2d> &lm_image_pts,
                       int &best_inlier_count,
                       vital::vector_3d const* guess) const
{
  double conf = 0;
  std::vector<vital::simple_camera_perspective> cam_sample(2);
  std::vector<vital::vector_2d> proj_sample(2);
  vital::vector_3d best_pt3d;
  best_inlier_count = 0;
  double best_inlier_ratio = 0;

  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::time_t time_res = std::time(nullptr);
  gen.seed(time_res);
  std::uniform_int_distribution<> dis(0, int(lm_cams.size() - 1));

  best_pt3d.setZero();

  if (lm_cams.size() < 2)
  {

    return best_pt3d;
  }

  int s_idx[2];
  vital::landmark_d lm;
  vital::feature_d f;

  for (int num_samples = 1;
       num_samples <= m_max_ransac_samples && conf < m_conf_thresh;
       ++num_samples)
  {
    //pick two random points
    int inlier_count = 0;
    s_idx[0] = dis(gen);
    s_idx[1] = s_idx[0];
    while (s_idx[0] == s_idx[1])
    {
      s_idx[1] = dis(gen);
    }

    cam_sample[0] = lm_cams[s_idx[0]];
    cam_sample[1] = lm_cams[s_idx[1]];
    proj_sample[0] = lm_image_pts[s_idx[0]];
    proj_sample[1] = lm_image_pts[s_idx[1]];

    vital::vector_3d pt3d;
    if (guess != NULL && num_samples == 1 )
    {
      pt3d = *guess;
    }
    else
    {
      if (!triangulate(cam_sample, proj_sample, pt3d))
      {
        continue;
      }
    }

    lm.set_loc(pt3d);
    //count inliers
    for (unsigned int idx = 0; idx < lm_cams.size(); ++idx)
    {
      auto depth = lm_cams[idx].depth(lm.loc());
      if (depth <= 0)
      {
        continue;
      }
      f.set_loc(lm_image_pts[idx]);
      double reproj_err_sq = reprojection_error_sqr(lm_cams[idx], lm, f);
      if (reproj_err_sq < m_inlier_threshold_pixels_sq)
      {
        ++inlier_count;
      }
    }

    if (inlier_count > best_inlier_count)
    {
      best_inlier_count = inlier_count;
      best_pt3d = pt3d;
      best_inlier_ratio = (double)best_inlier_count / (double)lm_cams.size();
    }

    conf = 1.0 - pow(1.0 - pow(best_inlier_ratio, 2.0), double(num_samples));
    if (lm_cams.size() == 2)
    {
      break;  //2 choose 2 only happens one way
    }
  }

  return best_pt3d;
}


void
triangulate_landmarks
::triangulate(vital::camera_map_sptr cameras,
              vital::feature_track_set_sptr tracks,
              vital::landmark_map_sptr& landmarks) const
{
  vital::track_map_t track_map;
  auto tks = tracks->tracks();
  for (auto const&t : tks)
  {
    track_map[t->id()] = t;
  }
  triangulate(cameras, track_map, landmarks);
}

// Triangulate the landmark locations given sets of cameras and tracks
void
triangulate_landmarks
::triangulate(vital::camera_map_sptr cameras,
              vital::track_map_t track_map,
              vital::landmark_map_sptr& landmarks) const
{
  using namespace kwiver;
  if( !cameras || !landmarks )
  {
    // TODO throw an exception for missing input data
    return;
  }

  typedef vital::camera_map::map_camera_t map_camera_t;
  typedef vital::landmark_map::map_landmark_t map_landmark_t;

  // extract data from containers
  map_camera_t cams = cameras->cameras();
  map_landmark_t lms = landmarks->landmarks();

  // the set of landmark ids which failed to triangulate
  std::set<vital::landmark_id_t> failed_landmarks;
  std::set<vital::landmark_id_t> failed_outlier, failed_angle;


  //minimum triangulation angle
  double thresh_triang_cos_ang = cos(vital::deg_to_rad * d_->m_min_angle_deg);

  std::vector<vital::simple_camera_perspective> lm_cams;
  std::vector<vital::simple_camera_rpc> lm_cams_rpc;
  std::vector<vital::vector_2d> lm_image_pts;
  std::vector<vital::feature_track_state_sptr> lm_features;

  map_landmark_t triangulated_lms;
  for(const map_landmark_t::value_type& p : lms)
  {
    lm_cams.clear();
    lm_cams_rpc.clear();
    lm_image_pts.clear();
    lm_features.clear();
    // extract the cameras and image points for this landmarks
    auto lm_observations = unsigned{ 0 };

    // get the corresponding track
    vital::track_map_t::const_iterator t_itr = track_map.find(p.first);
    if (t_itr == track_map.end())
    {
      // there is no track for the provided landmark
      failed_landmarks.insert(p.first);
      continue;
    }
    const vital::track& t = *t_itr->second;

    for (vital::track::history_const_itr tsi = t.begin(); tsi != t.end(); ++tsi)
    {
      auto fts = std::static_pointer_cast<vital::feature_track_state>(*tsi);
      if (!fts && !fts->feature)
      {
        // there is no valid feature for this track state
        continue;
      }
      map_camera_t::const_iterator c_itr = cams.find((*tsi)->frame());
      if (c_itr == cams.end())
      {
        // there is no camera for this track state.
        continue;
      }
      auto cam_ptr =
        std::dynamic_pointer_cast<vital::camera_perspective>(c_itr->second);
      if (cam_ptr)
      {
        lm_cams.push_back(vital::simple_camera_perspective(*cam_ptr));
      }
      auto rpc_ptr =
        std::dynamic_pointer_cast<vital::camera_rpc>(c_itr->second);
      if (rpc_ptr)
      {
        lm_cams_rpc.push_back( vital::simple_camera_rpc( *rpc_ptr ) );
      }
      if (cam_ptr || rpc_ptr)
      {
        lm_image_pts.push_back(fts->feature->loc());
        lm_features.push_back(fts);
        ++lm_observations;
      }
    }

    // if we found at least two views of this landmark, triangulate
    if (lm_cams.size() > 1)
    {
      int inlier_count = 0;
      vital::vector_3d pt3d;
      if (d_->m_ransac)
      {
        vital::vector_3d lm_cur_pt3d = p.second->loc();
        auto triang_guess = &lm_cur_pt3d;
        if (lm_cur_pt3d.x() == 0 && lm_cur_pt3d.y() == 0 && lm_cur_pt3d.z() == 0)
        {
          triang_guess = NULL;
        }

        pt3d = d_->ransac_triangulation(lm_cams, lm_image_pts, inlier_count, triang_guess);
        if (inlier_count < lm_image_pts.size() * d_->m_frac_track_inliers_to_keep_triangulated_point)
        {
          failed_landmarks.insert(p.first);
          failed_outlier.insert(p.first);
          continue;
        }
      }
      else
      {
        if (!d_->triangulate(lm_cams, lm_image_pts, pt3d))
        {
          failed_landmarks.insert(p.first);
          continue;
        }
        //test if the point is behind any of the cameras
        bool behind = false;
        for (auto const& lm_cam : lm_cams)
        {
          auto depth = lm_cam.depth(pt3d);
          if (depth <= 0)
          {
            behind = true;
            break;
          }
        }
        if (behind)
        {
          for (auto lm_feat : lm_features)
          {
            lm_feat->inlier = false;
          }
          failed_landmarks.insert(p.first);
          continue;
        }
      }

      //set inlier/outlier states for the measurements
      for (unsigned int idx = 0; idx < lm_cams.size(); ++idx)
      {
        vital::landmark_d lm;
        lm.set_loc(pt3d);
        double reproj_err_sq = reprojection_error_sqr(lm_cams[idx], lm, *lm_features[idx]->feature);
        if (reproj_err_sq < d_->m_inlier_threshold_pixels_sq)
        {
          lm_features[idx]->inlier = true;
        }
        else
        {
          lm_features[idx]->inlier = false;
        }
      }
      if (!pt3d.allFinite())
      {
        for (auto lm_feat : lm_features)
        {
          lm_feat->inlier = false;
        }
        failed_landmarks.insert(p.first);
        continue;
      }

      double triang_cos_ang = kwiver::arrows::bundle_angle_max(lm_cams, pt3d);
      bool bad_triangulation = triang_cos_ang > thresh_triang_cos_ang;
      if (bad_triangulation)
      {
        failed_landmarks.insert(p.first);
        failed_angle.insert(p.first);
        for (auto lm_feat : lm_features)
        {
          lm_feat->inlier = false;
        }
        continue;
      }


      std::shared_ptr<vital::landmark_d> lm;
      // if the landmark already exists, copy it
      if (p.second)
      {
        lm = std::make_shared<vital::landmark_d>(*p.second);  //automatically copies the tracks_ data
        lm->set_loc(pt3d);
      }
      // otherwise make a new landmark
      else
      {
        lm = std::make_shared<vital::landmark_d>(pt3d);
      }
      lm->set_cos_observation_angle(triang_cos_ang);
      lm->set_observations(lm_observations);
      triangulated_lms[p.first] = lm;
    }
    else if ( lm_cams_rpc.size() > 1 )
    {
      vital::vector_3d pt3d =
        kwiver::arrows::triangulate_rpc(lm_cams_rpc, lm_image_pts);

      // TODO: is there a way to check for bad triangulations for RPC cameras?
      auto lm = std::make_shared<vital::landmark_d>(*p.second);
      lm->set_loc(pt3d);
      lm->set_observations(lm_observations);
      triangulated_lms[p.first] = lm;
    }
  }
  if( !failed_landmarks.empty() )
  {
    LOG_WARN( logger(),
              "failed to triangulate " << failed_landmarks.size()
              << " with " << failed_angle.size() << " for angle, "
              << failed_outlier.size() << " outliers");
  }
  landmarks = vital::landmark_map_sptr(new vital::simple_landmark_map(triangulated_lms));
}

} // end namespace core
} // end namespace arrows
} // end namespace kwiver
