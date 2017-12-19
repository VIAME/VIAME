/*ckwg +29
 * Copyright 2014-2017 by Kitware, Inc.
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

#define _USE_MATH_DEFINES
#include <math.h>

#include <vital/logger/logger.h>

#include <arrows/core/triangulate.h>
#include <arrows/core/metrics.h>

namespace kwiver {
namespace arrows {
namespace core {

/// Private implementation class
class triangulate_landmarks::priv
{
public:
  /// Constructor
  priv()
    : homogeneous(false),
      min_angle_deg(1.0f),
      m_logger( vital::get_logger( "arrows.core.triangulate_landmarks" ))
  {
  }

  priv(const priv& other)
    : homogeneous(other.homogeneous),
      min_angle_deg(other.min_angle_deg),
      m_logger( vital::get_logger( "arrows.core.triangulate_landmarks" ))
  {
  }

  vital::vector_3d
  ransac_triangulation(const std::vector<vital::simple_camera> &lm_cams,
    const std::vector<vital::vector_2d> &lm_image_pts,
    float inlier_threshold) const;

  /// use the homogeneous method for triangulation
  bool homogeneous;
  float min_angle_deg;
  /// logger handle
  vital::logger_handle_t m_logger;
};


/// Constructor
triangulate_landmarks
::triangulate_landmarks()
: d_(new priv)
{
}


/// Copy Constructor
triangulate_landmarks
::triangulate_landmarks(const triangulate_landmarks& other)
: d_(new priv(*other.d_))
{
}


/// Destructor
triangulate_landmarks
::~triangulate_landmarks()
{
}


/// Get this alg's \link vital::config_block configuration block \endlink
vital::config_block_sptr
triangulate_landmarks
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config
    = vital::algo::triangulate_landmarks::get_configuration();

  // Bad frame detection parameters
  config->set_value("homogeneous", d_->homogeneous,
                    "Use the homogeneous method for triangulating points. "
                    "The homogeneous method can triangulate points at or near "
                    "infinity and discard them.");

  config->set_value("min_angle_deg", d_->min_angle_deg,
                    "minimum angle required to triangulate a point.");

  return config;
}


/// Set this algorithm's properties via a config block
void
triangulate_landmarks
::set_configuration(vital::config_block_sptr in_config)
{
  // Starting with our generated config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);

  // Settings for bad frame detection
  d_->homogeneous = config->get_value<bool>("homogeneous", d_->homogeneous);

  d_->min_angle_deg = config->get_value<float>("min_angle_deg", d_->min_angle_deg);
}


/// Check that the algorithm's currently configuration is valid
bool
triangulate_landmarks
::check_configuration(vital::config_block_sptr config) const
{
  return true;
}

/// Triangulate the landmark with RANSAC robust estimation 
vital::vector_3d
triangulate_landmarks::priv
::ransac_triangulation(const std::vector<vital::simple_camera> &lm_cams,
  const std::vector<vital::vector_2d> &lm_image_pts,
  float inlier_threshold) const
{
  int max_samples = 20;
  double conf_thresh = 0.99;
  double conf = 0;
  std::vector<vital::simple_camera> cam_sample;
  std::vector<vital::vector_2d> proj_sample;
  cam_sample.resize(2);
  proj_sample.resize(2);
  vital::vector_3d best_pt3d;
  int best_inlier_count = -1;
  double best_inlier_ratio = 0;
  int num_samples = 0;

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

  while (conf < conf_thresh)
  {
    ++num_samples;
    //pick two random points
    int inlier_count = 0;
    s_idx[0] = dis(gen);
    s_idx[1] = s_idx[0];
    while (s_idx[0] == s_idx[1])
    {
      s_idx[1] = dis(gen);
    }
    for (int i = 0; i < 2; ++i)
    {
      cam_sample[i] = lm_cams[s_idx[i]];
      proj_sample[i] = lm_image_pts[s_idx[i]];
    }
    vital::vector_3d pt3d = kwiver::arrows::triangulate_inhomog(cam_sample, proj_sample);

    lm.set_loc(pt3d);
    //count inliers
    for (int idx = 0; idx < lm_cams.size(); ++idx)
    {
      f.set_loc(lm_image_pts[idx]);
      double reproj_err = reprojection_error(lm_cams[idx], lm, f);
      if (reproj_err < inlier_threshold)
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
    if (num_samples >= max_samples)
    {
      break;
    }
  }

  return best_pt3d;
}

/// Triangulate the landmark locations given sets of cameras and tracks
void
triangulate_landmarks
::triangulate(vital::camera_map_sptr cameras,
              vital::feature_track_set_sptr tracks,
              vital::landmark_map_sptr& landmarks) const
{
  using namespace kwiver;
  if( !cameras || !landmarks || !tracks )
  {
    // TODO throw an exception for missing input data
    return;
  }
  typedef vital::camera_map::map_camera_t map_camera_t;
  typedef vital::landmark_map::map_landmark_t map_landmark_t;

  // extract data from containers
  map_camera_t cams = cameras->cameras();
  map_landmark_t lms = landmarks->landmarks();
  std::vector<vital::track_sptr> trks = tracks->tracks();

  // build a track map by id
  typedef std::map<vital::track_id_t, vital::track_sptr> track_map_t;
  track_map_t track_map;
  for(const vital::track_sptr& t : trks)
  {
    track_map[t->id()] = t;
  }

  // the set of landmark ids which failed to triangulate
  std::set<vital::landmark_id_t> failed_landmarks;
  std::set<vital::landmark_id_t> failed_behind, failed_angle;


  //minimum triangulation angle
  double thresh_triang_cos_ang = cos((M_PI / 180.0) * d_->min_angle_deg);

  map_landmark_t triangulated_lms;
  for(const map_landmark_t::value_type& p : lms)
  {
    // get the corresponding track
    track_map_t::const_iterator t_itr = track_map.find(p.first);
    if (t_itr == track_map.end())
    {
      // there is no track for the provided landmark
      failed_landmarks.insert(p.first);
      continue;
    }
    const vital::track& t = *t_itr->second;

    // extract the cameras and image points for this landmarks
    std::vector<vital::simple_camera> lm_cams;
    std::vector<vital::vector_2d> lm_image_pts;

    auto lm_observations = unsigned{0};
    for (vital::track::history_const_itr tsi = t.begin(); tsi != t.end(); ++tsi)
    {
      auto fts = std::dynamic_pointer_cast<vital::feature_track_state>(*tsi);
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
      lm_cams.push_back(vital::simple_camera(*c_itr->second));
      lm_image_pts.push_back(fts->feature->loc());
      ++lm_observations;
    }

    // if we found at least two views of this landmark, triangulate
    if (lm_cams.size() > 1)
    {
      bool bad_triangulation = false;
      vital::vector_3d pt3d;
      if (d_->homogeneous)
      {
        vital::vector_4d pt4d = kwiver::arrows::triangulate_homog(lm_cams, lm_image_pts);
        if (std::abs(pt4d[3]) < 1e-6)
        {
          bad_triangulation = true;
          failed_landmarks.insert(p.first);
        }
        pt3d = pt4d.segment(0,3) / pt4d[3];
      }
      else
      {
        pt3d = d_->ransac_triangulation(lm_cams, lm_image_pts,2.0);
      }
      if (pt3d.allFinite())
      {
        for(vital::simple_camera const& cam : lm_cams)
        {
          bad_triangulation = true;
          failed_landmarks.insert(p.first);
          failed_behind.insert(p.first);
          break;
        }
      }
      if (!bad_triangulation)
      {
        double triang_cos_ang = kwiver::arrows::bundle_angle_max(lm_cams, pt3d);
        bad_triangulation = triang_cos_ang > thresh_triang_cos_ang;
        if (bad_triangulation)
        {
          failed_angle.insert(p.first);
        }
      }
      if( !bad_triangulation )
      {
        auto lm = std::make_shared<vital::landmark_d>(*p.second);
        lm->set_loc(pt3d);
        lm->set_observations(lm_observations);
        triangulated_lms[p.first] = lm;
      }
    }
  }
  if( !failed_landmarks.empty() )
  {
    LOG_WARN(d_->m_logger,
      "failed to triangulate " << failed_angle.size() << " landmarks for angle, " << failed_behind.size() << " landmarks behind");
  }
  landmarks = vital::landmark_map_sptr(new vital::simple_landmark_map(triangulated_lms));
}

} // end namespace core
} // end namespace arrows
} // end namespace kwiver
