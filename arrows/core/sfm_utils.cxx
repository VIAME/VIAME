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
* \brief Implementation of kwiver::arrows::sfm_utils utiltiy functions for
* structure from motion.
*/

#include "sfm_utils.h"

#include <vital/vital_types.h>
#include <vital/types/feature_track_set.h>
#include <arrows/core/metrics.h>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {

frame_coverage_vec
image_coverages(const track_set_sptr tracks,
  const kwiver::vital::landmark_map::map_landmark_t& lms,
  const camera_map::map_camera_t& cams )
{
  const int mask_w(16);
  const int mask_h(16);
  typedef Eigen::Matrix<int, mask_h, mask_w> vis_mask;
  typedef std::map<frame_id_t, vis_mask> frame_map_t;
  frame_map_t frame_masks;
  //calculate feature distribution masks for each candidate image.
  auto tks = tracks->tracks();

  struct im_dims {
    int w;
    int h;
  };

  //Get the dimension of each image.  Each image may be a different size.
  std::map<frame_id_t,im_dims> frame_dims;

  for (auto cam : cams)
  {
    if (cam.second)
    {
      im_dims dims;
      //this assumes the principal point is at the center of the image.
      dims.w = int(cam.second->intrinsics()->principal_point().x() * 2.0);
      dims.h = int(cam.second->intrinsics()->principal_point().y() * 2.0);
      frame_dims[cam.first] = dims;
    }
  }


  for (const track_sptr& t : tks)
  {
    if (lms.find(t->id()) != lms.end())
    {
      for (auto tsi = t->begin(); tsi != t->end(); ++tsi)
      {
        auto fid = (*tsi)->frame();
        auto fd_it = frame_dims.find(fid);
        if (fd_it == frame_dims.end())
        {
          continue;
        }

        feature_track_state_sptr fts =
          std::dynamic_pointer_cast<feature_track_state>(*tsi);
        if (!fts || !fts->feature)
        {
          continue;
        }

        //calculate the mask image location
        vector_2d x = fts->feature->loc();
        int mask_row = int(std::max<float>(0.0f,
          std::min<float>(mask_h*float(x.y()) / fd_it->second.h, mask_h - 1)));
        int mask_col = int(std::max<float>(0.0f,
          std::min<float>(mask_w*float(x.x()) / fd_it->second.w, mask_w - 1)));

        auto fm_it = frame_masks.find(fid);
        if (fm_it == frame_masks.end())
        {
          vis_mask msk;
          msk.setZero();
          msk(mask_row, mask_col) = 1;
          frame_masks.insert(std::pair<frame_id_t, vis_mask>(fid, msk));
        }
        else
        {
          fm_it->second(mask_row, mask_col) = 1;
        }
      }
    }
  }

  //ok, now we have calculated all the masks.
  frame_coverage_vec ret;
  for (auto fid : frame_dims)
  {
    auto fm_it = frame_masks.find(fid.first);
    if (fm_it == frame_masks.end())
    {
      ret.push_back(coverage_pair(fid.first, 0.0f));
    }
    else
    {
      float coverage = float(fm_it->second.sum()) / float(mask_w*mask_h);
      ret.push_back(coverage_pair(fid.first, coverage));
    }
  }
  return ret;
}

/// remove landmarks with IDs in the set
/**
* set the landmarks with the associated track ids to null
* \param [in] to_remove track ids for landmarks to set null
* \param [in,out] lms landmark map to remove landmarks from
*/

void
remove_landmarks(const std::set<track_id_t>& to_remove,
  landmark_map::map_landmark_t& lms)
{
  for (const track_id_t& tid : to_remove)
  {
    auto lm_it = lms.find(tid);
    if (lm_it != lms.end())
    {
      lms.erase(lm_it);
    }
  }
}

typedef std::vector<std::set<frame_id_t>> camera_components;

/// find connected components of cameras
/**
* Find connected components in the view graph.  Cameras that vew the same
* landmark are connected in the graph.
* \param [in] logger the logger to write debug info to
* \param [in] cams the cameras in the view graph.
* \param [in] lms the landmarks in the view graph.
* \param [in] tracks the track set
* \return camera_components vector.  Each set in the vector represents a
* different connected component.
*/

camera_components
connected_camera_components(
  kwiver::vital::logger_handle_t logger,
  const camera_map::map_camera_t& cams,
  const landmark_map::map_landmark_t& lms,
  track_set_sptr tracks)
{

  typedef std::map<landmark_id_t, landmark_sptr>::const_iterator lm_map_itr_t;
  typedef std::map<frame_id_t, camera_sptr>::const_iterator cam_map_itr_t;

  auto trks = tracks->tracks();

  camera_components comps;

  for (const track_sptr& t : trks)
  {
    lm_map_itr_t lmi = lms.find(t->id());
    if (lmi == lms.end() || !lmi->second)
    {
      // no landmark corresponding to this track
      continue;
    }

    //ok this track has an associated landmark
    const landmark& lm = *lmi->second;

    std::set<frame_id_t> cam_clique;
    for (track::history_const_itr tsi = t->begin(); tsi != t->end(); ++tsi)
    {
      auto fts = std::dynamic_pointer_cast<feature_track_state>(*tsi);
      if (!fts || !fts->feature)
      {
        // no feature for this track state.
        continue;
      }
      const feature& feat = *fts->feature;
      if (!fts->inlier)
      {
        //outliers don't connect cameras
        continue;
      }
      cam_map_itr_t ci = cams.find((*tsi)->frame());
      if (ci == cams.end() || !ci->second)
      {
        // no camera corresponding to this track state
        continue;
      }
      cam_clique.insert(ci->first);
    }
    if (cam_clique.empty())
    {
      continue; // nothing to do if no cameras found viewing this track.
    }

    //which of the existing cliques does cam_clique overlap with?
    std::vector<int> overlapping_comps;
    for (int comp_id = 0; comp_id < comps.size(); ++comp_id)
    {
      std::set<frame_id_t> &cur_comp = comps[comp_id];
      for (auto cn : cam_clique)
      {
        if (cur_comp.find(cn) != cur_comp.end())
        {
          overlapping_comps.push_back(comp_id);
          break; //go onto the next component
        }
      }
    }
    if (overlapping_comps.empty())
    {
      //make a new component
      comps.push_back(cam_clique);
    }
    else
    {
      std::set<frame_id_t> &final_comp = comps[overlapping_comps[0]];
      //add all cameras in cam_clique to final comp
      for (auto cn : cam_clique)
      {
        final_comp.insert(cn);
      }
      //merge all other overlapping components into final_comp
      for (int oc = 1; oc < overlapping_comps.size(); ++oc)
      {
        std::set<frame_id_t> &merged_comp = comps[overlapping_comps[oc]];
        final_comp.insert(merged_comp.begin(), merged_comp.end());
      }
      //remove all merged comps except for final_comp
      for (auto oc_it = overlapping_comps.rbegin();
        oc_it != overlapping_comps.rend(); ++oc_it)
      {
        int comp_idx = *oc_it;
        if (comp_idx == overlapping_comps[0])
        {
          continue;  // we don't erase the final comp which is the first
                     // overlapping comp
        }
        comps.erase(comps.begin() + comp_idx);
      }
    }
  }
  return comps;
}

/// clean a set of tracks.
/**
* Find connected components in the view graph.  Cameras that vew the same
* landmark are connected in the graph.
* \param [in] logger the logger to write debug info to
* \param [in] cams the cameras in the view graph.
* \param [in] lms the landmarks in the view graph.
* \param [in] tracks the track set
* \param [in] triang_cos_ang_thresh features must have one pair of rays that
*             meet this minimum intersection angle to keep
* \param [in] error_tol reprojection error threshold
* \return set of landmark ids (track_ids) that were bad and should be removed
*/

std::set<track_id_t>
clean_landmarks(
  kwiver::vital::logger_handle_t logger,
  const camera_map::map_camera_t& cams,
  const landmark_map::map_landmark_t& lms,
  track_set_sptr tracks,
  double triang_cos_ang_thresh,
  double error_tol = 5.0)
{
  //returns a set of un-constrained landmarks to be removed from the solution
  std::set<track_id_t> landmarks_to_remove;

  double ets = error_tol * error_tol;

  typedef std::map<landmark_id_t, landmark_sptr>::const_iterator lm_map_itr_t;
  typedef std::map<frame_id_t, camera_sptr>::const_iterator cam_map_itr_t;

  auto trks = tracks->tracks();

  size_t num_lm_removed_bad_angle = 0;
  size_t num_lm_found_from_tracks = 0;
  size_t num_unconstrained_landmarks_found = 0;

  for (const track_sptr& t : trks)
  {
    lm_map_itr_t lmi = lms.find(t->id());
    if (lmi == lms.end() || !lmi->second)
    {
      // no landmark corresponding to this track
      continue;
    }

    ++num_lm_found_from_tracks;

    //ok this track has an associated landmark
    const landmark& lm = *lmi->second;

    std::vector<kwiver::vital::simple_camera> observing_cams;
    for (track::history_const_itr tsi = t->begin(); tsi != t->end(); ++tsi)
    {
      auto fts = std::dynamic_pointer_cast<feature_track_state>(*tsi);
      if (!fts || !fts->feature)
      {
        // no feature for this track state.
        continue;
      }
      const feature& feat = *fts->feature;
      cam_map_itr_t ci = cams.find((*tsi)->frame());
      if (ci == cams.end() || !ci->second)
      {
        // no camera corresponding to this track state
        continue;
      }
      const camera& cam = *ci->second;

      auto d = cam.depth(lm.loc());
      if (d <= 0)
      {
        fts->inlier = false;
        continue;
      }

      double sq_err = kwiver::arrows::reprojection_error_sqr(cam, lm, feat);
      if (sq_err <= ets || error_tol < 0)
      {
        observing_cams.push_back(cam);
        fts->inlier = true;
      }
      else
      {
        fts->inlier = false;
      }
    }

    bool bad_ang = false;
    if (observing_cams.size() < 2)
    {
      ++num_unconstrained_landmarks_found;
      landmarks_to_remove.insert(t->id());
    }
    else
    {
      double cos_ang =
        kwiver::arrows::bundle_angle_max(observing_cams, lm.loc());

      bad_ang = cos_ang > triang_cos_ang_thresh;
      if (bad_ang)
      {
        ++num_lm_removed_bad_angle;
        landmarks_to_remove.insert(t->id());
      }
    }
  }
  if (logger)
  {
    LOG_DEBUG(logger, "num landmarks " << lms.size() << " num unconstrained " <<
      num_unconstrained_landmarks_found << " found from tracks " <<
      num_lm_found_from_tracks << " removed bad angle " <<
      num_lm_removed_bad_angle);
  }

  return landmarks_to_remove;
}

/// clean a set of cameras
/**
* Find cameras that don't meet the minimum required coverage and return them
* \param [in] logger the logger to write debug info to
* \param [in] cams the cameras in the view graph.
* \param [in] lms the landmarks in the view graph.
* \param [in] tracks the track set
* \param [in] coverage_thresh images that have less than this  fraction [0 - 1]
*             of coverage are included in the return set
* \return set of frames that do not meet the coverage requirement
*/

std::set<frame_id_t>
clean_cameras(
  kwiver::vital::logger_handle_t logger,
  const camera_map::map_camera_t& cams,
  landmark_map::map_landmark_t& lms,
  track_set_sptr tracks,
  float coverage_thresh)
{
  std::set<frame_id_t> rem_frames;

  frame_coverage_vec fc = image_coverages(tracks, lms, cams);


  for (auto cov : fc)
  {
    if (cov.second < coverage_thresh)
    {
      rem_frames.insert(cov.first);
    }
  }
  return rem_frames;
}



bool
clean_cameras_and_landmarks(
  camera_map::map_camera_t& cams,
  landmark_map::map_landmark_t& lms,
  track_set_sptr tracks,
  double triang_cos_ang_thresh,
  std::vector<frame_id_t> &removed_cams,
  kwiver::vital::logger_handle_t logger,
  float image_coverage_threshold,
  double error_tol)
{
  removed_cams.clear();
  //loop until no changes are done to further clean up the solution
  bool keep_cleaning = true;
  while (keep_cleaning)
  {
    keep_cleaning = false;
    std::set<track_id_t> lm_to_remove =
      clean_landmarks(logger, cams, lms, tracks, triang_cos_ang_thresh, error_tol);

    if (!lm_to_remove.empty())
    {
      keep_cleaning = true;
      if (logger)
      {
        LOG_DEBUG(logger, "removing " << lm_to_remove.size() <<
          " under constrained landmarks");
      }
    }
    remove_landmarks(lm_to_remove, lms);

    std::set<frame_id_t> cams_to_remove =
      clean_cameras(logger, cams, lms, tracks, image_coverage_threshold);

    for (auto frame_id : cams_to_remove)
    {
      cams[frame_id] = nullptr;
      removed_cams.push_back(frame_id);
      if (logger)
      {
        LOG_DEBUG(logger, "removing camera " << frame_id);
      }
    }

    camera_components comps =
      connected_camera_components(logger, cams, lms, tracks);

    if (comps.size() < 2)
    {
      //only one component so no need to remove disconnected cameras
      continue;
    }
    if (logger)
    {
      LOG_DEBUG(logger, "found " << comps.size() << " components");
    }
    int max_comp_idx = -1;
    size_t max_comp_size = 0;
    for (int ci = 0; ci < comps.size(); ++ci)
    {
      std::set<frame_id_t> &comp = comps[ci];
      if (logger)
      {
        LOG_DEBUG(logger, " comp size " << comp.size());
      }
      if (comp.size() > max_comp_size)
      {
        max_comp_size = comp.size();
        max_comp_idx = ci;
      }
    }
    //ok we have the largest component
    std::set<frame_id_t> &max_comp = comps[max_comp_idx];
    cams_to_remove.clear();
    for (auto cam : cams)
    {
      if (cam.second && max_comp.find(cam.first) == max_comp.end())
      {
        cams_to_remove.insert(cam.first);
      }
    }
    //remove cameras in disconnected components
    for (auto frame_id : cams_to_remove)
    {
      if (logger)
      {
        LOG_DEBUG(logger, "removing disconnected camera " << frame_id);
      }
      keep_cleaning = true;
      cams[frame_id] = nullptr;
      removed_cams.push_back(frame_id);
    }
    if (logger)
    {
      LOG_DEBUG(logger, "remaining cameras size " << cams.size());
    }
  }

  return true;
}


}
}