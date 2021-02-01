// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of utility functions for structure from motion.
 */

#include "sfm_utils.h"

#include <vital/vital_types.h>
#include <vital/types/bounding_box.h>
#include <vital/types/feature_track_set.h>
#include <arrows/mvg/metrics.h>
#include <arrows/core/match_matrix.h>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace mvg {

//-----------------------------------------------------------------------------
kwiver::vital::camera_perspective_sptr
crop_camera(const kwiver::vital::camera_perspective_sptr& cam,
            vital::bounding_box<int> crop)
{
  kwiver::vital::simple_camera_intrinsics new_intrinsics(*cam->intrinsics());
  kwiver::vital::vector_2d pp = new_intrinsics.principal_point();
  auto const tl = crop.upper_left();

  pp[0] -= tl.y(); // i
  pp[1] -= tl.x(); // j

  new_intrinsics.set_principal_point(pp);

  kwiver::vital::simple_camera_perspective crop_cam(cam->center(),
                                                   cam->rotation(),
                                                   new_intrinsics);

  return std::dynamic_pointer_cast<kwiver::vital::camera_perspective>(
    crop_cam.clone());
}

/// Detect tracks which remain stationary in the image
std::set<vital::track_sptr>
detect_stationary_tracks(vital::feature_track_set_sptr tracks,
                         double threshold)
{
  std::set<vital::track_sptr> stationary_tracks;
  auto tids = tracks->all_track_ids();
  for (auto tid : tids)
  {
    // Compute the mean image location of this track
    auto tk_sptr = tracks->get_track(tid);
    vital::vector_2d mean_loc(0.0, 0.0);
    double nfts = 0;
    for (auto fts : *tk_sptr | as_feature_track)
    {
      auto loc = fts->feature->loc();
      mean_loc *= (nfts / (nfts + 1.0));
      mean_loc += loc * (1.0 / (nfts + 1.0));
      nfts += 1.0;
    }
    if (nfts <= 0)
    {
      continue;
    }

    // If at least one point is far from the mean then this
    // this track is not considered stationary
    bool stationary_track = true;
    for (auto fts : *tk_sptr | as_feature_track)
    {
      auto loc = fts->feature->loc();
      auto diff = loc - mean_loc;
      if (diff.norm() > threshold)
      {
        stationary_track = false;
        break;
      }
    }

    if (stationary_track)
    {
      stationary_tracks.insert(tk_sptr);
    }
  }
  return stationary_tracks;
}

// anonymous namespace for local functions
namespace
{

// Helper function to subdivide the keyframe list as needed for better
// connectivity of tracks
void subdivide_keyframes(std::set<vital::frame_id_t>& key_idx,
                         Eigen::SparseMatrix<unsigned int> const& mm,
                         double ratio_threshold = 0.75)
{
  if (key_idx.empty())
  {
    return;
  }
  // Check each pair of adjacent keyframes to ensure that they
  // are reasonably well connected.  If not, recursively
  // subdivide adding more frames until adjacent frames are well
  // connected.
  std::set<vital::frame_id_t> new_key_idx;
  auto key2 = key_idx.begin();
  for (auto key1 = key2++; key2 != key_idx.end(); ++key1, ++key2)
  {
    // key1 and key2 are always sequential keyframes in the original sequence
    new_key_idx.insert(*key1);
    // k1 and k2 are candidates for sequential keyframe in the new sequence
    auto k1 = *key1;
    auto k2 = *key2;
    // If k1 == k2 then we have subdivided down to the sub-frame level
    // and we need to stop and move to the next interval.
    while (k1 < k2)
    {
      while (k1 < k2)
      {
        // As a baseline, consider the average number of tracks between
        // the nearest neighbors of each of k1 and k2 in this interval.
        // This is typically the best case, adjacent frames are most similar.
        double avg_tracks = (mm.coeff(k1, k1 + 1) + mm.coeff(k2 - 1, k2)) / 2;
        // We want the number of tracks over this interval to be within
        // a high percentage of the frame-to-frame track count
        auto num_tracks = mm.coeff(k1, k2);
        double ratio = num_tracks / avg_tracks;
        if (ratio > ratio_threshold)
        {
          // This interval is good, so move on to the next
          break;
        }
        // This interval is weak, so pick a new k2 half way between and retry
        k2 = (k1 + k2) / 2;
      }
      if (k1 < k2)
      {
        // We found an acceptable k2, so add it
        new_key_idx.insert(k2);
        // Move on to the other half of the interval
        k1 = k2;
        k2 = *key2;
      }
    }
  }
  key_idx.swap(new_key_idx);
}

}

/// Select keyframes that are a good starting point for SfM
std::set<vital::frame_id_t>
keyframes_for_sfm(feature_track_set_sptr tracks,
                  const frame_id_t radius,
                  const double ratio_threshold)
{
  std::set<vital::frame_id_t> keyframes;
  // Compute the match matrix for all frames
  auto all_frames_set = tracks->all_frame_ids();
  std::vector<frame_id_t> all_frames(all_frames_set.begin(),
                                     all_frames_set.end());
  auto mm = match_matrix(tracks, all_frames);

  // Compute the total track score on each frame.
  // The track score is the sum of the lengths of the tracks
  // passing through that frame.  This is equal to the sum
  // of the rows or columns of the match matrix.
  std::vector<std::pair<unsigned long, frame_id_t> > scores;
  scores.reserve(all_frames.size());
  for (frame_id_t k = 0; k < mm.outerSize(); ++k)
  {
    unsigned long score = 0;
    for (decltype(mm)::InnerIterator it(mm, k); it; ++it)
    {
      score += it.value();
    }
    scores.push_back(std::make_pair(score, k));
  }
  // Sort the frames in order of decreasing score
  typedef decltype(scores)::value_type s_t;
  std::sort(scores.begin(), scores.end(),
    [](const s_t &l, const s_t &r)
  {
    return l.first > r.first;
  });

  // Consider each score in order and use non-maximum suppression
  // to keep the strongest frames that are separated by at least
  // radius frames.
  std::vector<bool> mask(all_frames.size(), false);
  std::set<vital::frame_id_t> key_idx;
  for (auto const& s : scores)
  {
    auto const& idx = s.second;
    if (mask[idx])
    {
      // A nearby frame was already selected
      continue;
    }
    // Compute a range of indices within radius of the current
    // accounting for boundary conditions.
    const frame_id_t min_range = idx - std::min(idx, radius);
    const frame_id_t max_range =
      std::min(idx + radius, static_cast<frame_id_t>(mask.size() - 1));
    // Mark each frame within radius as covered
    for (frame_id_t i = min_range; i <= max_range; ++i)
    {
      mask[i] = true;
    }
    key_idx.insert(idx);
  }

  if (ratio_threshold > 0.0)
  {
    subdivide_keyframes(key_idx, mm, ratio_threshold);
  }

  // Look up the actual frame numbers for each selected index
  for (auto k : key_idx)
  {
    keyframes.insert(all_frames[k]);
  }

  return keyframes;
}

/// Calculate fraction of each image that is covered by landmark projections
frame_coverage_vec
image_coverages(
  std::vector<track_sptr> const& trks,
  vital::landmark_map::map_landmark_t const& lms,
  vital::simple_camera_perspective_map::frame_to_T_sptr_map const& cams )
{
  const int mask_w(16);
  const int mask_h(16);
  typedef Eigen::Matrix<int, mask_h, mask_w> vis_mask;
  typedef std::map<frame_id_t, vis_mask> frame_map_t;
  frame_map_t frame_masks;
  //calculate feature distribution masks for each candidate image.

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
      auto simp_cam = std::static_pointer_cast<simple_camera_perspective>(cam.second);
      im_dims dims;
      //this assumes the principal point is at the center of the image.
      dims.w = int(simp_cam->intrinsics()->principal_point().x() * 2.0);
      dims.h = int(simp_cam->intrinsics()->principal_point().y() * 2.0);
      frame_dims[cam.first] = dims;
    }
  }

  for (const track_sptr& t : trks)
  {
    if (lms.find(t->id()) != lms.end())
    {
      for (auto ts : *t)
      {
        auto fid = ts->frame();
        auto fd_it = frame_dims.find(fid);
        if (fd_it == frame_dims.end())
        {
          continue;
        }

        feature_track_state_sptr fts =
          std::dynamic_pointer_cast<feature_track_state>(ts);
        if (!fts || !fts->feature)
        {
          continue;
        }

        if (!fts->inlier)
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

/// Remove landmarks with IDs in the set
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

/// Find connected components of cameras
camera_components
connected_camera_components(
  vital::simple_camera_perspective_map::frame_to_T_sptr_map const& cams,
  landmark_map::map_landmark_t const& lms,
  feature_track_set_sptr tracks)
{
  auto trks = tracks->tracks();

  camera_components comps;

  for (const track_sptr& t : trks)
  {
    auto lmi = lms.find(t->id());
    if (lmi == lms.end() || !lmi->second)
    {
      // no landmark corresponding to this track
      continue;
    }

    std::unordered_set<frame_id_t> cam_clique;
    for (auto ts : *t)
    {
      auto fts = std::static_pointer_cast<feature_track_state>(ts);
      if (!fts || !fts->feature)
      {
        // no feature for this track state.
        continue;
      }
      if (!fts->inlier)
      {
        //outliers don't connect cameras
        continue;
      }
      auto ci = cams.find(ts->frame());
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
    for (unsigned int comp_id = 0; comp_id < comps.size(); ++comp_id)
    {
      auto &cur_comp = comps[comp_id];
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
      auto &final_comp = comps[overlapping_comps[0]];
      //add all cameras in cam_clique to final comp
      for (auto cn : cam_clique)
      {
        final_comp.insert(cn);
      }
      //merge all other overlapping components into final_comp
      for (unsigned int oc = 1; oc < overlapping_comps.size(); ++oc)
      {
       auto &merged_comp = comps[overlapping_comps[oc]];
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

/// Detect critical tracks that connect disjoint components
std::vector<track_sptr>
detect_critical_tracks(camera_components const& cc,
                       feature_track_set_sptr tracks)
{
  std::vector<track_sptr> critical_tracks;
  // build a mapping from frame number to connected component index
  std::map<frame_id_t, unsigned int> cc_map;
  for (unsigned int i=0; i<cc.size(); ++i)
  {
    for (auto const& f : cc[i])
    {
      cc_map[f] = i;
    }
  }

  // find tracks which span more than one connected component
  for (auto const& t : tracks->tracks())
  {
    unsigned int first_idx = cc_map[t->first_frame()];
    for (auto const& ts : *t)
    {
      auto idx = cc_map[ts->frame()];
      if (idx != first_idx)
      {
        critical_tracks.push_back(t);
        break;
      }
    }
  }
  return critical_tracks;
}

/// Detect underconstrained landmarks.
std::set<landmark_id_t>
detect_bad_landmarks(
  vital::simple_camera_perspective_map::frame_to_T_sptr_map const& cams,
  landmark_map::map_landmark_t const& lms,
  feature_track_set_sptr tracks,
  double triang_cos_ang_thresh,
  double error_tol,
  int min_landmark_inliers,
  double median_distance_multiple)
{
  //returns a set of un-constrained landmarks to be removed from the solution
  kwiver::vital::logger_handle_t logger(kwiver::vital::get_logger("arrows.mvg.sfm_utils"));
  std::set<landmark_id_t> landmarks_to_remove;

  double ets = error_tol * error_tol;

  auto trks = tracks->tracks();

  size_t num_lm_removed_bad_angle = 0;
  size_t num_lm_found_from_tracks = 0;
  size_t num_unconstrained_landmarks_found = 0;

  size_t observing_cams_thresh = std::max(2, min_landmark_inliers);

  std::vector<double> depths;
  for (auto const &lm_it: lms)
  {
    if (landmarks_to_remove.find(lm_it.first) != landmarks_to_remove.end())
    {
      //already removed, no need to process further.
      continue;
    }

    //this landmark has not already been marked for removal

    const auto lm = lm_it.second;
    auto t_id = lm_it.first;

    std::vector<kwiver::vital::simple_camera_perspective> observing_cams;

    auto t = tracks->get_track(t_id);
    if (!t)
    {
      continue;
    }

    for (auto ts : *t)
    {
      auto fts = std::static_pointer_cast<feature_track_state>(ts);
      if (!fts || !fts->feature)
      {
        // no feature for this track state.
        continue;
      }

      const feature& feat = *fts->feature;
      auto ci = cams.find(ts->frame());
      if (ci == cams.end() || !ci->second)
      {
        // no camera corresponding to this track state
        continue;
      }

      fts->inlier = false;

      const auto cam = ci->second;

      auto d = cam->depth(lm->loc());
      if (d <= 0)
      {
        continue;
      }

      double sq_err = reprojection_error_sqr(*cam, *lm, feat);
      if (sq_err <= ets || error_tol < 0)
      {
        observing_cams.push_back(*cam);
        fts->inlier = true;
        depths.push_back(d);
      }
    }

    if (observing_cams.size() < observing_cams_thresh)
    {
      ++num_unconstrained_landmarks_found;
      landmarks_to_remove.insert(lm_it.first);
    }
    else
    {
      if (!bundle_angle_is_at_least(observing_cams,lm->loc(),triang_cos_ang_thresh))
      {
        ++num_lm_removed_bad_angle;
        landmarks_to_remove.insert(lm_it.first);
      }
    }
  }

  size_t num_far_lm_removed = 0;

  if (median_distance_multiple > 0)
  {
    int med_loc = static_cast<int>(depths.size()*0.5);
    std::nth_element(depths.begin(), depths.begin() + med_loc, depths.end());
    double depth_thresh = -1;
    if (!depths.empty())
    {
      depth_thresh = depths[med_loc] * median_distance_multiple;
      for (auto const &lm_it : lms)
      {
        if (landmarks_to_remove.find(lm_it.first) != landmarks_to_remove.end())
        {
          //already removed, no need to process further.
          continue;
        }

        const auto lm = lm_it.second;
        auto t_id = lm_it.first;

        bool lm_too_far = false;

        auto t = tracks->get_track(t_id);
        if (!t)
        {
          continue;
        }
        for (auto ts : *t)
        {
          auto fts = std::static_pointer_cast<feature_track_state>(ts);
          if (!fts || !fts->feature)
          {
            // no feature for this track state.
            continue;
          }
          if (!fts->inlier)
          {
            continue;
          }

          auto ci = cams.find(ts->frame());
          if (ci == cams.end() || !ci->second)
          {
            // no camera corresponding to this track state
            continue;
          }
          const auto cam = std::static_pointer_cast<simple_camera_perspective>(ci->second);
          auto d = cam->depth(lm->loc());
          if (d > depth_thresh)
          {
            landmarks_to_remove.insert(lm_it.first);
            ++num_far_lm_removed;
            lm_too_far = true;
            break;
          }
        }
        if (lm_too_far)
        {
          break;
        }
      }
    }
  }

  for (auto &lm_id : landmarks_to_remove)
  {
    //mark all removed landmark feature track states as outliers
    auto t = tracks->get_track(lm_id);
    if (t)
    {
      for (auto ts : *t)
      {
        auto fts = std::static_pointer_cast<feature_track_state>(ts);
        if (fts && fts->feature)
        {
          fts->inlier = false;
        }
      }
    }
  }

  LOG_DEBUG(logger, "num landmarks " << lms.size() << " num unconstrained " <<
    num_unconstrained_landmarks_found << " found from tracks " <<
    num_lm_found_from_tracks << " removed bad angle " <<
    num_lm_removed_bad_angle << " removed too far " <<
    num_far_lm_removed);

  return landmarks_to_remove;
}

/// Detect bad cameras in sfm solution
std::set<frame_id_t>
detect_bad_cameras(
  vital::simple_camera_perspective_map::frame_to_T_sptr_map const& cams,
  landmark_map::map_landmark_t const& lms,
  feature_track_set_sptr tracks,
  float coverage_thresh)
{
  std::set<frame_id_t> rem_frames;

  frame_coverage_vec fc = image_coverages(tracks->tracks(), lms, cams);

  for (auto cov : fc)
  {
    if (cov.second < coverage_thresh)
    {
      rem_frames.insert(cov.first);
    }
  }
  return rem_frames;
}

/// Clean structure from motion solution
void
clean_cameras_and_landmarks(
  vital::simple_camera_perspective_map& cams_persp,
  landmark_map::map_landmark_t& lms,
  feature_track_set_sptr tracks,
  double triang_cos_ang_thresh,
  std::vector<frame_id_t> &removed_cams,
  const std::set<vital::frame_id_t> &active_cams,
  const std::set<vital::landmark_id_t> &active_lms,
  double image_coverage_threshold,
  double error_tol,
  int min_landmark_inliers)
{

  auto cams = cams_persp.T_cameras();

  landmark_map::map_landmark_t det_lms;
  if (active_lms.empty())
  {
    det_lms = lms;
  }
  else
  {
    for (auto lm_id : active_lms)
    {
      auto it = lms.find(lm_id);
      if (it == lms.end())
      {
        continue;
      }
      det_lms[lm_id] = it->second;
    }
  }

  simple_camera_perspective_map::frame_to_T_sptr_map det_cams;
  if (active_cams.empty())
  {
    det_cams = cams;
  }
  else
  {
    for (auto cam_id : active_cams)
    {
      auto it = cams.find(cam_id);
      if (it == cams.end())
      {
        continue;
      }
      det_cams[cam_id] = it->second;
    }
  }

  kwiver::vital::logger_handle_t logger(kwiver::vital::get_logger("arrows.mvg.sfm_utils"));

  removed_cams.clear();
  //loop until no changes are done to further clean up the solution
  bool keep_cleaning = true;
  while (keep_cleaning)
  {
    keep_cleaning = false;
    std::set<track_id_t> lm_to_remove =
      detect_bad_landmarks(cams, det_lms, tracks, triang_cos_ang_thresh, error_tol, min_landmark_inliers);

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
    remove_landmarks(lm_to_remove, det_lms);

    std::set<frame_id_t> cams_to_remove =
      detect_bad_cameras(det_cams, det_lms, tracks,
                         static_cast<float>(image_coverage_threshold));

    for (auto frame_id : cams_to_remove)
    {
      //set all features on removed camera to outliers
      for (auto ts : tracks->frame_states(frame_id))
      {
        auto fts = std::dynamic_pointer_cast<feature_track_state>(ts);
        if (!fts)
        {
          continue;
        }
        fts->inlier = false;
      }

      cams_persp.erase(frame_id);
      det_cams[frame_id] = nullptr;
      removed_cams.push_back(frame_id);
      LOG_DEBUG(logger, "removing camera " << frame_id);
    }
  }
}

/// Return true if the camera is upright
bool
camera_upright(vital::camera_perspective const& camera,
               vital::vector_3d const& up)
{
  return up.dot(camera.rotation().inverse() * vector_3d(0, -1, 0)) > 0;
}

/// Return true if most cameras are upright
bool
majority_upright(
  vital::camera_perspective_map::frame_to_T_sptr_map const& cameras,
  vital::vector_3d const& up)
{
  int net_cams_pointing_up = 0;
  for (auto const& cam : cameras)
  {
    if (camera_upright(*cam.second, up))
    {
      ++net_cams_pointing_up;
    }
    else
    {
      --net_cams_pointing_up;
    }
  }
  return net_cams_pointing_up > 0;
}

/// Return a subset of cameras on the positive side of a plane
vital::camera_perspective_map::frame_to_T_sptr_map
cameras_above_plane(
  vital::camera_perspective_map::frame_to_T_sptr_map const& cameras,
  vital::vector_4d const& plane)
{
  vital::camera_perspective_map::frame_to_T_sptr_map out_cams;
  for (auto const& cam : cameras)
  {
    if (cam.second->center().dot(plane.head(3)) + plane[3] > 0.0)
    {
      out_cams.insert(cam);
    }
  }
  return out_cams;
}

/// Compute the ground center of a collection of landmarks
vital::vector_3d
landmarks_ground_center(vital::landmark_map const& landmarks,
                        double ground_frac)
{
  const auto num_landmarks = landmarks.size();
  if (num_landmarks == 0)
  {
    return vital::vector_3d(0.0, 0.0, 0.0);
  }
  std::vector<double> x, y, z;
  x.reserve(num_landmarks);
  y.reserve(num_landmarks);
  z.reserve(num_landmarks);
  for (auto lm : landmarks.landmarks())
  {
    auto v = lm.second->loc();
    x.push_back(v[0]);
    y.push_back(v[1]);
    z.push_back(v[2]);
  }
  // compute the median in x and y
  size_t mid = x.size() / 2;
  std::nth_element(x.begin(), x.begin() + mid, x.end());
  std::nth_element(y.begin(), y.begin() + mid, y.end());
  // compute ground fraction index
  size_t gidx = static_cast<size_t>(x.size() * ground_frac);
  std::nth_element(z.begin(), z.begin() + gidx, z.end());

  return vital::vector_3d(x[mid], y[mid], z[gidx]);
}

}
}
}
