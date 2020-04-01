/*ckwg +29
 * Copyright 2018-2020 by Kitware, Inc.
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
 * \brief Implementation of keyframe camera and landmark initialization algorithm
 */

#include "initialize_cameras_landmarks_keyframe.h"

#include <random>
#include <unordered_map>
#include <deque>
#include <iterator>
#include <Eigen/StdVector>
#include <fstream>
#include <ctime>

#include <vital/math_constants.h>
#include <vital/exceptions.h>
#include <vital/io/eigen_io.h>

#include <vital/algo/estimate_essential_matrix.h>
#include <vital/algo/triangulate_landmarks.h>
#include <vital/algo/bundle_adjust.h>
#include <vital/algo/optimize_cameras.h>
#include <vital/algo/estimate_canonical_transform.h>
#include <vital/algo/estimate_similarity_transform.h>

#include <arrows/core/triangulate_landmarks.h>
#include <arrows/core/epipolar_geometry.h>
#include <arrows/core/metrics.h>
#include <arrows/core/match_matrix.h>
#include <arrows/core/necker_reverse.h>
#include <arrows/core/triangulate.h>
#include <arrows/core/transform.h>
#include <vital/algo/estimate_pnp.h>
#include <arrows/core/sfm_utils.h>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace core {

typedef std::map< frame_id_t, simple_camera_perspective_sptr >               map_cam_t;
typedef std::map<frame_id_t, simple_camera_perspective_sptr>::iterator       cam_map_itr_t;
typedef std::map<frame_id_t, simple_camera_perspective_sptr>::const_iterator const_cam_map_itr_t;

typedef vital::landmark_map::map_landmark_t map_landmark_t;

typedef std::pair<frame_id_t, float> coverage_pair;
typedef std::vector<coverage_pair> frame_coverage_vec;

class rel_pose {
public:
  vital::frame_id_t f0;
  vital::frame_id_t f1;
  matrix_3x3d R01;
  vector_3d t01;
  int well_conditioned_landmark_count;
  double angle_sum;
  float coverage_0;
  float coverage_1;
  static const double target_angle;

  double score(int min_matches = 30) const
  {
    if (well_conditioned_landmark_count == 0)
    {
      return 0;
    }

    double ave_angle = angle_sum / (double(well_conditioned_landmark_count));
    double angle_score = std::max(1.0 - fabs((ave_angle / target_angle) - 1.0), 0.0);

    // having more than 50 features really doesn't improve things
    double count_score = std::min(1.0, double(well_conditioned_landmark_count / 50.0));
    if (well_conditioned_landmark_count < min_matches)
    {
      count_score = 0;
    }

    // more than 20% overlap doesn't help
    double coverage_score = std::min(0.2f, std::min(coverage_0, coverage_1));
    return count_score * angle_score * angle_score * coverage_score;
  }

  bool operator<(const rel_pose & other) const
  {
    if (f0 < other.f0)
    {
      return true;
    }
    else if (f0 == other.f0)
    {
      return f1 < other.f1;
    }
    else
    {
      return false;
    }
  }

  friend std::ostream& operator<<(std::ostream& s, rel_pose const& rp);
  friend std::istream& operator>>(std::istream& s, rel_pose & rp);
};

const double rel_pose::target_angle = 20.0 * deg_to_rad;

/// output stream operator for a landmark base class
std::ostream&
operator<<(std::ostream& s, rel_pose const& rp)
{
  s << rp.f0 << " "
    << rp.f1 << "\n"
    << rp.R01 << "\n"
    << rp.t01 << "\n"
    << rp.well_conditioned_landmark_count << " "
    << rp.angle_sum << " "
    << rp.coverage_0 << " "
    << rp.coverage_1;
  return s;
}

/// input stream operator for a landmark
std::istream&
operator >> (std::istream& s, rel_pose & rp)
{
  s >> rp.f0
    >> rp.f1
    >> rp.R01
    >> rp.t01
    >> rp.well_conditioned_landmark_count
    >> rp.angle_sum
    >> rp.coverage_0
    >> rp.coverage_1;
  return s;
}

/// Private implementation class
class initialize_cameras_landmarks_keyframe::priv
{
public:
  /// Constructor
  priv();

  /// Destructor
  ~priv();

  /// Pass through this callback to another callback but cache the return value
  bool pass_through_callback(callback_t cb,
                             camera_map_sptr cams,
                             landmark_map_sptr lms,
                             feature_track_set_changes_sptr track_changes);

  void check_inputs(feature_track_set_sptr tracks);

  bool initialize_keyframes(
    simple_camera_perspective_map_sptr cameras,
    landmark_map_sptr& landmarks,
    feature_track_set_sptr tracks,
    sfm_constraints_sptr constraints,
    callback_t m_callback);

  bool vision_centric_keyframe_initialization(
    simple_camera_perspective_map_sptr cams,
    landmark_map_sptr& landmarks,
    feature_track_set_sptr tracks,
    sfm_constraints_sptr constraints,
    std::set<vital::frame_id_t> &keyframes,
    callback_t callback);

  bool metadata_centric_keyframe_initialization(
    simple_camera_perspective_map_sptr cams,
    bool use_existing_cams,
    landmark_map_sptr& landmarks,
    feature_track_set_sptr tracks,
    sfm_constraints_sptr constraints,
    std::set<vital::frame_id_t> &keyframes,
    callback_t callback);

  bool initialize_remaining_cameras(
    simple_camera_perspective_map_sptr cams,
    landmark_map_sptr& landmarks,
    feature_track_set_sptr tracks,
    sfm_constraints_sptr constraints,
    callback_t callback);

  bool bundle_adjust();

  rel_pose
  calc_rel_pose(frame_id_t frame_0, frame_id_t frame_1,
    const std::vector<track_sptr>& trks) const;

  bool write_rel_poses(std::string file_path) const;

  bool read_rel_poses(std::string file_path);

  std::set<frame_id_t>
  select_begining_frames_for_initialization(
    feature_track_set_sptr tracks) const;

  void calc_rel_poses(
      const std::set<frame_id_t> &frames,
      feature_track_set_sptr tracks);

  /// Re-triangulate all landmarks for provided tracks
  void retriangulate(landmark_map::map_landmark_t& lms,
    simple_camera_perspective_map_sptr cams,
    const std::vector<track_sptr>& trks,
    std::set<landmark_id_t>& inlier_lm_ids,
    unsigned int min_inlier_observations = 2,
    double inlier_threshold = 0.0) const;

  void triangulate_landmarks_visible_in_frames(
    landmark_map::map_landmark_t& lmks,
    simple_camera_perspective_map_sptr cams,
    feature_track_set_sptr tracks,
    std::set<frame_id_t> frame_ids,
    bool triangulate_only_outliers);

  void down_select_landmarks(
    landmark_map::map_landmark_t &lmks,
    simple_camera_perspective_map_sptr cams,
    feature_track_set_sptr tracks,
    std::set<frame_id_t> down_select_these_frames) const;

  bool fit_reconstruction_to_constraints(
    simple_camera_perspective_map_sptr cams,
    map_landmark_t &lms,
    feature_track_set_sptr tracks,
    sfm_constraints_sptr constraints,
    int &num_constraints_used);

  bool initialize_reconstruction(
    simple_camera_perspective_map_sptr cams,
    map_landmark_t &lms,
    feature_track_set_sptr tracks);

  frame_id_t select_next_camera(
    std::set<frame_id_t> &frames_to_resection,
    simple_camera_perspective_map_sptr cams,
    map_landmark_t lms,
    feature_track_set_sptr tracks);

  bool resection_camera(
    simple_camera_perspective_map_sptr cams,
    map_landmark_t lms,
    feature_track_set_sptr tracks,
    frame_id_t fid_to_resection);

  void three_point_pose(frame_id_t frame,
    vital::simple_camera_perspective_sptr &cam,
    feature_track_set_sptr tracks,
    vital::landmark_map::map_landmark_t lms,
    float coverage_threshold,
    vital::algo::estimate_pnp_sptr pnp);

  float image_coverage(
    simple_camera_perspective_map_sptr cams,
    const std::vector<track_sptr>& trks,
    const kwiver::vital::landmark_map::map_landmark_t& lms,
    frame_id_t frame) const;

  void remove_redundant_keyframe(
    simple_camera_perspective_map_sptr cameras,
    landmark_map_sptr& landmarks,
    feature_track_set_sptr tracks,
    sfm_constraints_sptr constraints,
    frame_id_t target_frame);

  void remove_redundant_keyframes(
    simple_camera_perspective_map_sptr cameras,
    landmark_map_sptr& landmarks,
    feature_track_set_sptr tracks,
    sfm_constraints_sptr constraints,
    std::deque<frame_id_t> &recently_added_frame_queue);

  int get_inlier_count(
    frame_id_t fid,
    landmark_map_sptr landmarks,
    feature_track_set_sptr tracks);

  int set_inlier_flags(
    frame_id_t fid,
    simple_camera_perspective_sptr cam,
    const map_landmark_t &lms,
    feature_track_set_sptr tracks,
    double reporj_thresh);

  void cleanup_necker_reversals(
    simple_camera_perspective_map_sptr cams,
    landmark_map_sptr landmarks,
    feature_track_set_sptr tracks,
    sfm_constraints_sptr constraints);

  std::set<landmark_id_t> find_visible_landmarks_in_frames(
    const map_landmark_t &lmks,
    feature_track_set_sptr tracks,
    const std::set<frame_id_t> &frames);

  vector_3d get_velocity(
    simple_camera_perspective_map_sptr cams,
    frame_id_t vel_frame) const;

  void get_registered_and_non_registered_frames(
    simple_camera_perspective_map_sptr cams,
    feature_track_set_sptr tracks,
    std::set<frame_id_t> &registered_frames,
    std::set<frame_id_t> &non_registered_frames) const;

  bool get_next_fid_to_register_and_its_closest_registered_cam(
    simple_camera_perspective_map_sptr cams,
    std::set<frame_id_t> &frames_to_register,
    frame_id_t &fid_to_register, frame_id_t &closest_frame) const;

  map_landmark_t get_sub_landmark_map(
    map_landmark_t &lmks,
    const std::set<landmark_id_t> &lm_ids) const;

  landmark_map_sptr store_landmarks(
    map_landmark_t &store_lms,
    map_landmark_t &to_store) const;

  bool initialize_next_camera(
    simple_camera_perspective_map_sptr cams,
    map_landmark_t& lmks,
    feature_track_set_sptr tracks,
    sfm_constraints_sptr constraints,
    frame_id_t &fid_to_register,
    std::set<frame_id_t> &frames_to_register,
    std::set<frame_id_t> &already_registred_cams);

  void windowed_clean_and_bundle(
    simple_camera_perspective_map_sptr cams,
    landmark_map_sptr& landmarks,
    map_landmark_t& lmks,
    feature_track_set_sptr tracks,
    sfm_constraints_sptr constraints,
    const std::set<frame_id_t> &already_bundled_cams,
    const std::set<frame_id_t> &frames_since_last_local_ba);

  feature_track_set_changes_sptr
    get_feature_track_changes(
      feature_track_set_sptr tracks,
      const simple_camera_perspective_map &cams) const;

  void init_base_camera_from_metadata(
    sfm_constraints_sptr constraints);

  void merge_landmarks(map_landmark_t &lmks,
    simple_camera_perspective_map_sptr const &cams,
    feature_track_set_sptr &tracks);

  bool verbose;
  bool continue_processing;
  double interim_reproj_thresh;
  double final_reproj_thresh;
  double image_coverage_threshold;
  double zoom_scale_thresh;
  vital::simple_camera_perspective m_base_camera;
  vital::algo::estimate_essential_matrix_sptr e_estimator;
  vital::algo::optimize_cameras_sptr camera_optimizer;
  vital::algo::triangulate_landmarks_sptr lm_triangulator;
  vital::algo::bundle_adjust_sptr bundle_adjuster;
  vital::algo::bundle_adjust_sptr global_bundle_adjuster;
  vital::algo::estimate_canonical_transform_sptr m_canonical_estimator;
  vital::algo::estimate_similarity_transform_sptr m_similarity_estimator;
  /// Logger handle
  vital::logger_handle_t m_logger;
  double m_thresh_triang_cos_ang;
  vital::algo::estimate_pnp_sptr m_pnp;
  std::set<rel_pose> m_rel_poses;
  std::set<frame_id_t> m_keyframes;
  Eigen::SparseMatrix<unsigned int> m_kf_match_matrix;
  std::vector<frame_id_t> m_kf_mm_frames;
  std::set<frame_id_t> m_frames_removed_from_sfm_solution;
  vital::track_map_t m_track_map;
  std::random_device m_rd;     // only used once to initialise (seed) engine
  std::mt19937 m_rng;    // random-number engine used (Mersenne-Twister in this case)
  double m_reverse_ba_error_ratio;
  bool m_solution_was_fit_to_constraints;
  int m_max_cams_in_keyframe_init;
  double m_frac_frames_for_init;
  double m_metadata_init_permissive_triang_thresh;
  bool m_init_intrinsics_from_metadata;
  bool m_config_defines_base_intrinsics;
  bool m_do_final_sfm_cleaning;
  bool m_force_common_intrinsics;
  std::set<landmark_id_t> m_already_merged_landmarks;
};

initialize_cameras_landmarks_keyframe::priv
::priv()
  : verbose(false),
  continue_processing(true),
  interim_reproj_thresh(10.0),
  final_reproj_thresh(2.0),
  image_coverage_threshold(0.05),
  zoom_scale_thresh(0.1),
  m_base_camera(),
  e_estimator(),
  camera_optimizer(),
  // use the core triangulation as the default, users can change it
  lm_triangulator(new core::triangulate_landmarks()),
  bundle_adjuster(),
  global_bundle_adjuster(),
  m_logger(vital::get_logger("arrows.core.initialize_cameras_landmarks_keyframe")),
  m_thresh_triang_cos_ang(cos(deg_to_rad * 2.0)),
  m_rng(m_rd()),
  m_reverse_ba_error_ratio(0.0),
  m_solution_was_fit_to_constraints(false),
  m_max_cams_in_keyframe_init(20),
  m_frac_frames_for_init(-1.0),
  m_metadata_init_permissive_triang_thresh(10000),
  m_init_intrinsics_from_metadata(true),
  m_config_defines_base_intrinsics(false),
  m_do_final_sfm_cleaning(false),
  m_force_common_intrinsics(true)
{

  }

initialize_cameras_landmarks_keyframe::priv
::~priv()
{
}

map_landmark_t
initialize_cameras_landmarks_keyframe::priv
::get_sub_landmark_map(map_landmark_t &lmks, const std::set<landmark_id_t> &lm_ids) const
{
  map_landmark_t sub_lmks;
  for (auto lm : lm_ids)
  {
    auto it = lmks.find(lm);
    if (it == lmks.end())
    {
      continue;
    }
    sub_lmks[lm] = it->second;
  }

  return sub_lmks;
}

landmark_map_sptr
initialize_cameras_landmarks_keyframe::priv
::store_landmarks(map_landmark_t &store_lms, map_landmark_t &to_store) const
{
  for (auto lm : to_store)
  {
    store_lms[lm.first] = lm.second;
  }
  return landmark_map_sptr(new simple_landmark_map(store_lms));
}

vector_3d
initialize_cameras_landmarks_keyframe::priv
::get_velocity(
  simple_camera_perspective_map_sptr cams,
  frame_id_t vel_frame) const
{
  vector_3d vel;
  vel.setZero();

  auto existing_cams = cams->T_cameras();
  frame_id_t closest_fid = -1;
  frame_id_t next_closest_fid = -1;
  auto it = existing_cams.find(vel_frame);
  simple_camera_perspective_sptr closest_cam, next_closest_cam;
  closest_cam = nullptr;
  next_closest_cam = nullptr;
  frame_id_t min_frame_diff = std::numeric_limits<frame_id_t>::max();

  if (it == existing_cams.end())
  {
    // find the closest existing cam to vel_frame
    for (auto &ec : existing_cams)
    {
      auto diff = abs(ec.first - vel_frame);
      if (diff < min_frame_diff)
      {
        min_frame_diff = diff;
        closest_cam = ec.second;
        closest_fid = ec.first;
      }
    }
  }
  else
  {
    closest_cam = it->second;
  }

  if (!closest_cam)
  {
    return vel;
  }

  // find the second closest cam
  min_frame_diff = std::numeric_limits<frame_id_t>::max();
  for (auto &ec : existing_cams)
  {
    if (ec.first == closest_fid)
    {
      continue;
    }
    auto diff = abs(ec.first - closest_fid);
    if (diff < min_frame_diff)
    {
      min_frame_diff = diff;
      next_closest_cam = ec.second;
      next_closest_fid = ec.first;
    }
  }

  if (!next_closest_cam || min_frame_diff > 4)
  {
    return vel;
  }

  double frame_diff = static_cast<double>(closest_fid - next_closest_fid);
  auto pos_diff = closest_cam->center() - next_closest_cam->center();

  vel = pos_diff * (1.0 / frame_diff);

  return vel;
}

void
initialize_cameras_landmarks_keyframe::priv
::check_inputs(feature_track_set_sptr tracks)
{
  if (!tracks)
  {
    VITAL_THROW( invalid_value, "required feature tracks are NULL.");
  }
  if (!e_estimator)
  {
    VITAL_THROW( invalid_value, "Essential matrix estimator not initialized.");
  }
  if (!lm_triangulator)
  {
    VITAL_THROW( invalid_value, "Landmark triangulator not initialized.");
  }
}


/// Pass through this callback to another callback but cache the return value
bool
initialize_cameras_landmarks_keyframe::priv
::pass_through_callback(callback_t cb,
  camera_map_sptr cams,
  landmark_map_sptr lms,
  feature_track_set_changes_sptr track_changes)
{
  this->continue_processing = cb(cams, lms, track_changes);
  return this->continue_processing;
}


/// Re-triangulate all landmarks for provided tracks
void
initialize_cameras_landmarks_keyframe::priv
::retriangulate(landmark_map::map_landmark_t& lms,
  simple_camera_perspective_map_sptr cams,
  const std::vector<track_sptr>& trks,
  std::set<landmark_id_t>& inlier_lm_ids,
  unsigned int min_inlier_observations,
  double inlier_threshold) const
{
  typedef landmark_map::map_landmark_t lm_map_t;
  lm_map_t init_lms;

  for (const track_sptr& t : trks)
  {
    const track_id_t& tid = t->id();
    lm_map_t::const_iterator li = lms.find(tid);
    if (li == lms.end())
    {
      auto lm = std::make_shared<landmark_d>(vector_3d(0, 0, 0));
      init_lms[static_cast<landmark_id_t>(tid)] = lm;
    }
    else
    {
      init_lms.insert(*li);
    }
  }

  auto triang_config = lm_triangulator->get_configuration();
  double triang_thresh_orig = triang_config->get_value<double>("inlier_threshold_pixels", 2.0);
  if (inlier_threshold > 0.0)
  {
    triang_config->set_value<double>("inlier_threshold_pixels",
                                     inlier_threshold);
    lm_triangulator->set_configuration(triang_config);
  }

  landmark_map_sptr lm_map = std::make_shared<simple_landmark_map>(init_lms);
  auto tracks = std::make_shared<feature_track_set>(trks);
  this->lm_triangulator->triangulate(cams, tracks, lm_map);

  if (inlier_threshold > 0.0)
  {
    triang_config->set_value<double>("inlier_threshold_pixels",
                                     triang_thresh_orig);
    lm_triangulator->set_configuration(triang_config);
  }

  inlier_lm_ids.clear();
  lms.clear();
  auto inlier_lms = lm_map->landmarks();
  for(auto lm: inlier_lms)
  {
    if (lm.second->observations() < min_inlier_observations)
    {
      lms.erase(lm.first);
    }
    else
    {
      inlier_lm_ids.insert(lm.first);
      lms[lm.first] = lm.second;
    }
  }
}

void
initialize_cameras_landmarks_keyframe::priv
::triangulate_landmarks_visible_in_frames(
  landmark_map::map_landmark_t& lmks,
  simple_camera_perspective_map_sptr cams,
  feature_track_set_sptr tracks,
  std::set<frame_id_t> frame_ids,
  bool triangulate_only_outliers)
{
  // only triangulates tracks that don't already have associated landmarks
  landmark_map::map_landmark_t new_lms;
  std::vector<track_sptr> triang_tracks;

  for (auto fid : frame_ids)
  {
    auto active_tracks = tracks->active_tracks(fid);
    for (auto &t : active_tracks)
    {
      auto fts = std::static_pointer_cast<feature_track_state>(*(t->find(fid)));
      if (fts->inlier && triangulate_only_outliers)
      {
        continue;
      }

      auto it = lmks.find(t->id());
      if (it != lmks.end())
      {
        continue;
      }

      triang_tracks.push_back(t);
    }
  }
  std::set<landmark_id_t> inlier_lm_ids;
  retriangulate(new_lms, cams, triang_tracks, inlier_lm_ids,3);

  for (auto &lm : new_lms)
  {
    lmks[lm.first] = lm.second;
  }
}

class gridded_mask {
public:
  gridded_mask(
    int input_w,
    int input_h,
    int min_features_per_cell):
    m_input_w(input_w),
    m_input_h(input_h),
    m_min_features_per_cell(min_features_per_cell)
  {
    m_mask.setZero();
  }

  void add_entry(vector_2d loc)
  {
    int cx = static_cast<int>(m_mask.cols() * (loc.x() / double(m_input_w)));
    int cy = static_cast<int>(m_mask.rows() * (loc.y() / double(m_input_h)));
    cx = std::min<int>(cx, static_cast<int>(m_mask.cols()) - 1);
    cy = std::min<int>(cy, static_cast<int>(m_mask.rows()) - 1);
    auto &mv = m_mask(cy, cx);
    ++mv;
  }

  bool conditionally_remove_entry(vector_2d loc)
  {
    int cx = static_cast<int>(m_mask.cols() * (loc.x() / double(m_input_w)));
    int cy = static_cast<int>(m_mask.rows() * (loc.y() / double(m_input_h)));
    cx = std::min<int>(cx, static_cast<int>(m_mask.cols()) - 1);
    cy = std::min<int>(cy, static_cast<int>(m_mask.rows()) - 1);
    auto &mv = m_mask(cy, cx);
    if (mv > m_min_features_per_cell)
    {
      --mv;
      return true;
    }
    else
    {
      return false;
    }
  }

private:

  int m_input_w;
  int m_input_h;
  int m_min_features_per_cell;
  Eigen::Matrix<int, 4, 4> m_mask;

};

typedef std::shared_ptr<gridded_mask> gridded_mask_sptr;


void
initialize_cameras_landmarks_keyframe::priv
::down_select_landmarks(
  landmark_map::map_landmark_t &lmks,
  simple_camera_perspective_map_sptr cams,
  feature_track_set_sptr tracks,
  std::set<frame_id_t> down_select_these_frames) const
{
  // go through landmarks visible in the down select frames
  // favor longer landmarks
  // get at least N landmarks per image region, if possible
  const int cells_w = 4;
  const int cells_h = 4;
  const int min_features_per_cell = 128 / (cells_w*cells_h);
  std::map<frame_id_t, gridded_mask_sptr> masks;

  frame_id_t first_frame = std::numeric_limits<frame_id_t>::max();
  frame_id_t last_frame = -1;
  std::set<track_sptr> lm_to_downsample_set;
  for (auto ds : down_select_these_frames)
  {
    auto active_tracks = tracks->active_tracks(ds);
    for (auto t : active_tracks)
    {
      if (lmks.find(t->id()) == lmks.end())
      {
        continue;
      }
      first_frame = std::min<frame_id_t>(first_frame, t->first_frame());
      last_frame = std::max<frame_id_t>(last_frame, t->last_frame());
      lm_to_downsample_set.insert(t);
    }
  }

  // Ok now we know what frames the down select tracks cover.
  // Build a mask for each of these frames.
  for (auto cam : cams->T_cameras())
  {
    auto fid = cam.first;
    if (fid < first_frame || fid > last_frame)
    {
      continue;
    }
    int image_w = static_cast<int>(2 * cam.second->intrinsics()->principal_point().x());
    int image_h = static_cast<int>(2 * cam.second->intrinsics()->principal_point().y());

    auto mask = std::make_shared<gridded_mask>(image_w, image_h,
                                               min_features_per_cell);
    masks[fid] = mask;

    auto active_tracks = tracks->active_tracks(fid);
    for (auto t : active_tracks)
    {
      if (lmks.find(t->id()) == lmks.end())
      {
        continue;
      }
      auto ts = *t->find(fid);
      auto fts = std::static_pointer_cast<feature_track_state>(ts);
      if (!fts->inlier)
      {
        continue;
      }
      mask->add_entry(fts->feature->loc());
    }
  }

  std::vector<track_sptr> lm_to_downsample;
  for (auto t : lm_to_downsample_set)
  {
    lm_to_downsample.push_back(t);
  }

  // Will be used to obtain a seed for the random number engine
  std::random_device rd;
  // Standard mersenne_twister_engine seeded with rd()
  std::mt19937 gen(rd());
  const int good_enough_size = 10;
  std::uniform_int_distribution<> dis(2, good_enough_size);

  std::set<landmark_id_t> lm_to_remove;
  // First we sample from the long_ds_landmarks because they will make
  // the reconstruction complete without as many breaks
  while (!lm_to_downsample.empty())
  {
    auto &t1 = lm_to_downsample.back();
    for(int ns = 0; ns < 20; ++ns)
    {
      int rand_idx = std::min<int>(static_cast<int>(
                                     (double(std::rand()) / double(RAND_MAX)) *
                                     lm_to_downsample.size()),
                                   static_cast<int>(lm_to_downsample.size()) - 1);
      t1 = lm_to_downsample[rand_idx];
      int length_thresh = dis(gen);
      int t1_effective_len = std::min<int>(static_cast<int>(t1->size()),
                                           good_enough_size);

      if (t1_effective_len <= length_thresh)
      {
        break;
      }
    }
    auto &t2 = lm_to_downsample.back();
    std::swap(t1, t2);
    auto t = t2;
    lm_to_downsample.pop_back();
    bool keep_lm = false;
    for (auto &ts : *t)
    {
      auto fid = ts->frame();
      auto mask_it = masks.find(fid);
      if (mask_it == masks.end())
      {
        continue;
      }
      auto fts = std::static_pointer_cast<feature_track_state>(ts);
      if (!fts->inlier)
      {
        continue;
      }

      if (!mask_it->second->conditionally_remove_entry(fts->feature->loc()))
      {
        keep_lm = true;
      }
    }
    if (!keep_lm)
    {
      lm_to_remove.insert(t->id());
    }
  }

  for (auto lm_id : lm_to_remove)
  {
    lmks.erase(lm_id);
  }
}

rel_pose
initialize_cameras_landmarks_keyframe::priv
::calc_rel_pose(frame_id_t frame_0, frame_id_t frame_1,
  const std::vector<track_sptr>& trks) const
{
  // extract coresponding image points and landmarks
  std::vector<vector_2d> pts_right, pts_left;

  auto base_intrinsics = m_base_camera.get_intrinsics()->clone();

  const camera_intrinsics_sptr cal_left = base_intrinsics;
  const camera_intrinsics_sptr cal_right = base_intrinsics;

  auto cal_left_no_dist = std::static_pointer_cast<simple_camera_intrinsics>(
    cal_left->clone());
  cal_left_no_dist->set_dist_coeffs(Eigen::VectorXd());
  auto cal_right_no_dist = std::static_pointer_cast<simple_camera_intrinsics>(
    cal_right->clone());
  cal_right_no_dist->set_dist_coeffs(Eigen::VectorXd());

  for (unsigned int i = 0; i<trks.size(); ++i)
  {
    auto frame_data_0 = std::dynamic_pointer_cast<feature_track_state>(
      *(trks[i]->find(frame_0)));
    auto frame_data_1 = std::dynamic_pointer_cast<feature_track_state>(
      *(trks[i]->find(frame_1)));
    if (!frame_data_0 || !frame_data_1)
    {
      continue;
    }
    auto undist_f1 = cal_left->unmap(frame_data_1->feature->loc());
    auto undist_f0 = cal_right->unmap(frame_data_0->feature->loc());

    pts_left.push_back(cal_left_no_dist->map(undist_f1));
    pts_right.push_back(cal_right_no_dist->map(undist_f0));
  }

  std::vector<bool> inliers;
  essential_matrix_sptr E_sptr = e_estimator->estimate(pts_right, pts_left,
    cal_right, cal_left,
    inliers, interim_reproj_thresh);
  const essential_matrix_d E(*E_sptr);

  unsigned num_inliers = static_cast<unsigned>(std::count(inliers.begin(),
    inliers.end(), true));

  LOG_DEBUG(m_logger, "E matrix num inliers = " << num_inliers
    << "/" << inliers.size());

  // get the first inlier index
  unsigned int inlier_idx = 0;
  for (; inlier_idx < inliers.size() && !inliers[inlier_idx]; ++inlier_idx);

  // get the first inlier correspondence to
  // disambiguate essential matrix solutions
  vector_2d left_pt = cal_left->unmap(pts_left[inlier_idx]);
  vector_2d right_pt = cal_right->unmap(pts_right[inlier_idx]);

  // compute the corresponding camera rotation and translation (up to scale)
  vital::simple_camera_perspective cam =
    kwiver::arrows::extract_valid_left_camera(E, left_pt, right_pt);
  cam.set_intrinsics(base_intrinsics);

  map_landmark_t lms;

  simple_camera_perspective_map::frame_to_T_sptr_map cams;
  cams[frame_1] = std::make_shared<simple_camera_perspective>(cam);
  cams[frame_0] = std::make_shared<simple_camera_perspective>();
  if (m_force_common_intrinsics)
  {
    cams[frame_0]->set_intrinsics(cams[frame_1]->get_intrinsics());
  }
  else
  {
    cams[frame_0]->set_intrinsics(cam.get_intrinsics()->clone());
  }

  auto cam_map = std::make_shared <simple_camera_perspective_map>(cams);

  auto trk_set = std::make_shared<feature_track_set>(trks);

  std::set<frame_id_t> inlier_lm_ids;
  retriangulate(lms, cam_map, trks, inlier_lm_ids);
  size_t inlier_count_prev = 0;

  // optimizing loop
  while (inlier_lm_ids.size() > inlier_count_prev)
  {
    inlier_count_prev = inlier_lm_ids.size();
#pragma omp critical
    {
      std::set<frame_id_t> empty_fixed_cams;
      std::set<landmark_id_t> empty_fixed_lms;
      global_bundle_adjuster->optimize(*cam_map, lms, trk_set,
                                       empty_fixed_cams, empty_fixed_lms);
    }
    inlier_lm_ids.clear();
    retriangulate(lms, cam_map, trks, inlier_lm_ids);
  }

  rel_pose rp;
  rp.f0 = frame_0;
  rp.f1 = frame_1;
  rp.R01 = cam.get_rotation().matrix();
  rp.t01 = cam.translation();
  rp.well_conditioned_landmark_count = static_cast<int>(inlier_lm_ids.size());
  rp.angle_sum = 0;

  rp.coverage_0 = image_coverage(cam_map, trks, lms, frame_0);
  rp.coverage_1 = image_coverage(cam_map, trks, lms, frame_1);

  for (auto lm : lms)
  {
    double ang = acos(lm.second->cos_obs_angle());
    rp.angle_sum += ang;
  }
  return rp;
}

bool
initialize_cameras_landmarks_keyframe::priv
::write_rel_poses(std::string file_path) const
{
  std::ofstream pose_stream;
  pose_stream.open(file_path);
  if (!pose_stream.is_open())
  {
    return false;
  }

  bool first = true;
  for (auto const& rp : m_rel_poses)
  {
    if (!first)
    {
      pose_stream << "\n";
    }
    pose_stream << rp;
    first = false;
  }

  return true;
}

bool
initialize_cameras_landmarks_keyframe::priv
::read_rel_poses(std::string file_path)
{
  m_rel_poses.clear();
  std::ifstream pose_stream;
  pose_stream.open(file_path);
  if (!pose_stream.is_open())
  {
    return false;
  }

  rel_pose rp;
  while (!pose_stream.eof())
  {
    pose_stream >> rp;
    m_rel_poses.insert(rp);
  }

  return true;
}

void
initialize_cameras_landmarks_keyframe::priv
::calc_rel_poses(
  const std::set<frame_id_t> &frames,
  feature_track_set_sptr tracks)
{
  std::string rel_pose_path = "rel_poses.txt";

  m_rel_poses.clear();

  unsigned frames_skip = std::max(1u, static_cast<unsigned>(frames.size() / 2));

  do {

    std::vector<frame_id_t> m_kf_mm_frames;
    int fid_idx = 0;
    for(auto fid: frames)
    {
      if (fid_idx%frames_skip == 0)
      {
        m_kf_mm_frames.push_back(fid);
      }
      ++fid_idx;
    }

    m_kf_match_matrix = match_matrix(tracks, m_kf_mm_frames);

    const int cols = static_cast<int>(m_kf_match_matrix.cols());

    const int min_matches = 100;

    std::vector<std::pair<frame_id_t, frame_id_t>> pairs_to_process;
    for (int k = 0; k < cols; ++k)
    {
      for (Eigen::SparseMatrix<unsigned int>::InnerIterator
             it(m_kf_match_matrix, k); it; ++it)
      {
        if (it.row() > k && it.value() > min_matches)
        {
          auto fid_0 = m_kf_mm_frames[it.row()];
          auto fid_1 = m_kf_mm_frames[k];
          if (fid_0 > fid_1)
          {
            std::swap(fid_0, fid_1);
          }
          rel_pose rp;
          rp.f0 = fid_0;
          rp.f1 = fid_1;
          if (m_rel_poses.find(rp) != m_rel_poses.end())
          {
            //we've already computed this relative pose
            continue;
          }
          pairs_to_process.push_back(std::make_pair(fid_0, fid_1));
        }
      }
    }

#pragma omp parallel for schedule(dynamic, 10)
    for (int64_t i = 0; i < static_cast<int64_t>(pairs_to_process.size()); ++i)
    {
      const auto &tp = pairs_to_process[i];
      auto fid_0 = tp.first;
      auto fid_1 = tp.second;
      auto tks0 = tracks->active_tracks(fid_0);
      auto tks1 = tracks->active_tracks(fid_1);
      std::sort(tks0.begin(), tks0.end());
      std::sort(tks1.begin(), tks1.end());
      std::vector<kwiver::vital::track_sptr> tks_01;
      std::set_intersection(tks0.begin(), tks0.end(),
                            tks1.begin(), tks1.end(),
                            std::back_inserter(tks_01));

      // ok now we have the common tracks between the two frames.
      // make the essential matrix, decompose it and store it in a relative pose
      rel_pose rp = calc_rel_pose(fid_0, fid_1, tks_01);
#pragma omp critical
      {
        if (rp.well_conditioned_landmark_count > 100)
        {
          m_rel_poses.insert(rp);
        }
      }
    }

    if (m_rel_poses.size() > 20)
    {
      // we have enough initialization poses.
      break;
    }
    frames_skip /= 2;

  } while (frames_skip >= 1);

  write_rel_poses(rel_pose_path);
}

frame_id_t
initialize_cameras_landmarks_keyframe::priv
::select_next_camera(
  std::set<frame_id_t> &frames_to_resection,
  simple_camera_perspective_map_sptr cams,
  map_landmark_t lms,
  feature_track_set_sptr tracks)
{
  std::map<frame_id_t, double> resection_frames_score;

  for (const auto& rp : m_rel_poses)
  {
    frame_id_t crossing_resection_frame_id = -1;

    if (cams->find(rp.f0) &&
      frames_to_resection.find(rp.f1) != frames_to_resection.end())
    {
      crossing_resection_frame_id = rp.f1;
    }

    if (cams->find(rp.f1) &&
      frames_to_resection.find(rp.f0) != frames_to_resection.end())
    {
      crossing_resection_frame_id = rp.f0;
    }

    if (crossing_resection_frame_id == -1)
    {
      continue;
    }
    // rel pose spans the current cameras and the frames to resection
    auto rfs_it = resection_frames_score.find(crossing_resection_frame_id);
    double rp_score = rp.score(0);
    if(rfs_it != resection_frames_score.end())
    {

      if (rp_score > rfs_it->second)
      {
        rfs_it->second = rp_score;
      }
    }
    else
    {
      resection_frames_score[crossing_resection_frame_id] = rp_score;
    }
  }

  // find the maximum resection_frames_score and return that pose
  double max_score = -1;
  frame_id_t selected_frame = -1;
  for (auto rfs : resection_frames_score)
  {
    if (rfs.second > max_score)
    {
      selected_frame = rfs.first;
      max_score = rfs.second;
    }
  }


  if (selected_frame == -1)
  {
    int max_existing_spacing = -1;
    auto reconstructed_cam_ids = cams->get_frame_ids();
    frame_id_t cid_prev = -1;
    for (auto cid : reconstructed_cam_ids)
    {
      if (cid_prev >= 0)
      {
        max_existing_spacing = std::max<int>(max_existing_spacing,
                                             static_cast<int>(cid - cid_prev));
      }
      cid_prev = cid;
    }

    std::set<landmark_id_t> currently_reconstructed_landmarks;
    for (auto lm : lms)
    {
      currently_reconstructed_landmarks.insert(lm.first);
    }

    // We can't use the relative pose scores any more to do the selection.
    // Score each remaining camera accordint to how far it is temporally from
    // the reconstructed cameras and how many 3D landmarks it sees.
    long best_score = -1;

    for (auto rc : frames_to_resection)
    {
      auto closest_frame_diff = std::numeric_limits<frame_id_t>::max();
      for (auto c : reconstructed_cam_ids)
      {
        auto diff = abs(rc - c);
        if (diff < closest_frame_diff)
        {
          closest_frame_diff = diff;
        }
      }

      if (closest_frame_diff > 1.2*max_existing_spacing)
      {
        // reconstruction has not successfully included a camera spaced out
        // much more than max_existing_spacing so don't try this one.
        continue;
      }

      auto rc_tracks_ids = tracks->active_track_ids(rc);
      std::vector<landmark_id_t> intersect_lmks;
      std::set_intersection(currently_reconstructed_landmarks.begin(),
                            currently_reconstructed_landmarks.end(),
                            rc_tracks_ids.begin(), rc_tracks_ids.end(),
                            std::back_inserter(intersect_lmks));

      if (intersect_lmks.size() < 10)
      {
        // without enough landmarks in common
        // there is no reason to try this frame
        continue;
      }
      long score = static_cast<long>(intersect_lmks.size() * closest_frame_diff);
      if (score > best_score)
      {
        best_score = score;
        selected_frame = rc;
      }

    }

  }

  return selected_frame;
}

/// Calculate fraction of image that is covered by landmark projections
/**
* For the frame find landmarks that project into the frame.  Mark the
* associated feature projection locations as occupied in a mask.  After masks
* has been accumulated calculate the fraction of each mask that is occupied.
* Return this coverage fraction.
* \param [in] tracks the set of feature tracks
* \param [in] lms landmarks to check coverage on
* \param [in] frame the frame to have coverage calculated on
* \param [in] im_w width of images
* \param [in] in_h height of images
* \return     the coverage fraction range [0 - 1]
*/

float
initialize_cameras_landmarks_keyframe::priv
::image_coverage(
  simple_camera_perspective_map_sptr cams,
  const std::vector<track_sptr>& trks,
  const kwiver::vital::landmark_map::map_landmark_t& lms,
  frame_id_t frame) const
{
  simple_camera_perspective_map::frame_to_T_sptr_map cam_map;
  cam_map[frame] = cams->find(frame);
  std::set<frame_id_t> frames;
  frames.insert(frame);
  auto coverages = image_coverages(trks, lms, cam_map);
  return coverages[0].second;
}


void
initialize_cameras_landmarks_keyframe::priv
::three_point_pose(frame_id_t frame,
  vital::simple_camera_perspective_sptr &cam,
  feature_track_set_sptr tracks,
  vital::landmark_map::map_landmark_t lms,
  float coverage_threshold,
  vital::algo::estimate_pnp_sptr pnp)
{
  std::vector<vital::vector_2d> pts2d;
  std::vector<vital::vector_3d> pts3d;
  kwiver::vital::camera_intrinsics_sptr cal;
  std::vector<bool> inliers;
  typedef std::pair<landmark_id_t, landmark_sptr> lm_pair;
  std::vector<lm_pair> frame_landmarks;
  std::vector<feature_track_state_sptr> frame_feats;
  auto tks = tracks->active_tracks(frame);
  for (auto tk : tks)
  {
    vital::track_id_t tid = tk->id();
    auto ts = tk->find(frame);
    if (ts == tk->end())
    {
      // just a double check but this should not happen
      continue;
    }

    feature_track_state_sptr fts = std::dynamic_pointer_cast<feature_track_state>(*ts);
    if (!fts || !fts->feature)
    {
      // make sure it's a feature track state.  Always should be.
      continue;
    }

    auto lm_it = lms.find(tid);
    if (lm_it == lms.end())
    {
      // no landmark for this track
      continue;
    }

    // ok we have a landmark with an associated track.
    pts3d.push_back(lm_it->second->loc());
    pts2d.push_back(fts->feature->loc());
    frame_feats.push_back(fts);
    frame_landmarks.push_back(lm_pair(lm_it->first, lm_it->second));
  }

  if (pts2d.size() < 4)
  {
    cam = simple_camera_perspective_sptr();
    return;
  }

  cam = std::static_pointer_cast<simple_camera_perspective>(
    pnp->estimate(pts2d, pts3d, cam->intrinsics(), inliers));

  if (!cam)
  {
    LOG_DEBUG(m_logger, "resectioning image " << frame << " failed");
    return;
  }

  size_t num_inliers = 0;
  float coverage = 0;
  std::map<landmark_id_t, landmark_sptr> inlier_lms;

  for (size_t i = 0; i < inliers.size(); ++i)
  {
    if (inliers[i])
    {
      ++num_inliers;
      inlier_lms.insert(frame_landmarks[i]);
      frame_feats[i]->inlier = true;
    }
    else
    {
      frame_feats[i]->inlier = false;
    }
  }
  auto cams = std::make_shared<simple_camera_perspective_map>();
  cams->insert(frame, cam);
  coverage = image_coverage(cams, tks, inlier_lms, frame);


  LOG_DEBUG(m_logger, "for frame " << frame << " P3P found " << num_inliers <<
    " inliers out of " << inliers.size() <<
    " feature projections with coverage " << coverage);

  if (coverage < coverage_threshold)
  {
    LOG_DEBUG(m_logger, "resectioning image " << frame <<
      " failed: insufficient coverage ( " << coverage << " )");
    cam = simple_camera_perspective_sptr();
    return;
  }

  // If the camera is not upright (relative to the Z-axis) then
  // Necker reverse about the plane fitting the inlier points
  // Note: this assumes that the landmarks have already been oriented
  // such that +Z is up.
  if (!camera_upright(*cam))
  {
    const vector_4d plane = landmark_plane(inlier_lms);
    necker_reverse_inplace(*cam, plane);
    LOG_DEBUG(m_logger, "Necker reversing camera for frame " << frame);
  }
}

bool
initialize_cameras_landmarks_keyframe::priv
::resection_camera(
  simple_camera_perspective_map_sptr cams,
  map_landmark_t lms,
  feature_track_set_sptr tracks,
  frame_id_t fid_to_resection)
{
  LOG_DEBUG(m_logger, "resectioning camera " << fid_to_resection);
  auto model_intrinsics = m_base_camera.intrinsics();

  // Find the closest existing camera in time and
  // use those intrinsics if available
  auto const all_cameras = cams->cameras();
  if (!all_cameras.empty())
  {
    auto closest_cam = all_cameras.lower_bound(fid_to_resection);
    if (closest_cam == all_cameras.end())
    {
      // use the last camera
      --closest_cam;
    }
    auto const cam_p = std::static_pointer_cast<camera_perspective>(
      closest_cam->second);
    model_intrinsics = cam_p->intrinsics();
  }
  if (!m_force_common_intrinsics)
  {
    model_intrinsics = model_intrinsics->clone();
  }
  auto nc = std::make_shared<vital::simple_camera_perspective>();
  nc->set_intrinsics(model_intrinsics);


  // do 3PT algorithm here
  three_point_pose(fid_to_resection, nc, tracks, lms,
                   static_cast<float>(image_coverage_threshold), m_pnp);

  if (!nc)
  {
    // three point pose failed.
    m_frames_removed_from_sfm_solution.insert(fid_to_resection);
    return false;
  }
  else
  {
    cams->insert(fid_to_resection, nc);
    return true;
  }
}

bool
initialize_cameras_landmarks_keyframe::priv
::fit_reconstruction_to_constraints(
  simple_camera_perspective_map_sptr cams,
  map_landmark_t &lms,
  feature_track_set_sptr tracks,
  sfm_constraints_sptr constraints,
  int &num_constraints_used)
{
  similarity_d sim;
  bool estimated_sim = false;
  num_constraints_used = 0;
  if (constraints && !constraints->get_camera_position_priors().empty() &&
    m_similarity_estimator)
  {
    // Estimate a similarity transform to align the cameras
    // with the constraints
    int max_frame_diff = 20;

    auto pos_priors = constraints->get_camera_position_priors();

    auto persp_cams = cams->T_cameras();

    std::vector<vector_3d> cam_positions, sensor_positions;

    for (auto &cam : persp_cams)
    {
      auto cam_fid = cam.first;
      frame_id_t best_fid_difference = std::numeric_limits<frame_id_t>::max();
      vector_3d closest_prior;
      for (auto &prior : pos_priors)
      {
        auto prior_fid = prior.first;
        auto cur_diff = abs(prior_fid - cam_fid);
        if (cur_diff < best_fid_difference)
        {
          best_fid_difference = cur_diff;
          closest_prior = prior.second;
        }
      }
      if (best_fid_difference > max_frame_diff)
      {
        continue;
      }
      cam_positions.push_back(cam.second->center());
      sensor_positions.push_back(closest_prior);
    }

    if (cam_positions.size() > 5)
    {
      Eigen::Matrix<double, 3, Eigen::Dynamic> cam_mat;
      cam_mat.resize(3, cam_positions.size());
      for (size_t v = 0; v < cam_positions.size(); ++v)
      {
        cam_mat(0, v) = cam_positions[v].x();
        cam_mat(1, v) = cam_positions[v].y();
        cam_mat(2, v) = cam_positions[v].z();
      }

      Eigen::Vector3d cam_mean;
      cam_mean(0) = cam_mat.row(0).mean();
      cam_mean(1) = cam_mat.row(1).mean();
      cam_mean(2) = cam_mat.row(2).mean();

      auto cam_mat_centered = cam_mat - cam_mean.replicate(1, cam_mat.cols());

      auto cov_mat = cam_mat_centered * cam_mat_centered.transpose();

      Eigen::JacobiSVD<vital::matrix_3x3d> svd(cov_mat, Eigen::ComputeFullV);
      auto sing_vals = svd.singularValues();
      if (sing_vals(0) < 100 * sing_vals(1))
      {
        sim = m_similarity_estimator->estimate_transform(cam_positions,
                                                         sensor_positions);
        estimated_sim = true;
        num_constraints_used = static_cast<int>(sensor_positions.size());
      }
    }
  }

  if (!estimated_sim)
  {
    // If no constraints then estimate the similarity transform to
    // bring the solution into canonical pose
    if (!m_canonical_estimator)
    {
      LOG_DEBUG(m_logger, "No canonial transform estimator defined. "
        "Skipping fit to constraints.");
      return false;
    }
    auto landmarks = std::make_shared<simple_landmark_map>(lms);
    sim = m_canonical_estimator->estimate_transform(cams, landmarks);
  }

  // Transform all the points and cameras
  transform_inplace(lms, sim);
  transform_inplace(*cams, sim);

  // Find a set of stable landmarks to estimate a scene plane
  map_landmark_t stable_lms;
  if (cams->size() > 2)
  {
    for (auto const& lm : lms)
    {
      if (lm.second->observations() > 2)
      {
        stable_lms.insert(lm);
      }
    }
    if (stable_lms.size() < 8)
    {
      std::set<landmark_id_t> inlier_lms;
      retriangulate(lms, cams, tracks->tracks(), inlier_lms);
      return true;
    }
  }
  else
  {
    std::set<landmark_id_t> inlier_lms;
    retriangulate(lms, cams, tracks->tracks(), inlier_lms);
    return true;
  }

  // Test that the cameras are upright and necker reverse if not
  const vector_4d plane = landmark_plane(stable_lms);
  auto cams_above = cameras_above_plane(cams->map_of_<camera_perspective>(),
                                        plane);
  if (cams_above.size() > 0 && cams_above.size() < cams->size())
  {
    // cameras are on both sides of the plane
    std::set<landmark_id_t> inlier_lms;
    retriangulate(lms, cams, tracks->tracks(), inlier_lms);
    return true;
  }
  bool reversed_cameras = false;
  for (auto const& cam : cams->T_cameras())
  {
    if (!camera_upright(*cam.second))
    {
      necker_reverse_inplace(*(cam.second), plane);
      reversed_cameras = true;
      LOG_DEBUG(m_logger, "Necker revsersed camera " << cam.first);
    }
  }
  if (reversed_cameras)
  {
    std::set<landmark_id_t> inlier_lms;
    retriangulate(lms, cams, tracks->tracks(), inlier_lms, 2, 10);

    std::set<frame_id_t> fixed_cams_empty;
    std::set<landmark_id_t> fixed_lms_empty;
    bundle_adjuster->optimize(*cams, lms, tracks,
                              fixed_cams_empty, fixed_lms_empty);
  }

  std::set<landmark_id_t> inlier_lms;
  retriangulate(lms, cams, tracks->tracks(), inlier_lms);

  return true;
}

bool
initialize_cameras_landmarks_keyframe::priv
::initialize_reconstruction(
  simple_camera_perspective_map_sptr cams,
  map_landmark_t &lms,
  feature_track_set_sptr tracks)
{
  struct {
    bool operator()(const rel_pose &a, const rel_pose &b) const
    {
      return a.score() < b.score();
    }
  } well_conditioned_less;
  std::vector<rel_pose> rel_poses_inlier_ordered(m_rel_poses.begin(),
                                                 m_rel_poses.end());
  std::sort(rel_poses_inlier_ordered.begin(),
            rel_poses_inlier_ordered.end(), well_conditioned_less);
  std::reverse(rel_poses_inlier_ordered.begin(),
               rel_poses_inlier_ordered.end());
  std::set<frame_id_t> fixed_cams_empty;
  std::set<landmark_id_t> fixed_lms_empty;

  bool good_initialization = false;
  for (auto &rp_init : rel_poses_inlier_ordered)
  {
    cams->clear();
    lms.clear();
    auto cam_0 = std::make_shared<vital::simple_camera_perspective>();
    auto cam_1 = std::make_shared<vital::simple_camera_perspective>();
    LOG_DEBUG(m_logger, "Base focal length: " << m_base_camera.intrinsics()->focal_length());
    if (m_force_common_intrinsics)
    {
      cam_0->set_intrinsics(m_base_camera.intrinsics()->clone());
      cam_1->set_intrinsics(cam_0->intrinsics());
    }
    else
    {
      cam_0->set_intrinsics(m_base_camera.intrinsics()->clone());
      cam_1->set_intrinsics(m_base_camera.intrinsics()->clone());
    }
    cam_1->set_rotation(vital::rotation_d(rp_init.R01));
    cam_1->set_translation(rp_init.t01);
    cams->insert(rp_init.f0, cam_0);
    cams->insert(rp_init.f1, cam_1);

    auto trks = tracks->tracks();

    std::set<landmark_id_t> inlier_lm_ids;
    retriangulate(lms, cams, trks, inlier_lm_ids);

    LOG_DEBUG(m_logger, "rp_init.well_conditioned_landmark_count "
                        << rp_init.well_conditioned_landmark_count
                        << " inlier_lm_ids size " << inlier_lm_ids.size());

    auto rev_cams = std::make_shared<simple_camera_perspective_map>();
    auto rev_cam_0 = std::make_shared<simple_camera_perspective>(*cam_0);
    rev_cam_0->set_intrinsics(cam_0->intrinsics()->clone());
    auto rev_cam_1 = std::make_shared<simple_camera_perspective>(*cam_1);
    if (m_force_common_intrinsics)
    {
      rev_cam_1->set_intrinsics(rev_cam_0->intrinsics());
    }
    else
    {
      rev_cam_1->set_intrinsics(rev_cam_0->intrinsics()->clone());
    }

    std::map<landmark_id_t, landmark_sptr> inlier_lms;
    for (auto const& inlier_id : inlier_lm_ids)
    {
      inlier_lms[inlier_id] = lms[inlier_id];
    }
    auto plane = landmark_plane(inlier_lms);
    necker_reverse_inplace(*rev_cam_0, plane);
    necker_reverse_inplace(*rev_cam_1, plane);
    rev_cams->insert(rp_init.f0, rev_cam_0);
    rev_cams->insert(rp_init.f1, rev_cam_1);
    auto rev_lms = map_landmark_t(lms);

    if (bundle_adjuster)
    {
      LOG_INFO(m_logger, "Running Global Bundle Adjustment on "
        << cams->size() << " cameras and "
        << lms.size() << " landmarks");


      double init_rmse = kwiver::arrows::reprojection_rmse(cams->cameras(),
                                                           lms, trks);
      LOG_DEBUG(m_logger, "initial reprojection RMSE: " << init_rmse);

      bundle_adjuster->optimize(*cams, lms, tracks,
                                fixed_cams_empty, fixed_lms_empty);
      double optimized_rmse = kwiver::arrows::reprojection_rmse(cams->cameras(),
                                                                lms, trks);

      inlier_lm_ids.clear();
      retriangulate(rev_lms, rev_cams, trks, inlier_lm_ids, 2, 5.0);
      LOG_DEBUG(m_logger, "reversed inlier count " << inlier_lm_ids.size());
      double rev_optimized_rmse = std::numeric_limits<double>::infinity();
      if (inlier_lm_ids.size() > 5)
      {
        bundle_adjuster->optimize(*rev_cams, rev_lms, tracks,
                                  fixed_cams_empty, fixed_lms_empty);

        inlier_lm_ids.clear();
        retriangulate(rev_lms, rev_cams, trks, inlier_lm_ids, 2);
        LOG_DEBUG(m_logger, "reversed inlier count " << inlier_lm_ids.size());
        if (inlier_lm_ids.size() > 5)
        {
          bundle_adjuster->optimize(*rev_cams, rev_lms, tracks,
                                    fixed_cams_empty, fixed_lms_empty);
          rev_optimized_rmse = kwiver::arrows::reprojection_rmse(rev_cams->cameras(),
                                                                 rev_lms, trks);
        }
      }


      LOG_DEBUG(m_logger, "optimized reprojection RMSE: " << optimized_rmse
                          << " focal len: "
                          << cams->find(rp_init.f0)->get_intrinsics()->focal_length());
      LOG_DEBUG(m_logger, "optimized reversed reprojection RMSE: " << rev_optimized_rmse
                          << " focal len: "
                          << rev_cams->find(rp_init.f0)->get_intrinsics()->focal_length());

      if (rev_optimized_rmse < optimized_rmse)
      {
        LOG_DEBUG(m_logger, "Using reversed initial cameras");
        cams = rev_cams;
        lms = rev_lms;
      }

      inlier_lm_ids.clear();
      retriangulate(lms, cams, trks, inlier_lm_ids);

      LOG_DEBUG(m_logger, "after ba rp_init.well_conditioned_landmark_count "
                          << rp_init.well_conditioned_landmark_count
                          << " inlier_lm_ids size " << inlier_lm_ids.size());

      std::vector<frame_id_t> removed_cams;
      std::set<frame_id_t> variable_cams;
      std::set<landmark_id_t> variable_lms;

      clean_cameras_and_landmarks(*cams, lms, tracks, m_thresh_triang_cos_ang,
                                  removed_cams, variable_cams, variable_lms,
                                  image_coverage_threshold,
                                  interim_reproj_thresh);

      if (cams->size() < 2)
      {
        continue;
      }
    }
    if (lms.size() > 0.1 * rp_init.well_conditioned_landmark_count)
    {
      LOG_DEBUG(m_logger, "initialization pair is  "
                          << rp_init.f0 << ", " << rp_init.f1);
      good_initialization = true;
      break;
    }
  }
  return good_initialization;
}

template<class T>
class set_map {
public:
  set_map() {};
  ~set_map() {};
  void add_set(const std::set<T> &in_set)
  {
    size_t hash = hash_set(in_set);
    if (!contains_set(in_set, hash))
    {
      m_map.insert(std::pair<size_t,std::set<T>>(hash, in_set));
    }
  }

  bool remove_set(const std::set<T> &rem_set)
  {
    size_t hash = hash_set(rem_set);
    auto it_pair = m_map.equal_range(hash);
    for (auto it = it_pair.first; it != it_pair.second; ++it)
    {
      std::set<T> &potential_match_set = *it;
      if (rem_set == potential_match_set)
      {
        m_map.erase(it);
        return true;
      }
    }
    return false;
  }

  bool contains_set(const std::set<T> &test_set)
  {
    size_t hash = hash_set(test_set);
    return contains_set(test_set, hash);
  }

  bool contains_set(const std::set<T> &test_set, size_t hash)
  {
    auto it_pair = m_map.equal_range(hash);
    for (auto it = it_pair.first; it != it_pair.second; ++it)
    {
      std::set<T> &potential_match_set = it->second;
      if (test_set == potential_match_set)
      {
        return true;
      }
    }
    return false;
  }

private:
  size_t hash_set(const std::set<T> &to_hash)
  {
    size_t sum = 0;
    for (auto & val : to_hash)
    {
      sum += val;
    }
    return sum;
  }
  std::unordered_multimap<size_t,std::set<T>> m_map;
};

void initialize_cameras_landmarks_keyframe::priv
::remove_redundant_keyframe(
  simple_camera_perspective_map_sptr cams,
  landmark_map_sptr& landmarks,
  feature_track_set_sptr tracks,
  sfm_constraints_sptr constraints,
  frame_id_t target_frame)
{

  if (m_track_map.empty())
  {
    // only build this once.
    auto tks = tracks->tracks();
    for (auto const& t : tks)
    {
      m_track_map[t->id()] = t;
    }
  }

  auto lmks = landmarks->landmarks();
  // build up camera to landmark and landmark to camera maps
  int landmarks_involving_target_frame = 0;
  int landmarks_lost_without_target_frame = 0;

  auto c = cams->find(target_frame);

  if (!c || (c->center().x() == 0 && c->center().y() == 0 && c->center().z() == 0))
  {
    return;
  }

  for (auto const& lm : lmks)
  {
    auto t = lm.first;
    bool found_target_frame = false;
    int lm_inlier_meas = 0;

    auto tk_it = m_track_map.find(t);
    if (tk_it == m_track_map.end())
    {
      continue;
    }
    auto tk = tk_it->second;
    for (auto ts_it = tk->begin(); ts_it != tk->end(); ++ts_it)
    {
      auto fts = static_cast<feature_track_state*>(ts_it->get());
      if (!fts->inlier)
      {
        continue;
      }

      auto const f = fts->frame();

      if (!cams->find(f))
      {
        continue;
      }
      // landmark is an inlier to one of the cameras in the reconstruction
      if (f == target_frame)
      {
        found_target_frame = true;
      }
      ++lm_inlier_meas;
    }

    if (found_target_frame)
    {
      ++landmarks_involving_target_frame;
      if (lm_inlier_meas <= 3)
      {
        ++landmarks_lost_without_target_frame;
      }
    }
  }

  if (landmarks_lost_without_target_frame < 0.1 * landmarks_involving_target_frame)
  {
    // less than 5% of the landmarks involving the target frame would have
    // less than three masurements if the target frame was removed.
    // So we will remove it.
    cams->erase(target_frame);

    std::set<landmark_id_t> inlier_lms;
    retriangulate(lmks, cams, tracks->tracks(), inlier_lms);

    landmarks = landmark_map_sptr(new simple_landmark_map(lmks));
  }
}

void initialize_cameras_landmarks_keyframe::priv
::remove_redundant_keyframes(
  simple_camera_perspective_map_sptr cams,
  landmark_map_sptr& landmarks,
  feature_track_set_sptr tracks,
  sfm_constraints_sptr constraints,
  std::deque<frame_id_t> &recently_added_frame_queue)
{
  if (recently_added_frame_queue.size() >= 5)
  {
    std::set<frame_id_t> latest_frames;
    for (auto it = recently_added_frame_queue.begin();
         it != recently_added_frame_queue.end(); ++it)
    {
      latest_frames.insert(*it);
    }
    recently_added_frame_queue.pop_front();

    for (int rem_try = 0; rem_try < 10; ++rem_try)
    {
      auto cameras = cams->cameras();
      // guaranteed unbiased
      std::uniform_int_distribution<size_t> uni(0, cameras.size() - 1);
      auto ri = uni(m_rng);
      auto cams_it = cameras.begin();
      for (unsigned int i = 0; i < ri; ++i)
      {
        ++cams_it;
      }
      frame_id_t potential_rem_frame = cams_it->first;
      if (latest_frames.find(potential_rem_frame) != latest_frames.end())
      {
        continue;
      }
      remove_redundant_keyframe(cams, landmarks, tracks,
                                constraints, potential_rem_frame);
    }
  }
}

std::set<frame_id_t>
initialize_cameras_landmarks_keyframe::priv
::select_begining_frames_for_initialization(
  feature_track_set_sptr tracks ) const
{
  std::set<frame_id_t> beginning_keyframes;
  auto keyframes = this->m_keyframes;

  auto first_fid = *keyframes.begin();
  auto ff_tracks = tracks->active_tracks(first_fid);
  auto all_frames = tracks->all_frame_ids();

  if (m_frac_frames_for_init > 0)
  {
    size_t num_init_keyframes = static_cast<size_t>(keyframes.size() *
                                                    m_frac_frames_for_init);
    for (auto kfid : keyframes)
    {
      beginning_keyframes.insert(kfid);
      if (beginning_keyframes.size() >= num_init_keyframes)
      {
        return beginning_keyframes;
      }
    }
  }

  std::vector<frame_id_t> last_continuous_track_frames;
  for (auto t : ff_tracks)
  {
    int tl = 0;
    frame_id_t lf = -1;
    for (auto fid : all_frames)
    {
      if (t->find(fid) != t->end())
      {
        ++tl;
        lf = fid;
      }
      else
      {
        break;
      }
    }
    // exclude very short tracks
    if (lf != -1 && tl > 2)
    {
      last_continuous_track_frames.push_back(lf);
    }
  }

  if (last_continuous_track_frames.empty())
  {
    return beginning_keyframes;
  }
  std::sort(last_continuous_track_frames.begin(),
            last_continuous_track_frames.end());

  auto last_kf_for_init =
    last_continuous_track_frames[static_cast<size_t>(0.7 *
                                   last_continuous_track_frames.size())];

  LOG_DEBUG(m_logger, "last_kf_for_init " << last_kf_for_init);

  for (auto kf_id : keyframes)
  {
    if (kf_id > last_kf_for_init)
    {
      break;
    }
    beginning_keyframes.insert(kf_id);
    if (beginning_keyframes.size() >= 40)
    {
      break;
    }
  }

  return beginning_keyframes;
}

feature_track_set_changes_sptr
initialize_cameras_landmarks_keyframe::priv
::get_feature_track_changes(
  feature_track_set_sptr tracks,
  const simple_camera_perspective_map &cams) const
{
  auto chgs = std::make_shared<feature_track_set_changes>();
  /*
  for (auto &cam : cams.T_cameras())
  {
    auto fid = cam.first;
    auto at = tracks->active_tracks(fid);
    for (auto tk : at)
    {
      auto tk_it = tk->find(fid);
      if (tk_it == tk->end())
      {
        continue;
      }
      auto fts = std::dynamic_pointer_cast<feature_track_state>(*tk_it);
      if (!fts)
      {
        continue;
      }
      chgs->add_change(fid, tk->id(), fts->inlier);
    }
  }
  */
  return chgs;
}

bool
initialize_cameras_landmarks_keyframe::priv
::vision_centric_keyframe_initialization(
  simple_camera_perspective_map_sptr cams,
  landmark_map_sptr& landmarks,
  feature_track_set_sptr tracks,
  sfm_constraints_sptr constraints,
  std::set<vital::frame_id_t> &keyframes,
  callback_t callback)
{

  auto beginning_keyframes = keyframes;

  map_landmark_t lms;

  LOG_DEBUG(m_logger, "beginning_keyframes size " << beginning_keyframes.size());

  // get relative pose constraints for keyframes
  calc_rel_poses(beginning_keyframes, tracks);

  if (!this->continue_processing)
  {
    return false;
  }

  if (!initialize_reconstruction(cams, lms, tracks))
  {
    LOG_DEBUG(m_logger, "unable to find a good initial pair for reconstruction");
    return false;
  }

  int num_constraints_used;
  if (fit_reconstruction_to_constraints(cams, lms, tracks,
                                        constraints, num_constraints_used))
  {
    std::set<frame_id_t> fixed_cams_empty;
    std::set<landmark_id_t> fixed_lms_empty;
    LOG_INFO(m_logger, "Running Global Bundle Adjustment on "
                       << cams->size() << " cameras and "
                       << lms.size() << " landmarks");
    bundle_adjuster->optimize(*cams, lms, tracks, fixed_cams_empty, fixed_lms_empty);
  }

  if (callback)
  {
    auto chgs = get_feature_track_changes(tracks, *cams);
    continue_processing =
      callback(cams, std::make_shared<simple_landmark_map>(lms), chgs);

    if (!continue_processing)
    {
      LOG_DEBUG(m_logger,
        "continue processing is false, exiting initialize loop");
      landmarks = landmark_map_sptr(new simple_landmark_map(lms));

      return true;
    }
  }

  // list remaining frames to resection
  std::set<frame_id_t> frames_to_resection = keyframes;
  auto sc_map = cams->T_cameras();
  for (auto c : sc_map)
  {
    frames_to_resection.erase(c.first);
  }

  sfm_constraints_sptr ba_constraints = nullptr;

  int prev_ba_lm_count = static_cast<int>(lms.size());
  auto trks = tracks->tracks();

  std::set<frame_id_t> frames_that_were_in_sfm_solution;
  std::set<frame_id_t> frames_that_failed_resection;

  for (auto c : cams->get_frame_ids())
  {
    frames_that_were_in_sfm_solution.insert(c);
  }

  int frames_resectioned_since_last_ba = 0;
  std::deque<frame_id_t> added_frame_queue;
  while (this->continue_processing &&
    !frames_to_resection.empty() &&
    (m_max_cams_in_keyframe_init < 0 ||
     cams->size() < static_cast<size_t>(m_max_cams_in_keyframe_init)))
  {
    frame_id_t next_frame_id = select_next_camera(frames_to_resection,
                                                  cams, lms, tracks);
    if (next_frame_id == -1)
    {
      break;
    }

    if (!resection_camera(cams, lms, tracks, next_frame_id))
    {
      frames_to_resection.erase(next_frame_id);
      frames_that_failed_resection.insert(next_frame_id);
      continue;
    }
    LOG_INFO(m_logger, "Resectioned frame " << next_frame_id);

    frames_that_were_in_sfm_solution.insert(next_frame_id);
    for (auto fid : frames_that_failed_resection)
    {
      if (frames_that_were_in_sfm_solution.find(fid) !=
          frames_that_were_in_sfm_solution.end())
      {
        // skip frames that were in the sfm solution but were later dropped
        continue;
      }
      frames_to_resection.insert(fid);
    }

    ++frames_resectioned_since_last_ba;

    added_frame_queue.push_back(next_frame_id);

    frames_to_resection.erase(next_frame_id);

    {
      // bundle adjust fixing all cameras but the new one
      auto cameras = cams->cameras();

      double before_new_cam_rmse =
        kwiver::arrows::reprojection_rmse(cameras, lms, trks);
      LOG_DEBUG(m_logger, "before new camera reprojection RMSE: "
                         << before_new_cam_rmse);

      std::set<frame_id_t> fixed_cameras;
      std::set<landmark_id_t> fixed_landmarks;
      for (auto c : cameras)
      {
        if (c.first != next_frame_id)
        {
          fixed_cameras.insert(c.first);
        }
      }

      LOG_DEBUG(m_logger, "Optimizing frame " << next_frame_id
                          << cams->size()-1 << " fixed cameras and "
                          << lms.size() << " landmarks");
      auto ba_config = bundle_adjuster->get_configuration();
      bool opt_focal_was_set = ba_config->get_value<bool>("optimize_focal_length");
      ba_config->set_value<bool>("optimize_focal_length", false);
      bundle_adjuster->set_configuration(ba_config);
      bundle_adjuster->optimize(*cams, lms, tracks,
                                fixed_cameras, fixed_landmarks,
                                ba_constraints);
      ba_config->set_value<bool>("optimize_focal_length", opt_focal_was_set);
      bundle_adjuster->set_configuration(ba_config);

      double after_new_cam_rmse =
        kwiver::arrows::reprojection_rmse(cams->cameras(), lms, trks);
      LOG_DEBUG(m_logger, "after new camera reprojection RMSE: "
                         << after_new_cam_rmse);

    }
    int num_constraints_used;
    ba_constraints = nullptr;
    m_solution_was_fit_to_constraints = false;
    if (fit_reconstruction_to_constraints(cams, lms, tracks,
                                          constraints, num_constraints_used))
    {
      if (num_constraints_used > 2)
      {
        ba_constraints = constraints;
        m_solution_was_fit_to_constraints = true;
      }
    }
    else
    {
      // fit_reconstruction_to_constraints does a retriangulation if successful.
      // This call makes sure we triangulate features for the newly resectioned
      // camera if it wasn't.
      std::set<landmark_id_t> inlier_lm_ids;
      retriangulate(lms, cams, trks, inlier_lm_ids);
    }

    {
      std::vector<frame_id_t> removed_cams;
      std::set<frame_id_t> variable_cams;
      std::set<landmark_id_t> variable_lms;
      clean_cameras_and_landmarks(*cams, lms, tracks, m_thresh_triang_cos_ang,
                                  removed_cams, variable_cams, variable_lms,
                                  image_coverage_threshold,
                                  interim_reproj_thresh);

      for (auto rem_fid : removed_cams)
      {
        m_frames_removed_from_sfm_solution.insert(rem_fid);
      }
    }

    int next_ba_cam_count = std::max<int>(static_cast<int>(cams->size() * 0.2), 5);

    if ((lms.size() > prev_ba_lm_count * 1.5 ||
      lms.size() < prev_ba_lm_count * 0.5) ||
      frames_resectioned_since_last_ba >= next_ba_cam_count ||
      frames_to_resection.empty() || cams->size() < 30)
    {
      frames_resectioned_since_last_ba = 0;
      // bundle adjust result because number of inliers has changed significantly
      if (bundle_adjuster)
      {
        double before_clean_rmse =
          kwiver::arrows::reprojection_rmse(cams->cameras(), lms, trks);
        LOG_DEBUG(m_logger, "before clean reprojection RMSE: "
                            << before_clean_rmse);

        std::vector<frame_id_t> removed_cams;
        std::set<frame_id_t> variable_cams;
        std::set<landmark_id_t> variable_lms;
        clean_cameras_and_landmarks(*cams, lms, tracks,
                                    m_thresh_triang_cos_ang, removed_cams,
                                    variable_cams, variable_lms,
                                    image_coverage_threshold,
                                    interim_reproj_thresh);
        for (auto rem_fid : removed_cams)
        {
          m_frames_removed_from_sfm_solution.insert(rem_fid);
        }

        double init_rmse =
          kwiver::arrows::reprojection_rmse(cams->cameras(), lms, trks);
        LOG_DEBUG(m_logger, "initial reprojection RMSE: " << init_rmse);

        // first a BA fixing all landmarks to correct the cameras
        std::set<frame_id_t> fixed_cameras;
        std::set<landmark_id_t> fixed_landmarks;
        for (auto l : lms)
        {
          fixed_landmarks.insert(l.first);
        }
        // now an overall ba
        LOG_INFO(m_logger, "Running optimization on "
                           << cams->size() << " cameras with "
                           << lms.size() << " fixed landmarks");
        bundle_adjuster->optimize(*cams, lms, tracks,
                                  fixed_cameras, fixed_landmarks,
                                  ba_constraints);

        double optimized_rmse =
          kwiver::arrows::reprojection_rmse(cams->cameras(), lms, trks);
        LOG_DEBUG(m_logger, "optimized reprojection RMSE: "
                            << optimized_rmse);
        std::set<landmark_id_t> inlier_lm_ids;
        retriangulate(lms, cams, trks, inlier_lm_ids);

        // now an overall ba
        LOG_INFO(m_logger, "Running Global Bundle Adjustment on "
                           << cams->size() << " cameras and "
                           << lms.size() << " landmarks");
        fixed_landmarks.clear();
        bundle_adjuster->optimize(*cams, lms, tracks,
                                  fixed_cameras, fixed_landmarks,
                                  ba_constraints);

        clean_cameras_and_landmarks(*cams, lms, tracks,
                                    m_thresh_triang_cos_ang, removed_cams,
                                    variable_cams, variable_lms,
                                    image_coverage_threshold,
                                    interim_reproj_thresh);
        for (auto rem_fid : removed_cams)
        {
          m_frames_removed_from_sfm_solution.insert(rem_fid);
        }

        if (m_reverse_ba_error_ratio > 0 &&
            !m_solution_was_fit_to_constraints)
        {
          // reverse cameras and optimize again
          auto nr_cams_perspec =
            std::make_shared<simple_camera_perspective_map>(cams->T_cameras());
          auto nr_cams = std::static_pointer_cast<camera_map>(nr_cams_perspec);

          landmark_map_sptr ba_lms2(new simple_landmark_map(lms));
          necker_reverse(nr_cams, ba_lms2, false);
          // set from base is required because necker_reverse returns new cameras
          nr_cams_perspec->set_from_base_cams(nr_cams);
          map_landmark_t nr_landmarks;

          std::set<landmark_id_t> nr_inlier_lms;
          retriangulate(nr_landmarks, nr_cams_perspec, trks, nr_inlier_lms);
          int rev_num_constraints_used;
          fit_reconstruction_to_constraints(nr_cams_perspec, nr_landmarks,
                                            tracks, ba_constraints,
                                            rev_num_constraints_used);

          init_rmse = kwiver::arrows::reprojection_rmse(nr_cams_perspec->cameras(),
                                                        nr_landmarks, trks);
          LOG_DEBUG(m_logger, "Necker reversed initial reprojection RMSE: "
                              << init_rmse);
          if (init_rmse < optimized_rmse * m_reverse_ba_error_ratio)
          {

            landmark_map_sptr nr_landmark_map(new simple_landmark_map(nr_landmarks));

            LOG_INFO(m_logger, "Running Necker reversed bundle adjustment for comparison");

            double before_final_ba_rmse2 =
              kwiver::arrows::reprojection_rmse(nr_cams->cameras(),
                                                nr_landmark_map->landmarks(),
                                                trks);
            LOG_DEBUG(m_logger, "Necker reversed before final reprojection RMSE: "
                                << before_final_ba_rmse2);

            global_bundle_adjuster->optimize(*nr_cams_perspec, nr_landmarks, tracks,
                                             fixed_cameras, fixed_landmarks,
                                             ba_constraints);

            double final_rmse2 =
              kwiver::arrows::reprojection_rmse(nr_cams->cameras(),
                                                nr_landmarks, trks);
            LOG_DEBUG(m_logger, "Necker reversed final reprojection RMSE: "
                                << final_rmse2);

            if (final_rmse2 < optimized_rmse)
            {
              LOG_INFO(m_logger, "Necker reversed solution is better");
              cams->set_from_base_cams(nr_cams);
              lms = nr_landmarks;
            }
          }
        }

        prev_ba_lm_count = static_cast<int>(lms.size());

        if (!continue_processing)
        {
          break;
        }
      }
    }
    if (callback)
    {
      auto chgs = get_feature_track_changes(tracks, *cams);
      continue_processing =
        callback(cams, std::make_shared<simple_landmark_map>(lms), chgs);

      if (!continue_processing)
      {
        LOG_DEBUG(m_logger,
          "continue processing is false, exiting initialize loop");
        break;
      }
    }
  }

  if (continue_processing)
  {
    LOG_INFO(m_logger, "Running Final Bundle Adjustment of initial keyframes "
                       << "with " << cams->size() << " cameras and "
                       << lms.size() << " landmarks");
    std::set<frame_id_t> fixed_cameras;
    std::set<landmark_id_t> fixed_landmarks;
    bundle_adjuster->optimize(*cams, lms, tracks,
                              fixed_cameras, fixed_landmarks,
                              ba_constraints);

    std::vector<frame_id_t> removed_cams;
    std::set<frame_id_t> empty_cam_set;
    std::set<landmark_id_t> empty_lm_set;
    clean_cameras_and_landmarks(*cams, lms, tracks,
                                m_thresh_triang_cos_ang, removed_cams,
                                empty_cam_set, empty_lm_set,
                                image_coverage_threshold,
                                interim_reproj_thresh);
  }

  landmarks = landmark_map_sptr(new simple_landmark_map(lms));

  return true;
}

bool
initialize_cameras_landmarks_keyframe::priv
::metadata_centric_keyframe_initialization(
  simple_camera_perspective_map_sptr cams,
  bool use_existing_cams,
  landmark_map_sptr& landmarks,
  feature_track_set_sptr tracks,
  sfm_constraints_sptr constraints,
  std::set<vital::frame_id_t> &keyframes,
  callback_t callback)
{
  auto intrinsics = m_base_camera.get_intrinsics()->clone();
  if (!use_existing_cams)
  {
    // initialize the cameras from metadata
    cams->clear();

    for (auto fid : keyframes)
    {
      vector_3d pos_loc;
      rotation_d R_loc;

      if (constraints->get_camera_position_prior_local(fid, pos_loc) &&
          constraints->get_camera_orientation_prior_local(fid, R_loc))
      {
        auto cam = std::make_shared<simple_camera_perspective>();

        cam->set_center(pos_loc);
        cam->set_rotation(R_loc);
        if (m_force_common_intrinsics)
        {
          cam->set_intrinsics(intrinsics);
        }
        else
        {
          cam->set_intrinsics(intrinsics->clone());
        }
        cams->insert(fid, cam);
      }
    }
  }
  else
  {
    // use the existing camera poses
    if (m_init_intrinsics_from_metadata || m_config_defines_base_intrinsics)
    {
      // set each camera's intrinsics to match the base camera's
      // that was set from the metadata
      auto sp_cams = cams->T_cameras();
      for (auto cam : sp_cams)
      {
        cam.second->set_intrinsics(intrinsics);
      }
    }
  }

  if (cams->size() < 2)
  {
    return false;
  }

  auto trks = tracks->tracks();
  map_landmark_t lms;
  std::set<landmark_id_t> inlier_lm_ids;
  size_t prev_inlier_lm_count = 0;
  size_t cur_inlier_lm_count = 0;
  int iterations = 0;
  int num_permissive_triangulation_iterations = 3;
  int min_non_permissive_triangulation_iterations = 2;
  auto ba_config = global_bundle_adjuster->get_configuration();
  bool opt_focal_was_set = ba_config->get_value<bool>("optimize_focal_length");
  do {
    prev_inlier_lm_count = cur_inlier_lm_count;

    if (iterations < num_permissive_triangulation_iterations)
    {
      retriangulate(lms, cams, trks, inlier_lm_ids, 3,
                    m_metadata_init_permissive_triang_thresh);
    }
    else
    {
      retriangulate(lms, cams, trks, inlier_lm_ids, 3);
    }


    if (iterations < num_permissive_triangulation_iterations)
    {
      // set all tracks to inliers
      for (auto lm : lms)
      {
        auto tid = lm.first;
        auto tk = tracks->get_track(tid);
        if (!tk)
        {
          continue;
        }
        for (auto ts_it : *tk)
        {
          auto fts = std::dynamic_pointer_cast<feature_track_state>(ts_it);
          if (!fts)
          {
            continue;
          }
          if (!cams->find(fts->frame()))
          {
            continue;
          }
          fts->inlier = true;
        }
      }
      ba_config->set_value<bool>("optimize_focal_length", false);
    }
    else
    {
      ba_config->set_value<bool>("optimize_focal_length", opt_focal_was_set);
    }
    global_bundle_adjuster->set_configuration(ba_config);

    double init_rmse =
      kwiver::arrows::reprojection_rmse(cams->cameras(), lms, trks);
    LOG_DEBUG(m_logger, "initial reprojection RMSE: " << init_rmse);

    std::set<frame_id_t> fixed_cameras;
    std::set<landmark_id_t> fixed_landmarks;
    global_bundle_adjuster->optimize(*cams, lms, tracks,
                                     fixed_cameras, fixed_landmarks,
                                     constraints);

    double optimized_rmse =
      kwiver::arrows::reprojection_rmse(cams->cameras(), lms, trks);
    LOG_DEBUG(m_logger, "optimized reprojection RMSE: " << optimized_rmse);

    std::vector<frame_id_t> removed_cams;
    std::set<frame_id_t> empty_cam_set;
    std::set<landmark_id_t> empty_lm_set;
    double coverage_thresh = iterations == 0 ? 0 : image_coverage_threshold;

    // gradually tighten the reprojection error threshold
    // after the initial permissive iterations
    double non_permissive_threshold =
      ((40.0 / pow(1.25,
        iterations - num_permissive_triangulation_iterations + 1)) + 10.0) *
          interim_reproj_thresh;

    double reproj_thresh =
      iterations < num_permissive_triangulation_iterations ?
        50.0 * interim_reproj_thresh :
        non_permissive_threshold;
    clean_cameras_and_landmarks(*cams, lms, tracks,
                                m_thresh_triang_cos_ang, removed_cams,
                                empty_cam_set, empty_lm_set,
                                coverage_thresh, reproj_thresh, 3);

    // If more than one connected component, find critical tracks that tie
    // the components back together and retriangulate them.
    auto cc = connected_camera_components(cams->T_cameras(), lms, tracks);
    if (cc.size() > 1)
    {
      auto critical_tracks = detect_critical_tracks(cc, tracks);
      retriangulate(lms, cams, critical_tracks, inlier_lm_ids);
      LOG_DEBUG(m_logger, "Found " << cc.size() << " connected components "
                          "with " << critical_tracks.size()
                          << " connecting them.\n"
                          "Retriangulated " << inlier_lm_ids.size()
                          << " landmarks");
    }

    cur_inlier_lm_count = lms.size();
    ++iterations;

    if (callback)
    {
      auto chgs = get_feature_track_changes(tracks, *cams);
      continue_processing =
        callback(cams, std::make_shared<simple_landmark_map>(lms), chgs);

      if (!continue_processing)
      {
        LOG_DEBUG(m_logger,
          "continue processing is false, exiting initialize loop");
        break;
      }
    }


  } while (cur_inlier_lm_count > prev_inlier_lm_count ||
           iterations < (num_permissive_triangulation_iterations +
                         min_non_permissive_triangulation_iterations));

  landmarks = landmark_map_sptr(new simple_landmark_map(lms));

  if (landmarks->size() > 0 && cams->size() >= 2)
  {
    return true;
  }

  return false;
}

bool
initialize_cameras_landmarks_keyframe::priv
::initialize_keyframes(
  simple_camera_perspective_map_sptr cams,
  landmark_map_sptr& landmarks,
  feature_track_set_sptr tracks,
  sfm_constraints_sptr constraints,
  callback_t callback)
{
  LOG_DEBUG(m_logger, "Start initialize_keyframes");

  m_solution_was_fit_to_constraints = false;

  // get set of keyframe ids
  auto keyframes = this->m_keyframes;

  // now try using the metadata to initialize the reconstruction
  if (metadata_centric_keyframe_initialization(cams, false, landmarks, tracks,
                                               constraints, keyframes,
                                               callback))
  {
    return true;
  }

  if (keyframes.empty())
  {
    LOG_DEBUG(m_logger, "No keyframes, cannot initilize reconstruction");
    return false;
  }

  // now try a vision only approach
  if(vision_centric_keyframe_initialization(cams, landmarks, tracks,
                                            constraints, keyframes,
                                            callback))
  {
    return true;
  }

  LOG_DEBUG(m_logger, "Vision centric initialization failed");
  return false;
}

int
initialize_cameras_landmarks_keyframe::priv
::get_inlier_count(frame_id_t fid,
                   landmark_map_sptr landmarks,
                   feature_track_set_sptr tracks)
{
  int inlier_count = 0;
  auto lmks = landmarks->landmarks();
  auto cur_tracks = tracks->active_tracks(fid);
  for (auto &t : cur_tracks)
  {
    auto ts = t->find(fid);
    if (ts == t->end())
    {
      continue;
    }

    auto fts = std::dynamic_pointer_cast<feature_track_state>(*ts);
    if (!fts || !fts->feature)
    {
      continue;
    }

    if (lmks.find(t->id()) == lmks.end())
    {
      continue;
    }
    if (fts->inlier)
    {
      ++inlier_count;
    }
  }
  return inlier_count;
}

int initialize_cameras_landmarks_keyframe::priv
::set_inlier_flags(
    frame_id_t fid,
    simple_camera_perspective_sptr cam,
    const map_landmark_t &lms,
    feature_track_set_sptr tracks,
    double reporj_thresh)
{
  const double reporj_thresh_sq = reporj_thresh*reporj_thresh;
  int inlier_count = 0;
  auto vital_cam = std::static_pointer_cast<camera>(cam);
  auto cur_tracks = tracks->active_tracks(fid);
  for (auto &t : cur_tracks)
  {
    auto ts = t->find(fid);
    if (ts == t->end())
    {
      continue;
    }

    auto fts = std::dynamic_pointer_cast<feature_track_state>(*ts);
    if (!fts || !fts->feature)
    {
      continue;
    }

    auto lm_it = lms.find(t->id());

    if (lm_it == lms.end())
    {
      continue;
    }

    double err = reprojection_error_sqr(*vital_cam,
                                        *lm_it->second, *fts->feature);
    if (err < reporj_thresh_sq)
    {
      fts->inlier = true;
      ++inlier_count;
    }
    else
    {
      fts->inlier = false;
    }
  }
  return inlier_count;
}

void
initialize_cameras_landmarks_keyframe::priv
::cleanup_necker_reversals(
  simple_camera_perspective_map_sptr cams,
  landmark_map_sptr landmarks,
  feature_track_set_sptr tracks,
  sfm_constraints_sptr constraints)
{
  // first record all camera positions
  std::map<frame_id_t, vector_3d> orig_positions;
  auto spc = cams->T_cameras();
  std::set<frame_id_t> fixed_cams;
  for (auto &c : spc)
  {
    orig_positions[c.first] = c.second->center();
    fixed_cams.insert(c.first);
  }

  auto lms = landmarks->landmarks();
  std::set<landmark_id_t> fixed_landmarks;
  for (auto const&lm : lms)
  {
    fixed_landmarks.insert(lm.first);
  }

  simple_camera_perspective_sptr prev_cam;
  for (auto &cur_cam_pair : spc)
  {
    auto fid = cur_cam_pair.first;
    if (prev_cam)
    {
      auto cur_cam = cur_cam_pair.second;
      auto nr_cam = std::make_shared<simple_camera_perspective_map>();
      nr_cam->insert(fid, std::static_pointer_cast<simple_camera_perspective>(
                            cur_cam->clone()));
      camera_map_sptr nr_cam_base(new simple_camera_map(nr_cam->cameras()));
      necker_reverse(nr_cam_base, landmarks, false);
      auto reversed_cam = std::static_pointer_cast<simple_camera_perspective>(
        nr_cam_base->cameras().begin()->second);

      auto non_rev_dist = (cur_cam->center() - prev_cam->center()).norm();
      auto rev_dist = (reversed_cam->center() - prev_cam->center()).norm();
      if (rev_dist < non_rev_dist)
      {
        // remove current frame from fixed cameras so it will be optimized
        fixed_cams.erase(fid);
        cams->insert(fid, reversed_cam);
        bundle_adjuster->optimize(*cams, lms, tracks,
                                  fixed_cams, fixed_landmarks,
                                  constraints);
        // now add it back to the fixed frames so it is fixed next time
        fixed_cams.insert(fid);
      }
    }
    // make sure we get the camera from cams in case it was changed by a reversal
    prev_cam = cams->find(fid);
  }

  // ok, now all cams should be consistent with the first cam.
  // Do I reverse them all first get all the camera pointers again,
  // in case they have changed.
  spc = cams->T_cameras();
  int reverse_it_if_positive = 0;
  for (auto cur_cam_pair : spc)
  {
    auto fid = cur_cam_pair.first;
    auto cur_cam = cur_cam_pair.second;
    auto nr_cam = std::make_shared<simple_camera_perspective_map>();
    nr_cam->insert(fid, std::static_pointer_cast<simple_camera_perspective>(
                          cur_cam->clone()));
    camera_map_sptr nr_cam_base(new simple_camera_map(nr_cam->cameras()));
    necker_reverse(nr_cam_base, landmarks, false);
    auto reversed_cam = std::static_pointer_cast<simple_camera_perspective>(
      nr_cam_base->cameras().begin()->second);
    auto orig_center = orig_positions[fid];
    auto non_rev_dist = (cur_cam->center() - orig_center).norm();
    auto rev_dist = (reversed_cam->center() - orig_center).norm();
    if (rev_dist < non_rev_dist)
    {
      ++reverse_it_if_positive;
    }
    else
    {
      --reverse_it_if_positive;
    }
  }

  if (reverse_it_if_positive > 0)
  {
    for (auto cur_cam_pair : spc)
    {
      auto fid = cur_cam_pair.first;
      auto cur_cam = cur_cam_pair.second;
      auto nr_cam = std::make_shared<simple_camera_perspective_map>();
      nr_cam->insert(fid, std::static_pointer_cast<simple_camera_perspective>(
                            cur_cam->clone()));
      camera_map_sptr nr_cam_base(new simple_camera_map(nr_cam->cameras()));
      necker_reverse(nr_cam_base, landmarks, false);
      auto reversed_cam = std::static_pointer_cast<simple_camera_perspective>(
        nr_cam_base->cameras().begin()->second);
      cams->insert(fid, reversed_cam);
    }
  }

  fixed_cams.clear();
  bundle_adjuster->optimize(*cams, lms, tracks,
                            fixed_cams, fixed_landmarks,
                            constraints);
}

std::set<landmark_id_t>
initialize_cameras_landmarks_keyframe::priv
::find_visible_landmarks_in_frames(
  const map_landmark_t &lmks,
  feature_track_set_sptr tracks,
  const std::set<frame_id_t> &frames)
{
  std::set<landmark_id_t> visible_landmarks;

  for (auto const fid : frames)
  {
    auto at = tracks->active_tracks(fid);
    for (auto t : at)
    {
      visible_landmarks.insert(t->id());
    }
  }

  return visible_landmarks;
}

void
initialize_cameras_landmarks_keyframe::priv
::get_registered_and_non_registered_frames(
  simple_camera_perspective_map_sptr cams,
  feature_track_set_sptr tracks,
  std::set<frame_id_t> &registered_frames,
  std::set<frame_id_t> &non_registered_frames) const
{
  registered_frames.clear();
  non_registered_frames.clear();

  auto pcams_map = cams->T_cameras();
  non_registered_frames = tracks->all_frame_ids();
  for (auto &p : pcams_map)
  {
    registered_frames.insert(p.first);
    non_registered_frames.erase(p.first);
  }
}

bool
initialize_cameras_landmarks_keyframe::priv
::get_next_fid_to_register_and_its_closest_registered_cam(
  simple_camera_perspective_map_sptr cams,
  std::set<frame_id_t> &frames_to_register,
  frame_id_t &fid_to_register, frame_id_t &closest_frame) const
{
  auto existing_cams = cams->T_cameras();
  frame_id_t min_frame_diff = std::numeric_limits<frame_id_t>::max();

  std::vector<std::pair<frame_id_t,frame_id_t>> min_diff_cams;

  for (auto f : frames_to_register)
  {
    for (auto &ec : existing_cams)
    {
      auto diff = abs(ec.first - f);
      if (diff < min_frame_diff)
      {
        min_diff_cams.clear();
        min_diff_cams.push_back(std::pair<frame_id_t, frame_id_t>(f, ec.first));
        min_frame_diff = diff;
      }
      else if (diff == min_frame_diff)
      {
        min_diff_cams.push_back(std::pair<frame_id_t, frame_id_t>(f, ec.first));
      }
    }
  }
  if (!min_diff_cams.empty())
  {
    std::random_shuffle(min_diff_cams.begin(), min_diff_cams.end());
    fid_to_register = min_diff_cams.begin()->first;
    closest_frame = min_diff_cams.begin()->second;
    return true;
  }
  else
  {
    return false;
  }
}

bool
initialize_cameras_landmarks_keyframe::priv
::initialize_next_camera(
  simple_camera_perspective_map_sptr cams,
  map_landmark_t& lmks,
  feature_track_set_sptr tracks,
  sfm_constraints_sptr constraints,
  frame_id_t &fid_to_register,
  std::set<frame_id_t> &frames_to_register,
  std::set<frame_id_t> &already_registred_cams)
{
  frame_id_t closest_cam_fid;
  if (!get_next_fid_to_register_and_its_closest_registered_cam(
        cams,
        frames_to_register,
        fid_to_register,
        closest_cam_fid))
  {
    return false;
  }
  frames_to_register.erase(fid_to_register);

  simple_camera_perspective_sptr closest_cam = cams->find(closest_cam_fid);
  simple_camera_perspective_sptr resectioned_cam, bundled_cam;
  int bundled_inlier_count = 0;
  int resection_inlier_count = 0;

  int min_inliers = 50;

  std::set<frame_id_t> cur_fid;
  cur_fid.insert(fid_to_register);
  auto cur_frame_landmarks =
    find_visible_landmarks_in_frames(lmks, tracks, cur_fid);
  auto cur_landmarks = get_sub_landmark_map(lmks, cur_frame_landmarks);

  // get a random subset of the current landmarks
  std::vector<landmark_id_t> lm_ids;
  lm_ids.reserve(cur_landmarks.size());
  for (auto &lm : cur_landmarks)
  {
    lm_ids.push_back(lm.first);
  }
  std::random_shuffle(lm_ids.begin(), lm_ids.end());
  map_landmark_t cur_landmarks_rand_sub;
  for (auto lm_id : lm_ids)
  {
    cur_landmarks_rand_sub[lm_id] = cur_landmarks[lm_id];
    if (cur_landmarks_rand_sub.size() >= 250)
    {
      break;
    }
  }

  auto ba_config = bundle_adjuster->get_configuration();
  bool opt_focal_was_set = ba_config->get_value<bool>("optimize_focal_length");
  ba_config->set_value<bool>("optimize_focal_length", false);
  bundle_adjuster->set_configuration(ba_config);

  for(int lp = 0; lp < 2; ++lp)
  {
    auto vel = get_velocity(cams, fid_to_register);
    if (lp == 1)
    {
      // try zero velocity if using the constant velocity model didn't work.
      vel.setZero();
    }
    // use the pose of the closest camera as starting point
    bundled_cam =
      std::static_pointer_cast<simple_camera_perspective>(closest_cam->clone());
    if (!m_force_common_intrinsics)
    {
      // make sure this camera has it's own intrinsics object
      bundled_cam->set_intrinsics(closest_cam->intrinsics()->clone());
    }

    // use constant velocity model
    bundled_cam->set_center(closest_cam->center() +
                            (fid_to_register - closest_cam_fid) * vel);
    cams->insert(fid_to_register, bundled_cam);

    int prev_bundled_inlier_count = -1;
    int loop_count = 0;
    bundled_inlier_count = set_inlier_flags(fid_to_register, bundled_cam,
                                            cur_landmarks, tracks, 50);

    while (bundled_inlier_count > prev_bundled_inlier_count)
    {
      // DOES THIS LOOP HELP?
      bundle_adjuster->optimize(*cams, cur_landmarks_rand_sub, tracks,
                                already_registred_cams, cur_frame_landmarks,
                                constraints);

      bundled_cam = cams->find(fid_to_register);
      prev_bundled_inlier_count = bundled_inlier_count;
      bundled_inlier_count = set_inlier_flags(fid_to_register, bundled_cam,
                                              cur_landmarks, tracks, 50);
      ++loop_count;
    }
    if (loop_count > 2)
    {
      LOG_DEBUG(m_logger, "ran " << loop_count << " hill climbing BA loops");
    }

    if (bundled_inlier_count >= 4 * min_inliers)
    {
      break;
    }
  }

  if (bundled_inlier_count < 4 * min_inliers)
  {
    if (resection_camera(cams, lmks, tracks, fid_to_register))
    {
      resectioned_cam = cams->find(fid_to_register);
      if (resectioned_cam)
      {
        int prev_resection_inlier_count = -1;
        int loop_count = 0;
        while (resection_inlier_count > prev_resection_inlier_count)
        {
          bundle_adjuster->optimize(*cams, cur_landmarks, tracks,
                                    already_registred_cams,
                                    cur_frame_landmarks,
                                    constraints);
          resectioned_cam = cams->find(fid_to_register);
          prev_resection_inlier_count = resection_inlier_count;
          resection_inlier_count =
            set_inlier_flags(fid_to_register, resectioned_cam,
                             cur_landmarks, tracks, 50);
          ++loop_count;
        }
        if (loop_count > 2)
        {
          LOG_DEBUG(m_logger, "ran " << loop_count
                              << " hill climbing resection BA loops");
        }
      }
    }
  }

  ba_config = bundle_adjuster->get_configuration();
  ba_config->set_value<bool>("optimize_focal_length", opt_focal_was_set);
  bundle_adjuster->set_configuration(ba_config);

  int inlier_count = std::max(resection_inlier_count, bundled_inlier_count);
  if (inlier_count < min_inliers)
  {
    cams->erase(fid_to_register);
    return false;
  }

  if (resection_inlier_count > bundled_inlier_count)
  {
    LOG_DEBUG(m_logger, "using resectioned camera for frame "
                        << fid_to_register
                        << " because resection inlier count "
                        << resection_inlier_count
                        << " greater than bundled inlier count "
                        << bundled_inlier_count);
    cams->insert(fid_to_register, resectioned_cam);
    set_inlier_flags(fid_to_register, resectioned_cam, lmks, tracks, 10);
  }
  else
  {
    cams->insert(fid_to_register, bundled_cam);
    set_inlier_flags(fid_to_register, bundled_cam, lmks, tracks, 10);
  }
  return true;
}

void
initialize_cameras_landmarks_keyframe::priv
::windowed_clean_and_bundle(
  simple_camera_perspective_map_sptr cams,
  landmark_map_sptr& landmarks,
  map_landmark_t& lmks,
  feature_track_set_sptr tracks,
  sfm_constraints_sptr constraints,
  const std::set<frame_id_t> &already_bundled_cams,
  const std::set<frame_id_t> &frames_since_last_local_ba)
{
  auto frames_to_fix = already_bundled_cams;
  // optimize camera and all landmarks it sees, fixing all other cameras.
  auto variable_frames = frames_since_last_local_ba;

  for (auto fid : variable_frames)
  {
    frames_to_fix.erase(fid);
  }

  std::set<frame_id_t> empty_frames;
  std::set<landmark_id_t> empty_landmarks;

  auto variable_landmark_ids =
    find_visible_landmarks_in_frames(lmks, tracks,
                                     frames_since_last_local_ba);
  std::vector<frame_id_t> removed_cams;
  clean_cameras_and_landmarks(*cams, lmks, tracks,
                              m_thresh_triang_cos_ang, removed_cams,
                              frames_since_last_local_ba,
                              variable_landmark_ids,
                              image_coverage_threshold,
                              interim_reproj_thresh);

  for (auto rem_fid : removed_cams)
  {
    m_frames_removed_from_sfm_solution.insert(rem_fid);
  }

  auto variable_landmarks =
    get_sub_landmark_map(lmks, variable_landmark_ids);
  std::set<landmark_id_t> empty_landmark_id_set;

  LOG_DEBUG(m_logger, "Bundle adjusting " << cams->size() << " cameras ("
                      << frames_to_fix.size() << " fixed) and "
                      << variable_landmarks.size() << " landmarks");
  bundle_adjuster->optimize(*cams, variable_landmarks, tracks,
                            frames_to_fix, empty_landmark_id_set,
                            constraints);

  clean_cameras_and_landmarks(*cams, lmks, tracks,
                              m_thresh_triang_cos_ang, removed_cams,
                              frames_since_last_local_ba,
                              variable_landmark_ids,
                              image_coverage_threshold,
                              interim_reproj_thresh, 3);

  for (auto rem_fid : removed_cams)
  {
    m_frames_removed_from_sfm_solution.insert(rem_fid);
  }

  landmarks = store_landmarks(lmks, variable_landmarks);
}


std::set<uint32_t> hash_point(vector_3d const& X_min,
                              vector_3d const& X,
                              double volume_unit_size)
{
  const int bins_per_dim(1000);
  auto X_bb = X - X_min;

  int x_bin = int(X_bb[0] / volume_unit_size) % bins_per_dim;
  int y_bin = int(X_bb[1] / volume_unit_size) % bins_per_dim;
  int z_bin = int(X_bb[2] / volume_unit_size) % bins_per_dim;

  std::set<uint32_t> hash_bins;

  for (int xx = -1; xx <= 1; ++xx)
  {
    for (int yy = -1; yy <= 1; ++yy)
    {
      for (int zz = -1; zz <= 1; ++zz)
      {
        uint32_t hash_val = (x_bin + xx) + bins_per_dim * (y_bin + yy) +
                            bins_per_dim * bins_per_dim * (z_bin + zz);
        hash_bins.insert(hash_val);
      }
    }
  }

  return hash_bins;
}


void
initialize_cameras_landmarks_keyframe::priv
::merge_landmarks(map_landmark_t &lmks,
                  simple_camera_perspective_map_sptr const &cams,
                  feature_track_set_sptr &tracks)
{

  // do I have any landmarks that don't have tracks?
  int num_landmarks_without_tracks = 0;
  for (auto &lm : lmks)
  {
    if (!tracks->get_track(lm.first))
    {
      ++num_landmarks_without_tracks;
    }
  }

  if (num_landmarks_without_tracks)
  {
    LOG_WARN(m_logger, "beginning num landmarks without tracks "
                       << num_landmarks_without_tracks);
  }


  // get bounding box of landmarks
  vector_3d X_min, X_max;
  X_min.setConstant(std::numeric_limits<double>::max());
  X_max.setConstant(-(std::numeric_limits<double>::max()/2));
  for (auto &lm : lmks)
  {
    auto X = lm.second->loc();
    for (int i = 0; i < 3; ++i)
    {
      X_min[i] = std::min<double>(X_min[i], X[i]);
      X_max[i] = std::max<double>(X_max[i], X[i]);
    }
  }

  // hash all landmarks

  const double volume_unit_size = 2.0;

  std::unordered_map<uint32_t, std::set<landmark_id_t>> lm_hash;
  lm_hash.reserve(lmks.size() / 2);

  for (auto &lm : lmks)
  {
    auto hashes = hash_point(X_min, lm.second->loc(), volume_unit_size);

    for (auto hv : hashes)
    {
      if (lm_hash.count(hv) == 0)
      {
        auto new_set = std::set<landmark_id_t>();
        new_set.insert(lm.first);
        lm_hash[hv] = new_set;
      }
      else
      {
        lm_hash[hv].insert(lm.first);
      }
    }
  }

  std::set<landmark_id_t> removed_landmarks;

  for (auto &lm : lmks)
  {
    if (m_already_merged_landmarks.count(lm.first) > 0)
    {
      // we already tried to merge this landmark
      continue;
    }

    if (removed_landmarks.count(lm.first) > 0)
    {
      // we already removed this landmark
      continue;
    }

    // check to make sure we have a track for the landmark
    auto cur_tk = tracks->get_track(lm.first);
    if (!cur_tk)
    {
      continue;
    }

    // Add the landmark to the already merged set so we won't try again.
    // Only when new landmarks are added to the set will we try to merge them
    // with landmarks near by.  This could cause problems after loop completion
    // when the model is bent significantly.  Therefore, at loop completion we
    // should clear m_already_marged_landmarks so that we try all the landmarks again.
    m_already_merged_landmarks.insert(lm.first);

    //ok get all the hashes for the point
    auto hashes = hash_point(X_min, lm.second->loc(), volume_unit_size);

    //collect all the points in the hash bins
    std::set< landmark_id_t> nearby_landmarks;
    for (auto hv : hashes)
    {
      auto it = lm_hash.find(hv);
      if (it == lm_hash.end())
      {
        // should not happen
        continue;
      }
      for (auto lm_id : it->second)
      {
        if (lm_id != lm.first && removed_landmarks.count(lm_id) == 0)
        {
          // a landmark is not near itself
          // only try to merge landmarks that have not already been removed
          nearby_landmarks.insert(lm_id);
        }
      }
    }

    for (auto nb_lm_id : nearby_landmarks)
    {
      auto lm_nb = lmks.find(nb_lm_id);
      if (lm_nb == lmks.end())
      {
        // should not happen
        continue;
      }
      auto lm_diff = lm_nb->second->loc() - lm.second->loc();
      if (lm_diff.norm() > volume_unit_size)
      {
        // only consider points actualy with one volume unit sphere of each other
        continue;
      }

      // average landmarks

      landmark_d lm_merged(0.5*(lm_nb->second->loc() + lm.second->loc()));

      // check reprojection errors into views of nearby landmark
      auto nb_tk_sptr = tracks->get_track(nb_lm_id);

      bool merge_successful = true;

      for (auto ts : *nb_tk_sptr)
      {
        auto fts = std::dynamic_pointer_cast<feature_track_state>(ts);
        if (!fts->inlier)
        {
          continue;
        }
        auto cam = cams->find(fts->frame());
        if(!cam)
        {
          continue;
        }

        double rpj_err = reprojection_error(*cam, lm_merged, *fts->feature);
        if (rpj_err > interim_reproj_thresh)
        {
          merge_successful = false;
          break;
        }
      }

      if (!merge_successful)
      {
        continue;
      }

      // check reprojection errors into views of current landmark

      for (auto ts : *cur_tk)
      {
        auto fts = std::dynamic_pointer_cast<feature_track_state>(ts);
        if (!fts->inlier)
        {
          continue;
        }
        auto cam = cams->find(fts->frame());
        if (!cam)
        {
          continue;
        }

        double rpj_err = reprojection_error(*cam, lm_merged, *fts->feature);
        if (rpj_err > interim_reproj_thresh)
        {
          merge_successful = false;
          break;
        }
      }

      if (!merge_successful)
      {
        continue;
      }

      /// NEED TO FINISH THIS AND TEST IT
      if (lm.first < nb_lm_id)
      {
        if (tracks->merge_tracks(nb_tk_sptr, cur_tk))
        {
          auto lm_d_sptr = std::dynamic_pointer_cast<landmark_d>(lm.second);
          if (lm_d_sptr)
          {
            lm_d_sptr->set_loc(lm_merged.loc());
          }
          else
          {
            auto lm_f_sptr = std::dynamic_pointer_cast<landmark_f>(lm.second);
            if (lm_f_sptr)
            {
              lm_f_sptr->set_loc(lm_merged.loc().cast<float>());
            }
          }
          removed_landmarks.insert(nb_tk_sptr->id());
        }
      }
      else
      {
        auto nb_lm_it = lmks.find(nb_tk_sptr->id());
        if (nb_lm_it == lmks.end())
        {
          // should not happen
          continue;
        }
        if (tracks->merge_tracks( cur_tk, nb_tk_sptr))
        {

          auto lm_d_sptr = std::dynamic_pointer_cast<landmark_d>(nb_lm_it->second);
          if (lm_d_sptr)
          {
            lm_d_sptr->set_loc(lm_merged.loc());
          }
          else
          {
            auto lm_f_sptr = std::dynamic_pointer_cast<landmark_f>(nb_lm_it->second);
            if (lm_f_sptr)
            {
              lm_f_sptr->set_loc(lm_merged.loc().cast<float>());
            }
          }
          removed_landmarks.insert(cur_tk->id());
        }
      }

      // if all reprojection errors check out, merge the landmarks
      // lowest landmark id persists.  merge the tracks
      // set the lowest landmark's position to lm_merged
      // delete the higher id landmark
    }
  }

  LOG_WARN(m_logger, "merged " << removed_landmarks.size() << " landmarks");
  for (auto rm : removed_landmarks)
  {
    lmks.erase(rm);
  }

  // do I have any landmarks that don't have tracks?
  num_landmarks_without_tracks = 0;
  for (auto &lm : lmks)
  {
    if (!tracks->get_track(lm.first))
    {
      ++num_landmarks_without_tracks;
    }
  }

  if (num_landmarks_without_tracks)
  {
    LOG_WARN(m_logger, "end num landmarks without tracks "
                       << num_landmarks_without_tracks);
  }


}


bool
initialize_cameras_landmarks_keyframe::priv
::initialize_remaining_cameras(
  simple_camera_perspective_map_sptr cams,
  landmark_map_sptr& landmarks,
  feature_track_set_sptr tracks,
  sfm_constraints_sptr constraints,
  callback_t callback)
{
  // we will lock the original cameras in the bundle adjustment,
  // could also exclude them
  time_t prev_callback_time;
  time(&prev_callback_time);
  const double callback_min_period = 2;
  int lmks_last_down_select = 0;

  auto lmks = landmarks->landmarks();
  sfm_constraints_sptr constraints_to_ba = nullptr;

  std::set<frame_id_t> already_registred_cams,
                       frames_to_register,
                       windowed_bundled_cams,
                       all_frames_to_register,
                       keyframes_to_register,
                       non_keyframes_to_register;
  get_registered_and_non_registered_frames(cams, tracks,
                                           already_registred_cams,
                                           all_frames_to_register);

  // enforce registering only keyframes
  for (auto fid : all_frames_to_register)
  {
    if (m_keyframes.find(fid) != m_keyframes.end())
    {
      keyframes_to_register.insert(fid);
    }
    else
    {
      non_keyframes_to_register.insert(fid);
    }
  }

  if (keyframes_to_register.empty())
  {
    frames_to_register = non_keyframes_to_register;
  }
  else
  {
    frames_to_register = keyframes_to_register;
  }

  std::set<frame_id_t> frames_since_last_local_ba;

  auto existing_fids = cams->get_frame_ids();
  for (auto fid : existing_fids)
  {
    frames_since_last_local_ba.insert(fid);
  }

  m_frames_removed_from_sfm_solution.clear();

  int max_constraints_used = 0;

  bool done_registering_keyframes = false;
  bool disable_windowing = true;

  int frames_since_last_ba = 0;

  std::map<frame_id_t, double> last_reproj_by_cam;
  while(!frames_to_register.empty() &&
        this->continue_processing)
  {
    if (max_constraints_used < 10)
    {
      int num_constraints_used;
      if (fit_reconstruction_to_constraints(cams, lmks, tracks,
                                            constraints,
                                            num_constraints_used))
      {
        max_constraints_used =
          std::max(num_constraints_used, max_constraints_used);
        constraints_to_ba = constraints;
      }
      else
      {
        constraints_to_ba = nullptr;
      }
    }

    frame_id_t fid_to_register;

    if (!initialize_next_camera(cams, lmks, tracks, nullptr,
                                fid_to_register, frames_to_register,
                                already_registred_cams))
    {
      continue;
    }
    LOG_DEBUG(m_logger, "Resectioned frame " << fid_to_register);

    // Triangulate only landmarks visible in latest cameras
    std::set<frame_id_t> fids_to_triang;
    fids_to_triang.insert(fid_to_register);

    triangulate_landmarks_visible_in_frames(lmks, cams, tracks,
                                            fids_to_triang, false);

    if (lmks.size() > 1.4 * lmks_last_down_select || frames_to_register.empty())
    {
      // merging landmarks is very slow with limited benefit, so disable for now.
      // merge_landmarks(lmks, cams, tracks);
      down_select_landmarks(lmks, cams, tracks, fids_to_triang);
      lmks_last_down_select = static_cast<int>(lmks.size());
    }

    frames_since_last_local_ba.insert(fid_to_register);

    auto reporj_by_cam =
      kwiver::arrows::reprojection_rmse_by_cam(cams->cameras(),
                                               lmks, tracks->tracks());

    double rebundle_thresh = final_reproj_thresh * 4.0;
    bool bundle_because_of_reproj = false;
    int num_cams_over_thresh = 0;
    for (auto& cd : reporj_by_cam)
    {
      if (cd.second > rebundle_thresh)
      {
        auto last_it = last_reproj_by_cam.find(cd.first);
        if (last_it != last_reproj_by_cam.end())
        {
          if (last_it->second > rebundle_thresh)
          {
            // ignore this one, it didn't get better with bundling.
            continue;
          }
        }
        ++num_cams_over_thresh;
        if (num_cams_over_thresh >= 10)
        {
          bundle_because_of_reproj = true;
          break;
        }
      }
    }

    ++frames_since_last_ba;
    if (bundle_because_of_reproj ||
        frames_to_register.empty() ||
        frames_since_last_ba > 50)
    {
      frames_since_last_ba = 0;
      windowed_clean_and_bundle(cams, landmarks, lmks, tracks,
                                constraints_to_ba,
                                windowed_bundled_cams,
                                frames_since_last_local_ba);

      if (!disable_windowing)
      {
        for (auto ll_fid : frames_since_last_local_ba)
        {
          windowed_bundled_cams.insert(ll_fid);
        }
        frames_since_last_local_ba.clear();
      }

      last_reproj_by_cam =
        kwiver::arrows::reprojection_rmse_by_cam(cams->cameras(),
                                                 lmks, tracks->tracks());
    }

    already_registred_cams.insert(fid_to_register);

    time_t cur_time;
    time(&cur_time);
    double seconds_since_last_disp = difftime(cur_time, prev_callback_time);

    if (callback && seconds_since_last_disp > callback_min_period)
    {
      time(&prev_callback_time);
      auto chgs = get_feature_track_changes(tracks, *cams);
      continue_processing =
        callback(cams, std::make_shared<simple_landmark_map>(lmks),chgs);
    }
    if (!continue_processing)
    {
      LOG_DEBUG(m_logger, "continue processing is false, "
        "exiting initialize_remaining_cameras loop");
      break;
    }
    if ( frames_to_register.empty() &&
         !done_registering_keyframes &&
         !keyframes_to_register.empty())
    {
      done_registering_keyframes = true;
      disable_windowing = false;
      frames_to_register = non_keyframes_to_register;
      LOG_INFO(m_logger, "Finished processing key frames, "
                         "start filling intermediate frames ");
    }
  }

  if (m_do_final_sfm_cleaning)
  {
    std::set<frame_id_t> empty_frames;
    std::set<landmark_id_t> empty_landmarks;
    std::vector<frame_id_t> removed_cams;
    clean_cameras_and_landmarks(*cams, lmks, tracks,
                                m_thresh_triang_cos_ang, removed_cams,
                                empty_frames, empty_landmarks,
                                image_coverage_threshold,
                                interim_reproj_thresh, 3);

    for (auto rem_fid : removed_cams)
    {
      m_frames_removed_from_sfm_solution.insert(rem_fid);
    }
  }
  return true;
}

bool
initialize_cameras_landmarks_keyframe::priv
::bundle_adjust()
{
  return true;
}

void
initialize_cameras_landmarks_keyframe::priv
::init_base_camera_from_metadata(
  sfm_constraints_sptr constraints)
{
  if (!constraints)
  {
    return;
  }

  auto base_intrin = m_base_camera.get_intrinsics();

  float focal_length;

  if (constraints->get_focal_length_prior(-1, focal_length))
  {
    int im_h, im_w;

    auto pp = base_intrin->principal_point();
    if (constraints->get_image_height(-1, im_h) &&
        constraints->get_image_width(-1, im_w))
    {
      pp[0] = im_w*0.5;
      pp[1] = im_h*0.5;
    }

    auto intrin2 =
      std::make_shared<simple_camera_intrinsics>(focal_length,
                                                 pp,
                                                 base_intrin->aspect_ratio(),
                                                 base_intrin->skew());

    auto dist_coeffs = base_intrin->dist_coeffs();
    if (dist_coeffs.size() == 5)
    {
      Eigen::VectorXd dist;
      dist.resize(5);
      for (int i = 0; i < 5; ++i)
      {
        dist[i] = dist_coeffs[i];
      }
      intrin2->set_dist_coeffs(dist);
    }

    m_base_camera.set_intrinsics(intrin2);
  }
}

//-----------------------------------------------------------------------------
// start: initialize_cameras_landmarks_keyframe

/// Constructor
initialize_cameras_landmarks_keyframe
::initialize_cameras_landmarks_keyframe()
: m_priv(new priv)
{
}

/// Destructor
initialize_cameras_landmarks_keyframe
::~initialize_cameras_landmarks_keyframe()
{
}

/// Get this algorithm's \link vital::config_block configuration block \endlink
vital::config_block_sptr
initialize_cameras_landmarks_keyframe
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config =
      vital::algo::initialize_cameras_landmarks::get_configuration();

  const camera_intrinsics_sptr K = m_priv->m_base_camera.get_intrinsics();

  config->set_value("verbose", m_priv->verbose,
                    "If true, write status messages to the terminal showing "
                    "debugging information");

  config->set_value("force_common_intrinsics", m_priv->m_force_common_intrinsics,
                    "If true, then all cameras will share a single set of camera "
                    "intrinsic parameters");

  config->set_value("frac_frames_for_init", m_priv->m_frac_frames_for_init,
                    "fraction of keyframes used in relative pose initialization");

  config->set_value("interim_reproj_thresh", m_priv->interim_reproj_thresh,
                    "Threshold for rejecting landmarks based on reprojection "
                    "error (in pixels) during intermediate processing steps.");

  config->set_value("final_reproj_thresh", m_priv->final_reproj_thresh,
                    "Relative threshold for rejecting landmarks based on "
                    "reprojection error relative to the median error after "
                    "the final bundle adjustment.  For example, a value of 2 "
                    "mean twice the median error");

  config->set_value("zoom_scale_thresh", m_priv->zoom_scale_thresh,
                    "Threshold on image scale change used to detect a camera "
                    "zoom. If the resolution on target changes by more than "
                    "this fraction create a new camera intrinsics model.");

  config->set_value("base_camera:focal_length", K->focal_length(),
                    "focal length of the base camera model");

  config->set_value("base_camera:principal_point", K->principal_point().transpose(),
                    "The principal point of the base camera model \"x y\".\n"
                    "It is usually safe to assume this is the center of the "
                    "image.");

  config->set_value("base_camera:aspect_ratio", K->aspect_ratio(),
                    "the pixel aspect ratio of the base camera model");

  config->set_value("base_camera:skew", K->skew(),
                    "The skew factor of the base camera model.\n"
                    "This is almost always zero in any real camera.");

  config->set_value("max_cams_in_keyframe_init", m_priv->m_max_cams_in_keyframe_init,
                    "the maximum number of cameras to reconstruct in "
                    "initialization step before switching to resectioning "
                    "remaining cameras.");

  config->set_value("metadata_init_permissive_triang_thresh",
                    m_priv->m_metadata_init_permissive_triang_thresh,
                    "threshold to apply to triangulation in the first "
                    "permissive rounds of metadata based reconstruction "
                    "initialization");

  double ang_thresh_cur = acos(m_priv->m_thresh_triang_cos_ang) * rad_to_deg;
  config->set_value("feature_angle_threshold", ang_thresh_cur,
                    "feature must have this triangulation angle to keep");

  config->set_value("do_final_sfm_cleaning",
                    m_priv->m_do_final_sfm_cleaning,
                    "run a final sfm solution cleanup when solution is complete");

  double r1 = 0;
  double r2 = 0;
  double r3 = 0;
  auto dc = K->dist_coeffs();
  if (dc.size() == 5)
  {
    r1 = dc[0];
    r2 = dc[1];
    r3 = dc[4];
  }
  config->set_value("base_camera:r1", r1, "r^2 radial distortion term");
  config->set_value("base_camera:r2", r2, "r^4 radial distortion term");
  config->set_value("base_camera:r3", r3, "r^6 radial distortion term");

  config->set_value("init_intrinsics_from_metadata",
                    m_priv->m_init_intrinsics_from_metadata,
                    "initialize camera's focal length from metadata");

  // nested algorithm configurations
  vital::algo::estimate_essential_matrix
      ::get_nested_algo_configuration("essential_mat_estimator",
                                      config, m_priv->e_estimator);
  vital::algo::optimize_cameras
      ::get_nested_algo_configuration("camera_optimizer",
                                      config, m_priv->camera_optimizer);
  vital::algo::triangulate_landmarks
      ::get_nested_algo_configuration("lm_triangulator",
                                      config, m_priv->lm_triangulator);
  vital::algo::bundle_adjust
      ::get_nested_algo_configuration("bundle_adjuster",
                                      config, m_priv->bundle_adjuster);
  vital::algo::bundle_adjust
      ::get_nested_algo_configuration("global_bundle_adjuster",
                                      config, m_priv->global_bundle_adjuster);
  vital::algo::estimate_pnp
    ::get_nested_algo_configuration("estimate_pnp", config, m_priv->m_pnp);

  vital::algo::estimate_canonical_transform
    ::get_nested_algo_configuration("canonical_estimator", config,
                                    m_priv->m_canonical_estimator);

  vital::algo::estimate_similarity_transform
    ::get_nested_algo_configuration("similarity_estimator", config,
                                    m_priv->m_similarity_estimator);

  return config;
}


/// Set this algorithm's properties via a config block
void
initialize_cameras_landmarks_keyframe
::set_configuration(vital::config_block_sptr config)
{
  const camera_intrinsics_sptr K = m_priv->m_base_camera.get_intrinsics();

  // Set nested algorithm configurations
  vital::algo::estimate_essential_matrix
      ::set_nested_algo_configuration("essential_mat_estimator",
                                      config, m_priv->e_estimator);
  vital::algo::optimize_cameras
      ::set_nested_algo_configuration("camera_optimizer",
                                      config, m_priv->camera_optimizer);
  vital::algo::triangulate_landmarks
      ::set_nested_algo_configuration("lm_triangulator",
                                      config, m_priv->lm_triangulator);
  vital::algo::bundle_adjust
      ::set_nested_algo_configuration("bundle_adjuster",
                                      config, m_priv->bundle_adjuster);
  vital::algo::bundle_adjust
      ::set_nested_algo_configuration("global_bundle_adjuster",
                                      config, m_priv->global_bundle_adjuster);

  // make sure the callback is applied to any new instances of
  // nested algorithms
  this->set_callback(this->m_callback);

  vital::algo::estimate_canonical_transform
    ::set_nested_algo_configuration("canonical_estimator",
                                    config, m_priv->m_canonical_estimator);

  vital::algo::estimate_similarity_transform
    ::set_nested_algo_configuration("similarity_estimator",
                                    config, m_priv->m_similarity_estimator);

  m_priv->m_frac_frames_for_init =
    config->get_value<double>("frac_frames_for_init",
                              m_priv->m_frac_frames_for_init);

  m_priv->verbose = config->get_value<bool>("verbose",
                                        m_priv->verbose);

  m_priv->m_force_common_intrinsics =
    config->get_value<bool>("force_common_intrinsics",
                            m_priv->m_force_common_intrinsics);

  m_priv->interim_reproj_thresh =
      config->get_value<double>("interim_reproj_thresh",
                                m_priv->interim_reproj_thresh);

  m_priv->final_reproj_thresh =
      config->get_value<double>("final_reproj_thresh",
                                m_priv->final_reproj_thresh);

  m_priv->m_max_cams_in_keyframe_init =
    config->get_value<int>("max_cams_in_keyframe_init",
      m_priv->m_max_cams_in_keyframe_init);

  m_priv->m_metadata_init_permissive_triang_thresh =
    config->get_value<double>("metadata_init_permissive_triang_thresh",
      m_priv->m_metadata_init_permissive_triang_thresh);

  m_priv->zoom_scale_thresh =
      config->get_value<double>("zoom_scale_thresh",
                                m_priv->zoom_scale_thresh);

  m_priv->m_init_intrinsics_from_metadata =
    config->get_value<bool>("init_intrinsics_from_metadata",
                            m_priv->m_init_intrinsics_from_metadata);

  m_priv->m_do_final_sfm_cleaning =
    config->get_value<bool>("do_final_sfm_cleaning",
                            m_priv->m_do_final_sfm_cleaning);


  vital::config_block_sptr bc = config->subblock("base_camera");

  m_priv->m_config_defines_base_intrinsics =
    bc->has_value("focal_length") ||
    bc->has_value("principal_point") ||
    bc->has_value("aspect_ratio") ||
    bc->has_value("skew");

  simple_camera_intrinsics K2(bc->get_value<double>("focal_length",
                                                    K->focal_length()),
                              bc->get_value<vector_2d>("principal_point",
                                                       K->principal_point()),
                              bc->get_value<double>("aspect_ratio",
                                                    K->aspect_ratio()),
                              bc->get_value<double>("skew",
                                                    K->skew()));
  double r1 = bc->get_value<double>("r1", 0);
  double r2 = bc->get_value<double>("r2", 0);
  double r3 = bc->get_value<double>("r3", 0);

  Eigen::VectorXd dist;
  dist.resize(5);
  dist.setZero();
  dist[0] = r1;
  dist[1] = r2;
  dist[4] = r3;
  K2.set_dist_coeffs(dist);

  m_priv->m_base_camera.set_intrinsics(K2.clone());

  double ang_thresh_cur = acos(m_priv->m_thresh_triang_cos_ang) * rad_to_deg;
  double ang_thresh_config = config->get_value("feature_angle_threshold",
                                               ang_thresh_cur);

  m_priv->m_thresh_triang_cos_ang = cos(deg_to_rad * ang_thresh_config);



  vital::algo::estimate_pnp::set_nested_algo_configuration(
    "estimate_pnp", config, m_priv->m_pnp);
}


/// Check that the algorithm's currently configuration is valid
bool
initialize_cameras_landmarks_keyframe
::check_configuration(vital::config_block_sptr config) const
{
  if (config->get_value<std::string>("camera_optimizer", "") != ""
      && !vital::algo::optimize_cameras
              ::check_nested_algo_configuration("camera_optimizer", config))
  {
    return false;
  }
  if (config->get_value<std::string>("bundle_adjuster", "") != ""
      && !vital::algo::bundle_adjust
              ::check_nested_algo_configuration("bundle_adjuster", config))
  {
    return false;
  }
  if (config->get_value<std::string>("global_bundle_adjuster", "") != ""
      && !vital::algo::bundle_adjust
         ::check_nested_algo_configuration("global_bundle_adjuster", config))
  {
    return false;
  }
  return
    vital::algo::estimate_essential_matrix
      ::check_nested_algo_configuration("essential_mat_estimator",config) &&
    vital::algo::triangulate_landmarks
      ::check_nested_algo_configuration("lm_triangulator", config) &&
    vital::algo::estimate_canonical_transform
      ::check_nested_algo_configuration("canonical_estimator", config) &&
    vital::algo::estimate_similarity_transform
      ::check_nested_algo_configuration("similarity_estimator", config);
}

/// Initialize the camera and landmark parameters given a set of tracks
void
initialize_cameras_landmarks_keyframe
::initialize(camera_map_sptr& cameras,
             landmark_map_sptr& landmarks,
             feature_track_set_sptr tracks,
             sfm_constraints_sptr constraints) const
{
  // Remove stationary tracks.
  // These tracks are likely on a heads-up display, dead pixel, lens dirt, etc.
  auto stationary_tracks = detect_stationary_tracks(tracks);
  for (auto tk : stationary_tracks)
  {
    tracks->remove(tk);
  }

  // Compute keyframes to use for SFM
  m_priv->m_keyframes = keyframes_for_sfm(tracks);

  m_priv->m_already_merged_landmarks.clear();
  m_priv->check_inputs(tracks);

  auto cams = std::make_shared<simple_camera_perspective_map>();
  cams->set_from_base_cams(cameras);

  if (m_priv->m_init_intrinsics_from_metadata)
  {
    m_priv->init_base_camera_from_metadata(constraints);
  }

  if (!m_priv->initialize_keyframes(cams, landmarks, tracks,
                                    constraints, this->m_callback))
  {
    return;
  }

  if (cams->size() == 0 ||
      !m_priv->initialize_remaining_cameras(cams, landmarks, tracks,
                                            constraints, this->m_callback))
  {
    return;
  }

  cameras = std::make_shared<simple_camera_map>(cams->cameras());
}


/// Set a callback function to report intermediate progress
void
initialize_cameras_landmarks_keyframe
::set_callback(callback_t cb)
{
  vital::algo::initialize_cameras_landmarks::set_callback(cb);
  // pass callback on to bundle adjusters if available
  if ((m_priv->bundle_adjuster || m_priv->global_bundle_adjuster) &&
      this->m_callback)
  {
    using std::placeholders::_1;
    using std::placeholders::_2;
    using std::placeholders::_3;
    callback_t pcb =
      std::bind(&initialize_cameras_landmarks_keyframe::priv
        ::pass_through_callback,
        m_priv.get(), this->m_callback, _1, _2, _3);
    if (m_priv->bundle_adjuster)
    {
      m_priv->bundle_adjuster->set_callback(pcb);
    }
    if (m_priv->global_bundle_adjuster)
    {
      m_priv->global_bundle_adjuster->set_callback(pcb);
    }
  }
}

} // end namespace core
} // end namespace arrows
} // end namespace kwiver
