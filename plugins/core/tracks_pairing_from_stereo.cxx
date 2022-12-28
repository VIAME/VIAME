#include <vital/types/timestamp.h>
#include "tracks_pairing_from_stereo.h"

void viame::core::tracks_pairing_from_stereo::load_camera_calibration() {
  try {
    auto intrinsics_path = m_cameras_directory + "/intrinsics.yml";
    auto extrinsics_path = m_cameras_directory + "/extrinsics.yml";
    auto fs = cv::FileStorage(intrinsics_path, cv::FileStorage::Mode::READ);

    // load left camera intrinsic parameters
    if (!fs.isOpened())
      VITAL_THROW(kwiver::vital::file_not_found_exception, intrinsics_path, "could not locate file in path");

    fs["M1"] >> m_K1;
    fs["D1"] >> m_D1;
    fs["M2"] >> m_K2;
    fs["D2"] >> m_D2;
    fs.release();

    // load extrinsic parameters
    fs = cv::FileStorage(extrinsics_path, cv::FileStorage::Mode::READ);
    if (!fs.isOpened())
      VITAL_THROW(kwiver::vital::file_not_found_exception, extrinsics_path, "could not locate file in path");

    fs["Q"] >> m_Q;
    fs["R1"] >> m_R1;
    fs["P1"] >> m_P1;
    fs["R2"] >> m_R2;
    fs["P2"] >> m_P2;
    fs["R"] >> m_R;
    fs["T"] >> m_T;

    // Compute RVec from matrix for later use
    cv::Rodrigues(m_R, m_Rvec);

    fs.release();
  } catch (const kwiver::vital::file_not_found_exception &e) {
    VITAL_THROW(kwiver::vital::invalid_data, "Calibration file not found : " + std::string(e.what()));
  }
}

float viame::core::tracks_pairing_from_stereo::compute_median(std::vector<float> values, bool is_sorted) {
  float median = 0;
  size_t size = values.size();
  if (size > 0) {
    if (!is_sorted)
      std::sort(values.begin(), values.end());

    if (size % 2 == 0)
      median = (values[size / 2 - 1] + values[size / 2]) / 2;
    else
      median = values[size / 2];
  }
  return median;
}

cv::Rect viame::core::tracks_pairing_from_stereo::bbox_to_mask_rect(const kwiver::vital::bounding_box_d &bbox) {
  return {cv::Point2d{bbox.upper_left().x(), bbox.upper_left().y()},
          cv::Point2d{bbox.lower_right().x(), bbox.lower_right().y()}};
}


kwiver::vital::bounding_box_d viame::core::tracks_pairing_from_stereo::mask_rect_to_bbox(const cv::Rect &rect) {
  return {{rect.tl().x, rect.tl().y},
          {rect.br().x, rect.br().y}};
}


std::tuple<cv::Mat, cv::Rect>
viame::core::tracks_pairing_from_stereo::get_standard_mask(const kwiver::vital::detected_object_sptr &det) {
  auto vital_mask = det->mask();
  if (!vital_mask) {
    return {};
  }
  using ic = kwiver::arrows::ocv::image_container;
  cv::Mat mask = ic::vital_to_ocv(vital_mask->get_image(), ic::OTHER_COLOR);
  auto size = bbox_to_mask_rect(det->bounding_box()).size();
  cv::Rect intersection(0, 0, std::min(size.width, mask.cols), std::min(size.height, mask.rows));

  if (mask.size() == size) {
    return {mask, intersection};
  }
  cv::Mat standard_mask(size, CV_8UC1, cv::Scalar(0));
  mask(intersection).copyTo(standard_mask(intersection));
  return {standard_mask, intersection};
}

inline void print(const kwiver::vital::bounding_box_d &bbox, const std::string &context = "") {
  if (!bbox.is_valid()) {
    std::cout << context << " - kv::bounding_box_d( INVALID )" << std::endl;
    return;
  }

  std::cout << context << " - kv::bounding_box_d( upperLeft {" << bbox.upper_left().x() << ", " << bbox.upper_left().y()
            << "}, lowerRight {" << bbox.lower_right().x() << ", " << bbox.lower_right().y() << "})" << std::endl;
}

inline void print(const Eigen::Matrix<double, 2, 1> &mat, const std::string &context) {
  std::cout << context << " {" << mat.x() << ", " << mat.y() << "}" << std::endl;
}


inline void print(const cv::Vec3f &mat, const std::string &context) {
  std::cout << context << " {" << mat[0] << ", " << mat[1] << ", " << mat[2] << "}" << std::endl;
}

inline void print(const cv::Vec2f &mat, const std::string &context) {
  std::cout << context << " {" << mat[0] << ", " << mat[1] << "}" << std::endl;
}

inline void print(const cv::Rect &bbox, const std::string &context = "") {
  std::cout << context << " - cv::Rect( upperLeft {" << bbox.tl().x << ", " << bbox.tl().y << "}, lowerRight {"
            << bbox.br().x << ", " << bbox.br().y << "})" << std::endl;
}


viame::core::Tracks3DPositions viame::core::tracks_pairing_from_stereo::estimate_3d_position_from_detection(
    const kwiver::vital::detected_object_sptr &detection, const cv::Mat &pos_3d_map) const {
  // Extract mask and corresponding mask bounding box from input detection
  auto mask_and_bbox = get_standard_mask(detection);
  auto mask = std::get<0>(mask_and_bbox);
  auto mask_bbox = std::get<1>(mask_and_bbox);

  // If mask is invalid return the estimated position from bounding box center
  if (mask.empty())
    return estimate_3d_position_from_bbox(detection->bounding_box(), pos_3d_map);

  // Otherwise, returns average 3D distance for each point in mask
  return estimate_3d_position_from_mask(mask_bbox, pos_3d_map, mask);
}

kwiver::vital::bounding_box_d
viame::core::tracks_pairing_from_stereo::get_rectified_bbox(const kwiver::vital::bounding_box_d &bbox,
                                                            bool is_left_image) const {
  const auto tl = undistort_point({bbox.upper_left().x(), bbox.upper_left().y()}, is_left_image);
  const auto br = undistort_point({bbox.lower_right().x(), bbox.lower_right().y()}, is_left_image);
  return {tl.x, tl.y, br.x, br.y};
}

cv::Rect viame::core::tracks_pairing_from_stereo::get_rectified_bbox(const cv::Rect &bbox) const {
  return bbox_to_mask_rect(get_rectified_bbox(mask_rect_to_bbox(bbox), true));
}

bool viame::core::tracks_pairing_from_stereo::point_is_valid(float x, float y, float z) {
  return ((z > 0) && std::isfinite(x) && std::isfinite(y) && std::isfinite(z));
}

bool viame::core::tracks_pairing_from_stereo::point_is_valid(const cv::Vec3f &pt) {
  return point_is_valid(pt[0], pt[1], pt[2]);
}

viame::core::Tracks3DPositions
viame::core::tracks_pairing_from_stereo::estimate_3d_position_from_bbox(const kwiver::vital::bounding_box_d &bbox,
                                                                        const cv::Mat &pos_3d_map) const {
  const auto rectified_bbox = get_rectified_bbox(bbox, true);

  // depth from median of values in the center part of the bounding box
  float ratio = 1.f / 3.f;
  float crop_width = ratio * (float) rectified_bbox.width();
  float crop_height = ratio * (float) rectified_bbox.height();
  cv::Rect crop_rect{(int) (rectified_bbox.center().x() - crop_width / 2),
                     (int) (rectified_bbox.center().y() - crop_height / 2), (int) crop_width, (int) crop_height};

  // Intersect crop rectangle with 3D map rect to avoid out of bounds crop
  crop_rect = crop_rect & cv::Rect(0, 0, pos_3d_map.size().width, pos_3d_map.size().height);
  print(crop_rect, "CROP RECT");

  // If resized crop is out of the 3D map (detection out of left / right ROI overlap) return 0
  if (crop_rect.width == 0 || crop_rect.height == 0) {
    return {};
  }

  // Compute medians in cropped patch
  cv::Mat channels[3];
  auto crop = pos_3d_map(crop_rect);
  cv::split(crop, channels);

  // Select for valid points (with z > 0 and z != inf ) and compute xs, ys, zs
  // median from those
  cv::Mat xs = channels[0].reshape(1, 1);
  cv::Mat ys = channels[1].reshape(1, 1);
  cv::Mat zs = channels[2].reshape(1, 1);
  int nb_val = zs.size().width;
  if (zs.depth() != CV_32F) {
    throw std::runtime_error("depth values are not of type cv::CV_32F.");
  }

  std::vector<float> valid_xs, valid_ys, valid_zs;
  for (int i = 0; i < nb_val; i++) {
    auto x = xs.at<float>(i);
    auto y = ys.at<float>(i);
    auto z = zs.at<float>(i);

    if (point_is_valid(x, y, z)) {
      valid_xs.push_back(x);
      valid_ys.push_back(y);
      valid_zs.push_back(z);
    }
  }

  // Return 3d position
  auto score = (float) valid_xs.size() / (crop_width * crop_height);
  return create_3d_position(valid_xs, valid_ys, valid_zs, rectified_bbox, pos_3d_map, score);
}

viame::core::Tracks3DPositions
viame::core::tracks_pairing_from_stereo::estimate_3d_position_from_mask(const cv::Rect &bbox, const cv::Mat &pos_3d_map,
                                                                        const cv::Mat &mask) const {
  // Early return if bbox crop is out of the 3D map
  if (bbox.width == 0 || bbox.height == 0)
    return {};

  // Find all distorted positions where mask is not empty
  std::vector<cv::Point2d> mask_distorted_coords;
  const auto mask_tl = bbox.tl();
  for (int i_x = 0; i_x < mask.size().width; i_x++) {
    for (int i_y = 0; i_y < mask.size().height; i_y++) {
      if (mask.at<int>(i_y, i_x) > 0) {
        mask_distorted_coords.emplace_back(cv::Point2d(mask_tl.x + i_x, mask_tl.y + i_y));
      }
    }
  }

  // If no segmentation, early return
  if (mask_distorted_coords.empty())
    return {};

  // Undistort mask points
  auto undistorted_mask_coords = undistort_point(mask_distorted_coords, true);

  // For each undistorted point coordinates, find 3D position corresponding to undistorted point
  const auto is_out_of_bounds = [&pos_3d_map](const cv::Point2d &pt) {
    return (pt.x < 0.) || (pt.y < 0.) || (pt.x >= pos_3d_map.size().width) || (pt.y >= pos_3d_map.size().height);
  };

  int n_total{};
  std::vector<float> xs, ys, zs;
  for (const auto &point: undistorted_mask_coords) {
    if (is_out_of_bounds(point))
      continue;

    // WARNING: cv::Mat::at<> Expects i_row and i_col as input. This is inverted with BBox x, y coord
    n_total += 1;
    auto point_3d = pos_3d_map.at<cv::Vec3f>((int) point.y, (int) point.x);

    if (point_is_valid(point_3d)) {
      xs.push_back(point_3d[0]);
      ys.push_back(point_3d[1]);
      zs.push_back(point_3d[2]);
    }
  }

  // Return score based on number of valid position pixels vs number of mask pixels
  auto score = n_total > 0 ? ((float) xs.size() / (float) n_total) : 0.f;
  return create_3d_position(xs, ys, zs, bounding_box_cv_to_kv(get_rectified_bbox(bbox)), pos_3d_map, score);
}

kwiver::vital::bounding_box_d viame::core::tracks_pairing_from_stereo::bounding_box_cv_to_kv(const cv::Rect &bbox) {
  return {(double) bbox.tl().x, (double) bbox.tl().y, (double) bbox.br().x, (double) bbox.br().y};
}

viame::core::Tracks3DPositions
viame::core::tracks_pairing_from_stereo::create_3d_position(const std::vector<float> &xs, const std::vector<float> &ys,
                                                            const std::vector<float> &zs,
                                                            const kwiver::vital::bounding_box_d &bbox,
                                                            const cv::Mat &pos_3d_map, float score) const {
  const auto saturate_corner = [&pos_3d_map](const Eigen::Matrix<double, 2, 1> &corner) {
    return Eigen::Matrix<double, 2, 1>{std::max(0., std::min(pos_3d_map.size().width - 1., corner.x())),
                                       std::max(0., std::min(pos_3d_map.size().height - 1., corner.y()))};
  };

  const auto saturate_bbox = [&saturate_corner](const kwiver::vital::bounding_box_d &bbox) {
    if (!bbox.is_valid())
      return bbox;

    return kwiver::vital::bounding_box_d{saturate_corner(bbox.upper_left()), saturate_corner(bbox.lower_right())};
  };

  const auto extract_3d_bbox_from_point_list = [&] {
    // Find bounding box of input
    cv::Vec3f tl_3d{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::max()};
    cv::Vec3f br_3d{std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                    std::numeric_limits<float>::lowest()};

    for (size_t i_pt = 0; i_pt < xs.size(); i_pt++) {
      tl_3d[0] = std::min(tl_3d[0], xs[i_pt]);
      tl_3d[1] = std::min(tl_3d[1], ys[i_pt]);
      tl_3d[2] = std::min(tl_3d[2], zs[i_pt]);

      br_3d[0] = std::max(br_3d[0], xs[i_pt]);
      br_3d[1] = std::max(br_3d[1], ys[i_pt]);
      br_3d[2] = std::max(br_3d[2], zs[i_pt]);
    }
    return std::vector<cv::Vec3f>{tl_3d, br_3d};
  };

  Tracks3DPositions position;
  position.score = score;
  position.center3d = cv::Point3f{compute_median(xs), compute_median(ys), compute_median(zs)};
  position.rectified_left_bbox = saturate_bbox(bbox);
  position.left_bbox_proj_to_right_image = saturate_bbox(xs.empty() ?//
                                                         project_to_right_image(bbox, pos_3d_map) ://
                                                         project_to_right_image(extract_3d_bbox_from_point_list()));

  // If position score is valid, project center3D to right image
  // Otherwise, keep 0, 0 value
  if (position.is_valid())
    position.center3d_proj_to_right_image = project_to_right_image(position.center3d);

  // Debug print
  print(position.center3d, "CENTER");
  print(position.center3d_proj_to_right_image, "PROJ CENTER TO RIGHT");
  print(position.rectified_left_bbox, "RECTIFIED LEFT BBOX");
  print(position.left_bbox_proj_to_right_image, "PROJ LEFT BBOX");

  return position;
}

kwiver::vital::bounding_box_d
viame::core::tracks_pairing_from_stereo::project_to_right_image(const kwiver::vital::bounding_box_d &bbox,
                                                                const cv::Mat &pos_3d_map) const {
  auto saturate_pos = [&pos_3d_map](const Eigen::Matrix<double, 2, 1> &corner) {
    auto x = std::min(std::max(corner.x(), 0.), pos_3d_map.size().width - 1.);
    auto y = std::min(std::max(corner.y(), 0.), pos_3d_map.size().height - 1.);

    return Eigen::Matrix<double, 2, 1>{x, y};
  };

  // Saturate upper left and lower right coordinates to image coordinates
  auto bbox_ul = saturate_pos(bbox.upper_left());
  auto bbox_lr = saturate_pos(bbox.lower_right());

  // Find 3D points associated with input bounding box
  // WARNING: cv::Mat::at<> Expects i_row and i_col as input. This is inverted with BBox x, y coord
  auto tl_3d = pos_3d_map.at<cv::Vec3f>((int) bbox_ul.y(), (int) bbox_ul.x());
  auto br_3d = pos_3d_map.at<cv::Vec3f>((int) bbox_lr.y(), (int) bbox_lr.x());

  if (!point_is_valid(tl_3d) || !point_is_valid(br_3d))
    return {};

  // Project points to right camera coordinates
  return project_to_right_image(std::vector<cv::Vec3f>{tl_3d, br_3d});
}

kwiver::vital::bounding_box_d
viame::core::tracks_pairing_from_stereo::project_to_right_image(const std::vector<cv::Vec3f> &points_3d) const {
  // Sanity check on input vect list
  if (points_3d.size() != 2)
    VITAL_THROW(kwiver::vital::invalid_data,
                "Wrong input 3D point number. Expected 2, got : " + std::to_string(points_3d.size()));

  // Project points to right camera coordinates
  std::vector<cv::Point2f> projectedPoints;
  cv::projectPoints(points_3d, m_Rvec, m_T, m_K2, m_D2, projectedPoints);
  return {projectedPoints[0].x, projectedPoints[0].y, projectedPoints[1].x, projectedPoints[1].y};
}

cv::Point2f viame::core::tracks_pairing_from_stereo::project_to_right_image(const cv::Point3f &points_3d) const {
  std::vector<cv::Point2f> projectedPoints;
  cv::projectPoints(std::vector<cv::Point3f>{points_3d}, m_Rvec, m_T, m_K2, m_D2, projectedPoints);
  return projectedPoints[0];
}


std::tuple<std::vector<kwiver::vital::track_sptr>, std::vector<viame::core::Tracks3DPositions>>
viame::core::tracks_pairing_from_stereo::update_left_tracks_3d_position(
    const std::vector<kwiver::vital::track_sptr> &tracks, const cv::Mat &cv_disparity_map,
    const kwiver::vital::timestamp &timestamp) {
  const auto cv_pos_3d_map = reproject_3d_depth_map(cv_disparity_map);

  std::vector<kwiver::vital::track_sptr> filtered_tracks;
  std::vector<Tracks3DPositions> tracks_positions;

  for (const auto &track: tracks) {
    // Check if a 3d track already exists with this id in order to update it
    // instead of generating a new track
    kwiver::vital::track_sptr tracks_3d;
    auto t_ptr = m_tracks_with_3d_left.find(track->id());
    if (t_ptr != m_tracks_with_3d_left.end()) {
      tracks_3d = t_ptr->second;
    } else {
      tracks_3d = track;
      m_tracks_with_3d_left[track->id()] = tracks_3d;
    }

    // Add 3D track to output
    filtered_tracks.push_back(tracks_3d);

    // Skip 3d processing for tracks without current frame and associate empty position to avoid further pairing
    auto state = std::dynamic_pointer_cast<kwiver::vital::object_track_state>(track->back());
    if ((track->last_frame() < timestamp.get_frame()) || !state) {
      tracks_positions.emplace_back(Tracks3DPositions{});
      continue;
    }

    // Process 3D coordinates for frame matching the current depth image
    auto position = estimate_3d_position_from_detection(state->detection(), cv_pos_3d_map);

    // Add 3d estimations to state if score is valid
    if (position.score > 0) {
      state->detection()->add_note(":x=" + std::to_string(position.center3d.x));
      state->detection()->add_note(":y=" + std::to_string(position.center3d.y));
      state->detection()->add_note(":z=" + std::to_string(position.center3d.z));
      state->detection()->add_note(":score=" + std::to_string(position.score));
    }

    // Update state information to tracks 3D and push tracks to output
    tracks_3d->append(state);
    tracks_positions.emplace_back(position);
  }

  // Sanity check on the filtered_tracks and track_position size
  if (filtered_tracks.size() != tracks_positions.size())
    VITAL_THROW(kwiver::vital::invalid_data,
                "Expected same tracks and position size. Got : " + std::to_string(filtered_tracks.size()) + " VS " +
                std::to_string(tracks_positions.size()));

  return {filtered_tracks, tracks_positions};
}

std::vector<kwiver::vital::track_sptr> viame::core::tracks_pairing_from_stereo::keep_right_tracks_in_current_frame(
    const std::vector<kwiver::vital::track_sptr> &tracks, const kwiver::vital::timestamp &timestamp) {
  std::vector<kwiver::vital::track_sptr> filtered_tracks;

  for (const auto &track: tracks) {
    // Get corresponding track in right track map or add track to map if not present yet
    if (m_right_tracks_memo.find(track->id()) == std::end(m_right_tracks_memo))
      m_right_tracks_memo[track->id()] = track;

    // Get track from dict (track present in dict contains updated ID
    auto paired_track = m_right_tracks_memo[track->id()];
    filtered_tracks.push_back(paired_track);

    // Skip tracks without state or which frames don't match current frame
    auto state = std::dynamic_pointer_cast<kwiver::vital::object_track_state>(track->back());
    if (((track->last_frame() < timestamp.get_frame()) || !state)) {
      continue;
    }

    // Update state in paired_track
    paired_track->append(state);
  }

  return filtered_tracks;
}

double viame::core::tracks_pairing_from_stereo::iou_distance(const kwiver::vital::bounding_box_d &bbox1,
                                                             const kwiver::vital::bounding_box_d &bbox2) {
  Eigen::AlignedBox<double, 2> bbox1_eig{bbox1.upper_left(), bbox1.lower_right()};
  Eigen::AlignedBox<double, 2> bbox2_eig{bbox2.upper_left(), bbox2.lower_right()};

  // Early return if the input bounding boxes are invalid or don't intersect
  if (!bbox1.is_valid() || !bbox2.is_valid() || !bbox1_eig.intersects(bbox2_eig))
    return 0;

  auto bbox_intersection = bbox1_eig.intersection(bbox2_eig).volume();
  auto bbox_union = bbox1_eig.volume() + bbox2_eig.volume() - bbox_intersection;
  return bbox_intersection / bbox_union;
}


kwiver::vital::track_id_t viame::core::tracks_pairing_from_stereo::last_left_right_track_id() const {
  auto get_map_ids = [](const std::map<kwiver::vital::track_id_t, kwiver::vital::track_sptr> &map) {
    std::set<kwiver::vital::track_id_t> track_ids;
    for (const auto &track: map) {
      track_ids.emplace(track.second->id());
    }
    return track_ids;
  };

  auto ids_left = get_map_ids(m_tracks_with_3d_left);
  auto ids_right = get_map_ids(m_right_tracks_memo);
  if (ids_left.empty() || ids_right.empty())
    return ids_left.empty() ? ids_right.empty() ? 1 : *ids_right.rbegin() + 1 : *ids_left.rbegin() + 1;

  return std::max(*ids_left.rbegin(), *ids_right.rbegin());
}


cv::Mat viame::core::tracks_pairing_from_stereo::reproject_3d_depth_map(const cv::Mat &cv_disparity_left) const {
  cv::Mat cv_pos_3d_left_map;
  cv::reprojectImageTo3D(cv_disparity_left, cv_pos_3d_left_map, m_Q, false);
  return cv_pos_3d_left_map;
}


cv::Point2d
viame::core::tracks_pairing_from_stereo::undistort_point(const cv::Point2d &point, bool is_left_image) const {
  return undistort_point(std::vector<cv::Point2d>{point}, is_left_image)[0];
}


std::vector<cv::Point2d> viame::core::tracks_pairing_from_stereo::undistort_point(const std::vector<cv::Point2d> &point,
                                                                                  bool is_left_image) const {
  std::vector<cv::Point2d> points_undist;

  if (is_left_image)
    cv::undistortPoints(std::vector<cv::Point2d>{point}, points_undist, m_K1, m_D1, m_R1, m_P1);
  else
    cv::undistortPoints(std::vector<cv::Point2d>{point}, points_undist, m_K2, m_D2, m_R2, m_P2);
  return points_undist;
}


/// @brief Helper structure to store the most likely pair to a left track
struct MostLikelyPair {
  int frame_count{-1};
  kwiver::vital::track_id_t right_id{-1};
};


std::tuple<std::vector<kwiver::vital::track_sptr>, std::vector<kwiver::vital::track_sptr>>
viame::core::tracks_pairing_from_stereo::get_left_right_tracks_with_pairing() {
  std::vector<kwiver::vital::track_sptr> left_tracks, right_tracks;

  auto proc_tracks = m_do_split_detections ? split_paired_tracks_to_new_tracks(left_tracks, right_tracks)
                                           : select_most_likely_pairing(left_tracks, right_tracks);

  // Append other tracks
  const auto append_unprocessed_tracks = [](
      const std::map<kwiver::vital::track_id_t, kwiver::vital::track_sptr> &tracks_map,
      std::set<kwiver::vital::track_id_t> &processed, std::vector<kwiver::vital::track_sptr> &vect) {
    for (const auto &pair: tracks_map) {
      if (processed.find(pair.first) != std::end(processed))
        continue;

      processed.emplace(pair.first);
      vect.emplace_back(pair.second->clone());
    }
  };

  append_unprocessed_tracks(m_tracks_with_3d_left, std::get<0>(proc_tracks), left_tracks);
  append_unprocessed_tracks(m_right_tracks_memo, std::get<1>(proc_tracks), right_tracks);

  return {filter_tracks_with_threshold(left_tracks), filter_tracks_with_threshold(right_tracks)};
}

std::tuple<std::set<kwiver::vital::track_id_t>, std::set<kwiver::vital::track_id_t>>
viame::core::tracks_pairing_from_stereo::select_most_likely_pairing(std::vector<kwiver::vital::track_sptr> &left_tracks,
                                                                    std::vector<kwiver::vital::track_sptr> &right_tracks) {
  std::map<kwiver::vital::track_id_t, MostLikelyPair> most_likely_left_to_right_pair;
  // Find the most likely pair from all pairs across all frames
  for (const auto &pair: m_left_to_right_pairing) {
    const auto id_pair = pair.second.left_right_id_pair;
    if (most_likely_left_to_right_pair.find(id_pair.left_id) == std::end(most_likely_left_to_right_pair))
      most_likely_left_to_right_pair[id_pair.left_id] = MostLikelyPair{};

    const auto pair_frame_count = (int) pair.second.frame_set.size();
    if (pair_frame_count > most_likely_left_to_right_pair[id_pair.left_id].frame_count) {
      most_likely_left_to_right_pair[id_pair.left_id].frame_count = (int) pair_frame_count;
      most_likely_left_to_right_pair[id_pair.left_id].right_id = id_pair.right_id;
    }
  }

  // Iterate over all paired tracks first
  std::set<kwiver::vital::track_id_t> proc_left, proc_right;
  auto last_track_id = last_left_right_track_id() + 1;
  for (const auto &pair: most_likely_left_to_right_pair) {
    const auto right_id = pair.second.right_id;
    const auto left_id = pair.first;
    if (proc_right.find(right_id) != std::end(proc_right)) {
      std::cout << "RIGHT TRACK ALREADY PAIRED " << left_id << ", " << right_id << std::endl;
      continue;
    }

    proc_left.emplace(left_id);
    proc_right.emplace(right_id);

    auto left_track = m_tracks_with_3d_left[left_id]->clone();
    auto right_track = m_right_tracks_memo[right_id]->clone();

    // Update left right pairing in case the pairing id is different.
    // Otherwise, keep the track id
    if (left_id != right_id) {
      std::cout << "PAIRING " << left_id << ", " << right_id << " TO " << last_track_id << std::endl;
      left_track->set_id(last_track_id);
      right_track->set_id(last_track_id);
      last_track_id++;
    } else {
      std::cout << "KEEPING PAIRING " << left_id << ", " << right_id << std::endl;
    }
    left_tracks.emplace_back(left_track);
    right_tracks.emplace_back(right_track);
  }

  return {proc_left, proc_right};
}

std::tuple<std::set<kwiver::vital::track_id_t>, std::set<kwiver::vital::track_id_t>>
viame::core::tracks_pairing_from_stereo::split_paired_tracks_to_new_tracks(
    std::vector<kwiver::vital::track_sptr> &left_tracks, std::vector<kwiver::vital::track_sptr> &right_tracks) {
  const auto ranges = create_split_ranges_from_track_pairs(m_left_to_right_pairing);
  return split_ranges_to_tracks(ranges, left_tracks, right_tracks);
}


void viame::core::tracks_pairing_from_stereo::append_paired_frame(const kwiver::vital::track_sptr &left_track,
                                                                  const kwiver::vital::track_sptr &right_track,
                                                                  const kwiver::vital::timestamp &timestamp) {
  // Pair left to right for given frame
  auto pairing = cantor_pairing(left_track->id(), right_track->id());
  std::cout << "PAIRING (" << pairing << "): " << left_track->id() << ", " << right_track->id() << std::endl;

  if (m_left_to_right_pairing.find(pairing) == std::end(m_left_to_right_pairing))
    m_left_to_right_pairing[pairing] = Pairing{{},
                                               {left_track->id(), right_track->id()}};

  m_left_to_right_pairing[pairing].frame_set.emplace(timestamp.get_frame());
}


void viame::core::tracks_pairing_from_stereo::pair_left_right_tracks_using_3d_center(
    const std::vector<kwiver::vital::track_sptr> &left_tracks,
    const std::vector<viame::core::Tracks3DPositions> &left_3d_pos,
    const std::vector<kwiver::vital::track_sptr> &right_tracks, const kwiver::vital::timestamp &timestamp) {
  const auto most_probable_right_track = [&right_tracks, &timestamp](const Eigen::Matrix<double, 2, 1> &left_point,
                                                                     const std::string &left_class) {
    int i_best = -1;
    auto dist_best = std::numeric_limits<double>::max();

    for (size_t i_right = 0; i_right < right_tracks.size(); i_right++) {
      // Skip right tracks not in current frame or with different detection class
      const auto &right_track = right_tracks[i_right];
      if (right_track->back()->frame() < timestamp.get_frame() ||
          most_likely_detection_class(right_track) != left_class)
        continue;

      auto right_bbox = get_last_detection_bbox(right_track);
      if (!right_bbox.is_valid() || !right_bbox.contains(left_point))
        continue;

      const auto dist = (right_bbox.center() - left_point).norm();
      if (dist < dist_best) {
        i_best = (int) i_right;
        dist_best = dist;
      }
    }
    return i_best;
  };

  for (size_t i_left = 0; i_left < left_tracks.size(); i_left++) {
    const auto &left_track = left_tracks[i_left];
    const auto left_track_class = most_likely_detection_class(left_track);

    // Skip left tracks not in current frame or invalid
    if (left_track->back()->frame() < timestamp.get_frame() || !left_3d_pos[i_left].is_valid())
      continue;

    // Find most probable right track match given projected center point
    const auto proj_left_point = left_3d_pos[i_left].center3d_proj_to_right_image;
    const auto i_right = most_probable_right_track({proj_left_point.x, proj_left_point.y}, left_track_class);
    if (i_right < 0)
      continue;

    append_paired_frame(left_track, right_tracks[i_right], timestamp);
  }
}

kwiver::vital::bounding_box_d
viame::core::tracks_pairing_from_stereo::get_last_detection_bbox(const kwiver::vital::track_sptr &track) {
  const auto track_state = std::dynamic_pointer_cast<kwiver::vital::object_track_state>(track->back());
  if (!track_state)
    return {};

  if (!track_state->detection())
    return {};

  return track_state->detection()->bounding_box();
}


void viame::core::tracks_pairing_from_stereo::pair_left_right_tracks_using_bbox_iou(
    const std::vector<kwiver::vital::track_sptr> &left_tracks,
    const std::vector<kwiver::vital::track_sptr> &right_tracks, const kwiver::vital::timestamp &timestamp,
    bool do_rectify_bbox) {

  const auto most_probable_right_track = [&right_tracks, &timestamp, do_rectify_bbox, this](
      const kwiver::vital::track_sptr &left_track, const std::string &left_class) {
    int i_best = -1;
    auto best_iou = std::numeric_limits<double>::lowest();
    auto left_bbox = get_last_detection_bbox(left_track);
    if (do_rectify_bbox)
      left_bbox = get_rectified_bbox(left_bbox, true);

    if (!left_bbox.is_valid())
      return i_best;

    for (size_t i_right = 0; i_right < right_tracks.size(); i_right++) {
      // Skip right tracks not in current frame or with different detection class
      const auto &right_track = right_tracks[i_right];
      if (right_track->back()->frame() < timestamp.get_frame() ||
          most_likely_detection_class(right_track) != left_class)
        continue;

      auto right_bbox = get_last_detection_bbox(right_track);
      if (!right_bbox.is_valid())
        continue;

      if (do_rectify_bbox)
        right_bbox = get_rectified_bbox(right_bbox, true);

      const auto iou = iou_distance(left_bbox, right_bbox);
      if ((iou > m_iou_pair_threshold) && (iou > best_iou)) {
        i_best = (int) i_right;
        best_iou = iou;
      }
    }
    return i_best;
  };

  for (const auto &left_track: left_tracks) {
    const auto left_track_class = most_likely_detection_class(left_track);

    // Skip left tracks not in current frame
    if (left_track->back()->frame() < timestamp.get_frame())
      continue;

    // Find most probable right track match given projected center point
    const auto i_right = most_probable_right_track(left_track, left_track_class);
    if (i_right < 0)
      continue;

    append_paired_frame(left_track, right_tracks[i_right], timestamp);
  }
}


void viame::core::tracks_pairing_from_stereo::pair_left_right_tracks(
    const std::vector<kwiver::vital::track_sptr> &left_tracks,
    const std::vector<viame::core::Tracks3DPositions> &left_3d_pos,
    const std::vector<kwiver::vital::track_sptr> &right_tracks, const kwiver::vital::timestamp &timestamp) {
  bool do_rectify_bbox = m_pairing_method == "PAIRING_RECTIFIED_IOU";
  if (m_pairing_method == "PAIRING_3D")
    pair_left_right_tracks_using_3d_center(left_tracks, left_3d_pos, right_tracks, timestamp);
  else
    pair_left_right_tracks_using_bbox_iou(left_tracks, right_tracks, timestamp, do_rectify_bbox);
}


std::vector<kwiver::vital::track_sptr> viame::core::tracks_pairing_from_stereo::filter_tracks_with_threshold(
    std::vector<kwiver::vital::track_sptr> tracks) const {
  const auto is_outside_detection_number_threshold = [this](const kwiver::vital::track_sptr &track) {
    return ((int) track->size() < m_min_detection_number_threshold) ||
           ((int) track->size() > m_max_detection_number_threshold);
  };

  const auto is_outside_detection_area_threshold = [this](const kwiver::vital::track_sptr &track) {
    double avg_area{};
    for (const auto &state: *track | kwiver::vital::as_object_track) {
      if (state->detection())
        avg_area += state->detection()->bounding_box().area();
    }
    avg_area /= (double) track->size();

    return ((int) avg_area < m_min_detection_surface_threshold_pix) ||
           ((int) avg_area > m_max_detection_surface_threshold_pix);
  };

  tracks.erase(std::remove_if(std::begin(tracks), std::end(tracks), is_outside_detection_number_threshold),
               std::end(tracks));
  tracks.erase(std::remove_if(std::begin(tracks), std::end(tracks), is_outside_detection_area_threshold),
               std::end(tracks));

  return tracks;
}

std::string
viame::core::tracks_pairing_from_stereo::most_likely_detection_class(const kwiver::vital::track_sptr &track) {
  if (!track)
    return {};
  auto state = std::dynamic_pointer_cast<kwiver::vital::object_track_state>(track->back());
  if (!state || !state->detection())
    return {};
  auto detection_type = state->detection()->type();
  if (!detection_type)
    return {};

  std::string most_likely;
  detection_type->get_most_likely(most_likely);
  return most_likely;
}

/// @brief Helper function to erase elements from container given predicate
template<typename ContainerT, typename PredicateT>
void erase_if(ContainerT &items, const PredicateT &predicate) {
  for (auto it = items.begin(); it != items.end();) {
    if (predicate(*it)) it = items.erase(it);
    else ++it;
  }
}

inline std::string to_string(const std::map<size_t, viame::core::Pairing> &left_to_right_pairing) {
  std::stringstream ss;

  ss << "PAIRINGS TO SPLIT : " << std::endl;
  for (const auto &pair: left_to_right_pairing) {
    std::string print_frames{"{"};
    for (const auto &frame_id: pair.second.frame_set)
      print_frames += "," + std::to_string(frame_id);
    print_frames += "}";

    ss << "ID: " << pair.first << ", left: " << pair.second.left_right_id_pair.left_id << ", right: "
       << pair.second.left_right_id_pair.right_id << ", frames: " << print_frames << std::endl;
  }
  return ss.str();
}

std::vector<viame::core::tracks_pairing_from_stereo::Range>
viame::core::tracks_pairing_from_stereo::create_split_ranges_from_track_pairs(
    const std::map<size_t, Pairing> &left_to_right_pairing) const {

  std::cout << to_string(left_to_right_pairing) << std::endl;

  // Find last pairing frame id from all saved pairings
  kwiver::vital::frame_id_t last_pairings_frame_id{};

  for (const auto &pairing: left_to_right_pairing) {
    if (pairing.second.frame_set.empty())
      continue;

    const auto pairing_frame_id = *(pairing.second.frame_set.rbegin());
    if (pairing_frame_id > last_pairings_frame_id)
      last_pairings_frame_id = pairing_frame_id;
  }

  // Init output ID as last unused track ID
  auto last_track_id = last_left_right_track_id() + 1;

  // Initialize ranges being processed, pending ranges, and ranges which have been closed
  std::map<size_t, std::shared_ptr<Range>> open_ranges;
  std::set<std::shared_ptr<Range>> pending_ranges;
  std::vector<Range> ranges;

  // Helper function to find the ranges which are associated with input and need to be removed
  const auto get_pending_ranges_to_remove = [&](const std::shared_ptr<Range> &source_range) {
    std::set<std::shared_ptr<Range >> to_remove;

    if (source_range->detection_count < m_detection_split_threshold)
      return to_remove;

    for (const auto &pending: pending_ranges) {
      if (pending == source_range)
        continue;

      if ((pending->left_id == source_range->left_id) || (pending->right_id == source_range->right_id))
        to_remove.emplace(pending);
    }

    return to_remove;
  };

  // Helper function to remove list of ranges from pending ranges
  const auto remove_pending = [&](const std::shared_ptr<Range> &source_range,
                                  const std::set<std::shared_ptr<Range>> &to_remove) {
    // Remove source range if still pending while detection count is bigger than threshold
    if ((source_range->detection_count >= m_detection_split_threshold) &&
        (pending_ranges.find(source_range) != std::end(pending_ranges)))
      pending_ranges.erase(source_range);

    erase_if(pending_ranges,
             [&](const std::shared_ptr<Range> &range) { return to_remove.find(range) != std::end(to_remove); });
  };

  // Helper function to move input ranges to processed ranges and update open ranges map
  const auto move_open_ranges_to_processed_ranges = [&](const std::set<std::shared_ptr<Range>> &to_remove,
                                                        const std::shared_ptr<Range> &range) {
    auto id_last = range->frame_id_first - 1;


    // Update output ranges
    for (const auto &range_to_remove: to_remove) {
      if (range_to_remove->detection_count < m_detection_split_threshold)
        continue;

      range_to_remove->frame_id_last = id_last;
      ranges.push_back(*range_to_remove);
    }

    // Update open ranges map
    erase_if(open_ranges, [&](const std::pair<size_t, std::shared_ptr<Range>> &pair) {
      return to_remove.find(pair.second) != std::end(to_remove);
    });
  };

  const auto get_overlapping_ranges = [&](const Pairing &pairing) {
    std::vector<std::shared_ptr<Range>> overlapping;
    for (auto &pair: open_ranges) {
      if (pair.second->left_id == pairing.left_right_id_pair.left_id ||
          pair.second->right_id == pairing.left_right_id_pair.right_id)
        overlapping.push_back(pair.second);
    }

    return overlapping;
  };

  const auto create_range_from_pairing = [&](const Pairing &pairing,
                                             kwiver::vital::frame_id_t i_frame) {
    auto range = std::make_shared<Range>();
    range->left_id = pairing.left_right_id_pair.left_id;
    range->right_id = pairing.left_right_id_pair.right_id;
    range->new_track_id = last_track_id++;
    range->detection_count = 1;
    range->frame_id_first = i_frame;
    range->frame_id_last = range->frame_id_first + 1;
    return range;
  };

  const auto mark_overlapping_as_pending = [&](const std::vector<std::shared_ptr<Range>> &overlapping) {
    for (const auto &range: overlapping)
      pending_ranges.emplace(range);
  };


  // For each frame
  for (kwiver::vital::frame_id_t i_frame = 0; i_frame <= last_pairings_frame_id; i_frame++) {
    // For each pairing
    for (const auto &pairing: left_to_right_pairing) {
      // If pairing not in current frame -> continue
      if (pairing.second.frame_set.find(i_frame) == std::end(pairing.second.frame_set))
        continue;

      if (open_ranges.find(pairing.first) != std::end(open_ranges)) {
        // If pairing in opened ranges -> Update range and process associated pending ranges
        const auto &range = open_ranges.find(pairing.first)->second;
        range->detection_count += 1;
        range->frame_id_last = i_frame + 1;

        // Remove pending ranges
        const auto to_remove = get_pending_ranges_to_remove(range);
        remove_pending(range, to_remove);
        move_open_ranges_to_processed_ranges(to_remove, range);
      } else {
        // Get overlapping ranges from opened ranges
        const auto overlapping = get_overlapping_ranges(pairing.second);

        // Create new range and mark overlapping as pending
        open_ranges[pairing.first] = create_range_from_pairing(pairing.second, i_frame);
        pending_ranges.emplace(open_ranges[pairing.first]);
        mark_overlapping_as_pending(overlapping);
      }
    }
  }

  for (auto &open_range: open_ranges) {
    // Ignore pending results
    if (open_range.second->detection_count < m_detection_split_threshold)
      continue;

    // Update last frame for all opened ranges to max value
    open_range.second->frame_id_last = std::numeric_limits<int64_t>::max();

    // Add range to ranges to return
    ranges.push_back(*open_range.second);
  }

  // Return consolidated ranges
  return ranges;
}


std::tuple<std::set<kwiver::vital::track_id_t>, std::set<kwiver::vital::track_id_t>>
viame::core::tracks_pairing_from_stereo::split_ranges_to_tracks(const std::vector<Range> &ranges,
                                                                std::vector<kwiver::vital::track_sptr> &left_tracks,
                                                                std::vector<kwiver::vital::track_sptr> &right_tracks) {

  const auto split_track = [](const kwiver::vital::track_sptr &track, const Range &range) {
    auto new_track = kwiver::vital::track::create();
    new_track->set_id(range.new_track_id);

    for (const auto &state: *track | kwiver::vital::as_object_track)
      if (state->frame() >= range.frame_id_first && state->frame() < range.frame_id_last)
        new_track->append(state);

    return new_track;
  };

  std::set<kwiver::vital::track_id_t> proc_left, proc_right;
  for (const auto &range: ranges) {
    const auto print_i_last =
        range.frame_id_last == std::numeric_limits<int64_t>::max() ? "i_max" : std::to_string(range.frame_id_last);

    std::cout << "PAIRING [" << range.frame_id_first << "," << print_i_last << "] " << range.left_id << ", "
              << range.right_id << " TO " << range.new_track_id << std::endl;
    proc_left.emplace(range.left_id);
    proc_right.emplace(range.right_id);

    left_tracks.push_back(split_track(m_tracks_with_3d_left[range.left_id], range));
    right_tracks.push_back(split_track(m_right_tracks_memo[range.right_id], range));
  }

  return std::make_tuple(proc_left, proc_right);
}
