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


std::tuple <cv::Mat, cv::Rect>
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
viame::core::tracks_pairing_from_stereo::get_rectified_bbox(const kwiver::vital::bounding_box_d &bbox) const {
  const auto tl = undistort_point({bbox.upper_left().x(), bbox.upper_left().y()});
  const auto br = undistort_point({bbox.lower_right().x(), bbox.lower_right().y()});
  return {tl.x, tl.y, br.x, br.y};
}

cv::Rect viame::core::tracks_pairing_from_stereo::get_rectified_bbox(const cv::Rect &bbox) const {
  return bbox_to_mask_rect(get_rectified_bbox(mask_rect_to_bbox(bbox)));
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
  const auto rectified_bbox = get_rectified_bbox(bbox);

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
  // Early return if bbox crop is out of the 3D map (detection out of left / right ROI overlap)
  if (bbox.width == 0 || bbox.height == 0)
    return {};

  // Find all distorted positions where mask is not empty
  std::vector <cv::Point2d> mask_distorted_coords;
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
  auto undistorted_mask_coords = undistort_point(mask_distorted_coords);

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
    return Eigen::Matrix < double, 2, 1 > {std::max(0., std::min(pos_3d_map.size().width - 1., corner.x())),
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
    return std::vector < cv::Vec3f > {tl_3d, br_3d};
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

    return Eigen::Matrix < double, 2, 1 > {x, y};
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
  return project_to_right_image(std::vector < cv::Vec3f > {tl_3d, br_3d});
}

kwiver::vital::bounding_box_d
viame::core::tracks_pairing_from_stereo::project_to_right_image(const std::vector <cv::Vec3f> &points_3d) const {
  // Sanity check on input vect list
  if (points_3d.size() != 2)
    VITAL_THROW(kwiver::vital::invalid_data,
                "Wrong input 3D point number. Expected 2, got : " + std::to_string(points_3d.size()));

  // Project points to right camera coordinates
  std::vector <cv::Point2f> projectedPoints;
  cv::projectPoints(points_3d, m_Rvec, m_T, m_K2, m_D2, projectedPoints);
  return {projectedPoints[0].x, projectedPoints[0].y, projectedPoints[1].x, projectedPoints[1].y};
}

cv::Point2f viame::core::tracks_pairing_from_stereo::project_to_right_image(const cv::Point3f &points_3d) const {
  std::vector <cv::Point2f> projectedPoints;
  cv::projectPoints(std::vector < cv::Point3f > {points_3d}, m_Rvec, m_T, m_K2, m_D2, projectedPoints);
  return projectedPoints[0];
}


std::tuple <std::vector<kwiver::vital::track_sptr>, std::vector<viame::core::Tracks3DPositions>>
viame::core::tracks_pairing_from_stereo::update_left_tracks_3d_position(
    const std::vector <kwiver::vital::track_sptr> &tracks, const cv::Mat &cv_disparity_map,
    const kwiver::vital::timestamp &timestamp) {
  const auto cv_pos_3d_map = reproject_3d_depth_map(cv_disparity_map);

  std::vector <kwiver::vital::track_sptr> filtered_tracks;
  std::vector <Tracks3DPositions> tracks_positions;

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

std::vector <kwiver::vital::track_sptr> viame::core::tracks_pairing_from_stereo::keep_right_tracks_in_current_frame(
    const std::vector <kwiver::vital::track_sptr> &tracks, const kwiver::vital::timestamp &timestamp) {
  std::vector <kwiver::vital::track_sptr> filtered_tracks;

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

double
viame::core::tracks_pairing_from_stereo::iou_distance(const std::shared_ptr <kwiver::vital::object_track_state> &t1,
                                                      const std::shared_ptr <kwiver::vital::object_track_state> &t2) {
  if (!t1 || !t2)
    return 0.;

  return iou_distance(t1->detection()->bounding_box(), t2->detection()->bounding_box());
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

std::vector <std::vector<size_t>>
viame::core::tracks_pairing_from_stereo::group_overlapping_tracks_indexes_in_current_frame(
    const std::vector <kwiver::vital::track_sptr> &tracks, const kwiver::vital::timestamp &timestamp) const {
  // Init set of processed tracks
  std::set <kwiver::vital::track_id_t> processed;

  auto processed_contains = [&](const kwiver::vital::track_sptr &track) -> bool {
    return processed.find(track->id()) != std::end(processed);
  };

  auto do_skip_track = [&](const kwiver::vital::track_sptr &track) -> bool {
    return processed_contains(track) || (track->last_frame() < timestamp.get_frame());
  };

  std::vector <std::vector<size_t>> tracks_group;

  // For each track
  for (size_t i_t1 = 0; i_t1 < tracks.size(); i_t1++) {
    const auto &track1 = tracks[i_t1];

    // If track in set or track last frame doesn't match current frame, continue
    if (do_skip_track(track1))
      continue;

    // Add track to processed track
    processed.emplace(track1->id());

    // Init group with current track
    std::vector <size_t> group{i_t1};

    // For each track
    for (size_t i_t2 = 0; i_t2 < tracks.size(); i_t2++) {
      const auto &track2 = tracks[i_t2];

      // If track in set or track last frame doesn't match current frame, continue
      if (do_skip_track(track2))
        continue;

      // If the IOU distance of last track bounding box is more than threshold, push to group and add to process set
      if (iou_distance(std::dynamic_pointer_cast<kwiver::vital::object_track_state>(track1->back()),
                       std::dynamic_pointer_cast<kwiver::vital::object_track_state>(track2->back())) >=
          m_iou_merge_threshold) {
        processed.emplace(track2->id());
        group.push_back(i_t2);
      }
    }

    // Push group to groups
    tracks_group.push_back(group);
  }

  // returns groups
  return tracks_group;
}

std::vector <std::vector<kwiver::vital::track_sptr>>
viame::core::tracks_pairing_from_stereo::group_overlapping_tracks_in_current_frame(
    const std::vector <kwiver::vital::track_sptr> &tracks, const kwiver::vital::timestamp &timestamp) const {
  return group_vector_by_ids(tracks, group_overlapping_tracks_indexes_in_current_frame(tracks, timestamp));
}

std::vector <kwiver::vital::bounding_box_d> viame::core::tracks_pairing_from_stereo::merge_clustered_bbox(
    const std::vector <std::vector<kwiver::vital::track_sptr>> &clusters) {
  std::vector <std::vector<kwiver::vital::bounding_box_d>> clusters_bboxs;

  // Extract bounding boxes from grouped tracks
  for (const auto &cluster: clusters) {
    std::vector <kwiver::vital::bounding_box_d> cluster_bboxs;

    for (const auto &track: cluster) {
      auto track_state = std::dynamic_pointer_cast<kwiver::vital::object_track_state>(track->back());
      cluster_bboxs.emplace_back(
          track_state ? track_state->detection()->bounding_box() : kwiver::vital::bounding_box_d{});
    }

    clusters_bboxs.emplace_back(cluster_bboxs);
  }

  // Return merged bounding boxes
  return merge_clustered_bbox(clusters_bboxs);
}

std::vector <kwiver::vital::bounding_box_d> viame::core::tracks_pairing_from_stereo::merge_clustered_bbox(
    const std::vector <std::vector<kwiver::vital::bounding_box_d>> &clusters) {
  auto merge = [=](const kwiver::vital::bounding_box_d &bbox, const kwiver::vital::bounding_box_d &other_rect_bbox) {
    if (!other_rect_bbox.is_valid())
      return bbox;
    Eigen::AlignedBox<double, 2> merged_eig_bbox{bbox.upper_left(), bbox.lower_right()};
    Eigen::AlignedBox<double, 2> other_rect_eig_bbox{other_rect_bbox.upper_left(), other_rect_bbox.lower_right()};

    merged_eig_bbox = merged_eig_bbox.merged(other_rect_eig_bbox);
    return kwiver::vital::bounding_box_d{merged_eig_bbox.min(), merged_eig_bbox.max()};
  };

  std::vector <kwiver::vital::bounding_box_d> merged_bboxs;
  int i_merge{};
  for (const auto &cluster: clusters) {
    kwiver::vital::bounding_box_d merged;

    for (const auto &bbox: cluster) {
      merged = merge(merged, bbox);
    }

    print(merged, std::to_string(i_merge));
    i_merge++;
    merged_bboxs.push_back(merged);
  }

  return merged_bboxs;
}

std::vector <kwiver::vital::bounding_box_d>
viame::core::tracks_pairing_from_stereo::merge_3d_left_projected_to_right_bbox(
    const std::vector <std::vector<viame::core::Tracks3DPositions>> &clusters) {
  std::vector <std::vector<kwiver::vital::bounding_box_d>> clusters_bboxs;

  // Extract left bounding boxes projected in right image from grouped positions
  for (const auto &cluster: clusters) {
    std::vector <kwiver::vital::bounding_box_d> cluster_bboxs;

    for (const auto &position: cluster) {
      cluster_bboxs.emplace_back(position.left_bbox_proj_to_right_image);
    }

    clusters_bboxs.emplace_back(cluster_bboxs);
  }

  // Return merged bounding boxes
  return merge_clustered_bbox(clusters_bboxs);
}

kwiver::vital::track_id_t
viame::core::tracks_pairing_from_stereo::most_likely_paired_right_cluster(const kwiver::vital::bounding_box_d &bbox,
                                                                          const std::vector <kwiver::vital::bounding_box_d> &other_bboxs) const {
  double max_iou = std::numeric_limits<double>::lowest();
  kwiver::vital::track_id_t i_max_iou = -1;

  for (size_t i_other = 0; i_other < other_bboxs.size(); i_other++) {
    auto iou = iou_distance(bbox, other_bboxs[i_other]);
    if ((iou >= m_iou_pair_threshold) && (iou > max_iou)) {
      max_iou = iou;
      i_max_iou = (kwiver::vital::track_id_t) i_other;
    }
  }

  return i_max_iou;
}

std::vector <std::tuple<std::vector <
                        kwiver::vital::track_sptr>, std::vector<kwiver::vital::track_sptr>, std::vector<viame::core::Tracks3DPositions>>>
viame::core::tracks_pairing_from_stereo::pair_left_right_clusters(
    const std::vector <std::vector<kwiver::vital::track_sptr>> &left_cluster,
    const std::vector <std::vector<kwiver::vital::track_sptr>> &right_cluster,
    const std::vector <std::vector<viame::core::Tracks3DPositions>> &left_3d_pos) const {
  // Init output clusters
  std::vector < std::tuple < std::vector < kwiver::vital::track_sptr > , std::vector < kwiver::vital::track_sptr >,
      std::vector < Tracks3DPositions>>> clusters;

  // For each left cluster, merge bbox in right image
  std::cout << "LEFT BBOXES" << std::endl;
  auto merged_proj_left_bbox = merge_3d_left_projected_to_right_bbox(left_3d_pos);

  // For each right cluster, process merged BBox in left ref using stereo rectification
  std::cout << "RIGHT BBOXES" << std::endl;
  auto merged_right_bbox = merge_clustered_bbox(right_cluster);

  // For each left cluster, calculate IOU with each right cluster and push result which are above threshold
  std::set <kwiver::vital::track_id_t> processed;
  for (size_t i_left = 0; i_left < left_cluster.size(); i_left++) {
    auto i_most_likely = most_likely_paired_right_cluster(merged_proj_left_bbox[i_left], merged_right_bbox);

    if (i_most_likely < 0 || (processed.find(i_most_likely) != std::end(processed)))
      continue;

    // Push most likely left / right cluster pair and remove pair from further pairing
    processed.emplace(i_most_likely);
    clusters.emplace_back(left_cluster[i_left], right_cluster[i_most_likely], left_3d_pos[i_left]);
  }

  // Return left / right clusters
  return clusters;
}

kwiver::vital::track_id_t viame::core::tracks_pairing_from_stereo::last_left_right_track_id() const {
  auto get_map_ids = [](const std::map <kwiver::vital::track_id_t, kwiver::vital::track_sptr> &map) {
    std::set <kwiver::vital::track_id_t> track_ids;
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


void viame::core::tracks_pairing_from_stereo::pair_left_right_tracks_in_each_cluster(
    const std::vector <kwiver::vital::track_sptr> &left_tracks,
    const std::vector <kwiver::vital::track_sptr> &right_tracks,
    const std::vector <viame::core::Tracks3DPositions> &left_positions) {
  // Initialize processed right tracks
  std::set<int> processed;

  auto find_closest_right_track_to_proj_left_bbox = [&](size_t i_left) {
    int i_closest = -1;
    auto left_pos = left_positions[i_left];
    double max_iou = std::numeric_limits<double>::lowest();

    // Early return if left already paired
    if (is_left_track_paired(left_tracks[i_left]))
      return i_closest;

    for (int i_right = 0; i_right < (int) right_tracks.size(); i_right++) {
      // Skip tracks already paired with left track
      if (processed.find(i_right) != std::end(processed) || is_right_track_paired(right_tracks[i_right]))
        continue;

      // Fetch right track bounding box
      const auto &right_track = std::dynamic_pointer_cast<kwiver::vital::object_track_state>(
          right_tracks[i_right]->back());
      if (!right_track)
        continue;

      // Intersect right and left bounding box
      auto right_bbox = right_track->detection()->bounding_box();
      auto iou = iou_distance(right_bbox, left_pos.left_bbox_proj_to_right_image);

      // Save best
      if (iou > max_iou) {
        i_closest = i_right;
        max_iou = iou;
      }
    }

    return i_closest;
  };

  auto update_track_ids = [this](const kwiver::vital::track_sptr &left, const kwiver::vital::track_sptr &right) {
    std::cout << "PAIRING : " << left->id() << ", " << right->id() << std::endl;
    m_left_to_right_pairing[left->id()] = right->id();
  };

  for (size_t i_left = 0; i_left < left_tracks.size(); i_left++) {
    auto i_closest = find_closest_right_track_to_proj_left_bbox(i_left);
    if (i_closest < 0)
      continue;

    processed.emplace(i_closest);
    update_track_ids(left_tracks[i_left], right_tracks[i_closest]);
  }
}


void viame::core::tracks_pairing_from_stereo::pair_left_right_tracks_using_bbox(
    const std::vector <kwiver::vital::track_sptr> &left_tracks,
    const std::vector <viame::core::Tracks3DPositions> &left_3d_pos,
    const std::vector <kwiver::vital::track_sptr> &right_tracks, const kwiver::vital::timestamp &timestamp) {

  // Cluster overlapping left/right tracks
  auto left_tracks_clusters_ids = group_overlapping_tracks_indexes_in_current_frame(left_tracks, timestamp);
  auto left_tracks_clusters = group_vector_by_ids(left_tracks, left_tracks_clusters_ids);
  auto left_clusters_3dpos = group_vector_by_ids(left_3d_pos, left_tracks_clusters_ids);
  auto right_tracks_clusters = group_overlapping_tracks_in_current_frame(right_tracks, timestamp);

  // Find overlapping left / right tracks clusters
  auto left_right_clusters = pair_left_right_clusters(left_tracks_clusters, right_tracks_clusters, left_clusters_3dpos);
  std::cout << "CLUSTER SIZE : " << left_right_clusters.size() << std::endl;

  // For each overlapping left / right tracks clusters, pair left / right tracks
  pair_left_right_tracks_in_each_cluster(left_right_clusters);
}


void viame::core::tracks_pairing_from_stereo::pair_left_right_tracks_in_each_cluster(const std::vector <std::tuple<
    std::vector < kwiver::vital::track_sptr>, std::vector<kwiver::vital::track_sptr>, std::vector<Tracks3DPositions>>
> &left_right_clusters) {
for (
auto left_right_cluster
: left_right_clusters) {
pair_left_right_tracks_in_each_cluster(std::get<0>(left_right_cluster), std::get<1>(left_right_cluster), std::get<2>(
    left_right_cluster)
);
}}


cv::Mat viame::core::tracks_pairing_from_stereo::reproject_3d_depth_map(const cv::Mat &cv_disparity_left) const {
  cv::Mat cv_pos_3d_left_map;
  cv::reprojectImageTo3D(cv_disparity_left, cv_pos_3d_left_map, m_Q, false);
  return cv_pos_3d_left_map;
}

cv::Point2d viame::core::tracks_pairing_from_stereo::undistort_point(const cv::Point2d &point) const {
  return undistort_point(std::vector < cv::Point2d > {point})[0];
}

std::vector <cv::Point2d>
viame::core::tracks_pairing_from_stereo::undistort_point(const std::vector <cv::Point2d> &point) const {
  std::vector <cv::Point2d> points_undist;
  cv::undistortPoints(std::vector < cv::Point2d > {point}, points_undist, m_K1, m_D1, m_R1, m_P1);
  return points_undist;
}

std::tuple <std::vector<kwiver::vital::track_sptr>, std::vector<kwiver::vital::track_sptr>>
viame::core::tracks_pairing_from_stereo::get_left_right_tracks_with_pairing() {
  std::vector <kwiver::vital::track_sptr> left_tracks, right_tracks;

  // Iterate over all paired tracks first
  std::set <kwiver::vital::track_id_t> proc_left, proc_right;
  auto last_track_id = last_left_right_track_id() + 1;
  for (const auto &pair: m_left_to_right_pairing) {
    if (proc_right.find(pair.second) != std::end(proc_right)) {
      std::cout << "RIGHT TRACK ALREADY PAIRED " << pair.first << ", " << pair.second << std::endl;
      continue;
    }

    proc_left.emplace(pair.first);
    proc_right.emplace(pair.second);

    auto left_track = m_tracks_with_3d_left[pair.first]->clone();
    auto right_track = m_right_tracks_memo[pair.second]->clone();

    // Update left right pairing in case the pairing id is different.
    // Otherwise, keep the track id
    if (pair.first != pair.second) {
      std::cout << "PAIRING " << pair.first << ", " << pair.second << " TO " << last_track_id << std::endl;
      left_track->set_id(last_track_id);
      right_track->set_id(last_track_id);
      last_track_id++;
    } else {
      std::cout << "KEEPING PAIRING " << pair.first << ", " << pair.second << std::endl;
    }
    left_tracks.emplace_back(left_track);
    right_tracks.emplace_back(right_track);
  }

  // Append other tracks
  const auto append_unprocessed_tracks = [](
      const std::map <kwiver::vital::track_id_t, kwiver::vital::track_sptr> &tracks_map,
      std::set <kwiver::vital::track_id_t> &processed, std::vector <kwiver::vital::track_sptr> &vect) {
    for (const auto &pair: tracks_map) {
      if (processed.find(pair.first) != std::end(processed))
        continue;

      processed.emplace(pair.first);
      vect.emplace_back(pair.second->clone());
    }
  };

  append_unprocessed_tracks(m_tracks_with_3d_left, proc_left, left_tracks);
  append_unprocessed_tracks(m_right_tracks_memo, proc_right, right_tracks);

  return {left_tracks, right_tracks};
}

bool viame::core::tracks_pairing_from_stereo::is_left_track_paired(const kwiver::vital::track_sptr &left_track) const {
  return m_left_to_right_pairing.find(left_track->id()) != std::end(m_left_to_right_pairing);
}

bool
viame::core::tracks_pairing_from_stereo::is_right_track_paired(const kwiver::vital::track_sptr &right_track) const {
  return std::any_of(std::begin(m_left_to_right_pairing), std::end(m_left_to_right_pairing),
                     [&](const std::pair <kwiver::vital::track_id_t, kwiver::vital::track_id_t> &pair) {
                       return pair.second == right_track->id();
                     });
}

void viame::core::tracks_pairing_from_stereo::pair_left_right_tracks_using_3d_center(
    const std::vector <kwiver::vital::track_sptr> &left_tracks,
    const std::vector <viame::core::Tracks3DPositions> &left_3d_pos,
    const std::vector <kwiver::vital::track_sptr> &right_tracks, const kwiver::vital::timestamp &timestamp) {

  const auto most_probable_right_track = [&](const Eigen::Matrix<double, 2, 1> &left_point) {
    int i_best = -1;
    auto dist_best = std::numeric_limits<double>::max();

    for (size_t i_right = 0; i_right < right_tracks.size(); i_right++) {
      // Skip right tracks not in current frame, already processed or already paired
      const auto &right_track = right_tracks[i_right];
      if (right_track->back()->frame() < timestamp.get_frame() || is_right_track_paired(right_track))
        continue;

      const auto right_track_state = std::dynamic_pointer_cast<kwiver::vital::object_track_state>(right_track->back());
      if (!right_track_state)
        continue;

      const auto right_bbox = right_track_state->detection()->bounding_box();
      if (!right_bbox.contains(left_point))
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

    // Skip left tracks not in current frame, already paired or invalid
    if (left_track->back()->frame() < timestamp.get_frame() || is_left_track_paired(left_track) ||
        !left_3d_pos[i_left].is_valid())
      continue;

    // Find most probable right track match given projected center point
    const auto proj_left_point = left_3d_pos[i_left].center3d_proj_to_right_image;
    const auto i_right = most_probable_right_track({proj_left_point.x, proj_left_point.y});
    if (i_right < 0)
      continue;

    // Pair left to right
    std::cout << "PAIRING : " << left_track->id() << ", " << right_tracks[i_right]->id() << std::endl;
    m_left_to_right_pairing[left_track->id()] = right_tracks[i_right]->id();
  }
}




