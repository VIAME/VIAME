#include "ocv_pair_stereo_detections.h"

#include <arrows/ocv/image_container.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d/calib3d.hpp>

void viame::ocv_pair_stereo_detections::load_camera_calibration() {
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

float viame::ocv_pair_stereo_detections::compute_median(std::vector<float> values, bool is_sorted) {
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

cv::Rect viame::ocv_pair_stereo_detections::bbox_to_mask_rect(const kwiver::vital::bounding_box_d &bbox) {
  return {cv::Point2d{bbox.upper_left().x(), bbox.upper_left().y()},
          cv::Point2d{bbox.lower_right().x(), bbox.lower_right().y()}};
}


kwiver::vital::bounding_box_d viame::ocv_pair_stereo_detections::mask_rect_to_bbox(const cv::Rect &rect) {
  return {{rect.tl().x, rect.tl().y},
          {rect.br().x, rect.br().y}};
}


cv::Mat viame::ocv_pair_stereo_detections::get_standard_mask(const kwiver::vital::detected_object_sptr &det) {
  auto vital_mask = det->mask();
  if (!vital_mask) {
    return {};
  }
  using ic = kwiver::arrows::ocv::image_container;
  cv::Mat mask = ic::vital_to_ocv(vital_mask->get_image(), ic::OTHER_COLOR);
  auto size = bbox_to_mask_rect(det->bounding_box()).size();
  cv::Rect intersection(0, 0, std::min(size.width, mask.cols), std::min(size.height, mask.rows));

  if (mask.size() == size) {
    return mask;
  }
  cv::Mat standard_mask(size, CV_8UC1, cv::Scalar(0));
  mask(intersection).copyTo(standard_mask(intersection));
  return standard_mask;
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


viame::Detections3DPositions viame::ocv_pair_stereo_detections::estimate_3d_position_from_detection(
    const kwiver::vital::detected_object_sptr &detection, const cv::Mat &pos_3d_map, bool do_undistort_points,
    float bbox_crop_ratio) const {
  // Extract mask and corresponding mask bounding box from input detection
  auto mask = get_standard_mask(detection);

  // If mask is invalid return the estimated position from bounding box center
  if (mask.empty())
    return estimate_3d_position_from_bbox(detection->bounding_box(), pos_3d_map, bbox_crop_ratio, do_undistort_points);

  // Otherwise, returns average 3D distance for each point in mask
  return estimate_3d_position_from_unrectified_mask(detection->bounding_box(), pos_3d_map, mask, do_undistort_points);
}

kwiver::vital::bounding_box_d
viame::ocv_pair_stereo_detections::get_rectified_bbox(const kwiver::vital::bounding_box_d &bbox,
                                                                bool is_left_image) const {
  const auto tl = undistort_point({bbox.upper_left().x(), bbox.upper_left().y()}, is_left_image);
  const auto br = undistort_point({bbox.lower_right().x(), bbox.lower_right().y()}, is_left_image);
  return {tl.x, tl.y, br.x, br.y};
}

bool viame::ocv_pair_stereo_detections::point_is_valid(float x, float y, float z) {
  return ((z > 0) && std::isfinite(x) && std::isfinite(y) && std::isfinite(z));
}

bool viame::ocv_pair_stereo_detections::point_is_valid(const cv::Vec3f &pt) {
  return point_is_valid(pt[0], pt[1], pt[2]);
}

viame::Detections3DPositions
viame::ocv_pair_stereo_detections::estimate_3d_position_from_bbox(const kwiver::vital::bounding_box_d &bbox,
                                                                            const cv::Mat &pos_3d_map, float crop_ratio,
                                                                            bool do_undistort_points) const {
  const auto rectified_bbox = do_undistort_points ? get_rectified_bbox(bbox, true) : bbox;

  // depth from median of values in the center part of the bounding box
  float crop_width = crop_ratio * (float) rectified_bbox.width();
  float crop_height = crop_ratio * (float) rectified_bbox.height();
  cv::Rect crop_rect{(int) (rectified_bbox.center().x() - crop_width / 2),
                     (int) (rectified_bbox.center().y() - crop_height / 2), (int) crop_width, (int) crop_height};

  // Intersect crop rectangle with 3D map rect to avoid out of bounds crop
  crop_rect = crop_rect & cv::Rect(0, 0, pos_3d_map.size().width, pos_3d_map.size().height);

  if (m_verbose)
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

viame::Detections3DPositions
viame::ocv_pair_stereo_detections::estimate_3d_position_from_unrectified_mask(
    const kwiver::vital::bounding_box_d &bbox, const cv::Mat &pos_3d_map, const cv::Mat &mask,
    bool do_undistort_points) const {
  // Early return if bbox crop is out of the 3D map
  if (bbox.width() == 0 || bbox.height() == 0)
    return {};

  // Find all distorted positions where mask is not empty
  std::vector<cv::Point2d> mask_distorted_coords;
  const auto mask_tl = bbox.upper_left();
  for (int i_x = 0; i_x < mask.size().width; i_x++) {
    for (int i_y = 0; i_y < mask.size().height; i_y++) {
      if (mask.at<uchar>(i_y, i_x) > 0) {
        mask_distorted_coords.emplace_back(cv::Point2d(mask_tl.x() + i_x, mask_tl.y() + i_y));
      }
    }
  }

  // If no segmentation, early return
  if (mask_distorted_coords.empty())
    return {};

  // Undistort mask points
  auto undistorted_mask_coords = do_undistort_points ? undistort_point(mask_distorted_coords, true)
                                                     : mask_distorted_coords;

  const auto rectified_bbox = do_undistort_points ? get_rectified_bbox(bbox, true) : bbox;
  return estimate_3d_position_from_point_coordinates(rectified_bbox, undistorted_mask_coords, pos_3d_map);
}

viame::Detections3DPositions
viame::ocv_pair_stereo_detections::estimate_3d_position_from_point_coordinates(
    const kwiver::vital::bounding_box_d &rectified_bbox, const std::vector<cv::Point2d> &undistorted_mask_coords,
    const cv::Mat &pos_3d_map) const {
  // Early return if no segmentation points
  if (undistorted_mask_coords.empty())
    return {};

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
  return create_3d_position(xs, ys, zs, rectified_bbox, pos_3d_map, score);
}

viame::Detections3DPositions
viame::ocv_pair_stereo_detections::create_3d_position(const std::vector<float> &xs,
                                                                const std::vector<float> &ys,
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

  Detections3DPositions position;
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
  if (m_verbose) {
    print(position.center3d, "CENTER");
    print(position.center3d_proj_to_right_image, "PROJ CENTER TO RIGHT");
    print(position.rectified_left_bbox, "RECTIFIED LEFT BBOX");
    print(position.left_bbox_proj_to_right_image, "PROJ LEFT BBOX");
    std::cout << std::endl;
  }

  return position;
}

kwiver::vital::bounding_box_d
viame::ocv_pair_stereo_detections::project_to_right_image(const kwiver::vital::bounding_box_d &bbox,
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
viame::ocv_pair_stereo_detections::project_to_right_image(const std::vector<cv::Vec3f> &points_3d) const {
  // Sanity check on input vect list
  if (points_3d.size() != 2)
    VITAL_THROW(kwiver::vital::invalid_data,
                "Wrong input 3D point number. Expected 2, got : " + std::to_string(points_3d.size()));

  // Project points to right camera coordinates
  std::vector<cv::Point2f> projectedPoints;
  cv::projectPoints(points_3d, m_Rvec, m_T, m_K2, m_D2, projectedPoints);
  return {projectedPoints[0].x, projectedPoints[0].y, projectedPoints[1].x, projectedPoints[1].y};
}

cv::Point2f viame::ocv_pair_stereo_detections::project_to_right_image(const cv::Point3f &points_3d) const {
  std::vector<cv::Point2f> projectedPoints;
  cv::projectPoints(std::vector<cv::Point3f>{points_3d}, m_Rvec, m_T, m_K2, m_D2, projectedPoints);
  return projectedPoints[0];
}


std::vector<viame::Detections3DPositions>
viame::ocv_pair_stereo_detections::update_left_detections_3d_positions(
    const std::vector<kwiver::vital::detected_object_sptr> &detections, const cv::Mat &cv_disparity_map) const {
  const auto cv_pos_3d_map = reproject_3d_depth_map(cv_disparity_map);
  std::vector<Detections3DPositions> positions;
  for (const auto &detection: detections) {
    positions.emplace_back(update_left_detection_3d_position(detection, cv_pos_3d_map));
  }
  return positions;
}


viame::Detections3DPositions viame::ocv_pair_stereo_detections::update_left_detection_3d_position(
    const kwiver::vital::detected_object_sptr &detection, const cv::Mat &cv_pos_3d_map) const {

  // Process 3D coordinates for frame matching the current depth image
  auto position = estimate_3d_position_from_detection(detection, cv_pos_3d_map, true, 1.f / 3.f);

  // Add 3d estimations to state if score is valid
  if (position.score > 0) {
    detection->add_note(":stereo3d_x=" + std::to_string(position.center3d.x));
    detection->add_note(":stereo3d_y=" + std::to_string(position.center3d.y));
    detection->add_note(":stereo3d_z=" + std::to_string(position.center3d.z));
    detection->add_note(":score=" + std::to_string(position.score));
  }

  return position;
}


double viame::ocv_pair_stereo_detections::iou_distance(const kwiver::vital::bounding_box_d &bbox1,
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


cv::Mat viame::ocv_pair_stereo_detections::reproject_3d_depth_map(const cv::Mat &cv_disparity_left) const {
  cv::Mat cv_pos_3d_left_map;
  cv::reprojectImageTo3D(cv_disparity_left, cv_pos_3d_left_map, m_Q, false);
  return cv_pos_3d_left_map;
}


cv::Point2d
viame::ocv_pair_stereo_detections::undistort_point(const cv::Point2d &point, bool is_left_image) const {
  return undistort_point(std::vector<cv::Point2d>{point}, is_left_image)[0];
}


std::vector<cv::Point2d>
viame::ocv_pair_stereo_detections::undistort_point(const std::vector<cv::Point2d> &point,
                                                             bool is_left_image) const {
  std::vector<cv::Point2d> points_undist;

  if (is_left_image)
    cv::undistortPoints(std::vector<cv::Point2d>{point}, points_undist, m_K1, m_D1, m_R1, m_P1);
  else
    cv::undistortPoints(std::vector<cv::Point2d>{point}, points_undist, m_K2, m_D2, m_R2, m_P2);
  return points_undist;
}


/// @class ProcessTracker
/// @brief Helper class to track the processed detection during the different processing
template<typename T>
class ProcessTracker {
public:
  bool is_processed(const T &value) const {
    return m_processed.find(value) != std::end(m_processed);
  }

  void emplace(const T &value) {
    m_processed.emplace(value);
  }

  void clear() {
    m_processed.clear();
  }
private:
  std::set<T> m_processed;
};

std::vector<std::vector<size_t>>
viame::ocv_pair_stereo_detections::pair_left_right_detections_using_3d_center(
    const std::vector<kwiver::vital::detected_object_sptr> &left_detections,
    const std::vector<viame::Detections3DPositions> &left_3d_pos,
    const std::vector<kwiver::vital::detected_object_sptr> &right_detections) {

  std::vector<std::vector<size_t>> paired_detections;
  ProcessTracker<size_t> tracker;

  const auto most_probable_right_detection = [&right_detections, &tracker](
      const Eigen::Matrix<double, 2, 1> &left_point, const std::string &left_class) {
    int i_best = -1;
    auto dist_best = std::numeric_limits<double>::max();

    for (size_t i_right = 0; i_right < right_detections.size(); i_right++) {
      // Skip right tracks not in current frame or with different detection class
      const auto &right_detection = right_detections[i_right];
      if (most_likely_detection_class(right_detection) != left_class || tracker.is_processed(i_right))
        continue;

      auto right_bbox = right_detection->bounding_box();
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

  for (size_t i_left = 0; i_left < left_detections.size(); i_left++) {
    const auto &left_detection = left_detections[i_left];
    const auto left_class = most_likely_detection_class(left_detection);

    // Skip left tracks not in current frame or invalid
    if (!left_3d_pos[i_left].is_valid())
      continue;

    // Find most probable right track match given projected center point
    const auto proj_left_point = left_3d_pos[i_left].center3d_proj_to_right_image;
    const auto i_right = most_probable_right_detection({proj_left_point.x, proj_left_point.y}, left_class);
    if (i_right < 0)
      continue;

    paired_detections.emplace_back(std::vector<size_t>{i_left, static_cast<size_t>(i_right)});
    tracker.emplace(i_right);
  }
  return paired_detections;
}


std::vector<std::vector<size_t>> viame::ocv_pair_stereo_detections::pair_left_right_tracks_using_bbox_iou(
    const std::vector<kwiver::vital::detected_object_sptr> &left_detections,
    const std::vector<kwiver::vital::detected_object_sptr> &right_detections, bool do_rectify_bbox) {

  std::vector<std::vector<size_t>> paired_detections;
  ProcessTracker<size_t> tracker;

  const auto most_probable_right_track = [&right_detections, do_rectify_bbox, &tracker, this](
      const kwiver::vital::detected_object_sptr &left_detection, const std::string &left_class) {
    int i_best = -1;
    auto best_iou = std::numeric_limits<double>::lowest();
    auto left_bbox = left_detection->bounding_box();
    if (do_rectify_bbox)
      left_bbox = get_rectified_bbox(left_bbox, true);

    if (!left_bbox.is_valid())
      return i_best;

    for (size_t i_right = 0; i_right < right_detections.size(); i_right++) {
      // Skip right tracks not in current frame or with different detection class
      const auto &right_track = right_detections[i_right];
      if (most_likely_detection_class(right_track) != left_class || tracker.is_processed(i_right))
        continue;

      auto right_bbox = right_track->bounding_box();
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

  for (size_t i_left = 0; i_left < left_detections.size(); i_left++) {
    const auto &left_detection = left_detections[i_left];
    const auto left_track_class = most_likely_detection_class(left_detection);

    // Find most probable right track match given projected center point
    const auto i_right = most_probable_right_track(left_detection, left_track_class);
    if (i_right < 0)
      continue;

    paired_detections.emplace_back(std::vector<size_t>{i_left, static_cast<size_t>(i_right)});
    tracker.emplace(i_right);
  }
  return paired_detections;
}


std::vector<std::vector<size_t>> viame::ocv_pair_stereo_detections::pair_left_right_detections(
    const std::vector<kwiver::vital::detected_object_sptr> &left_detections,
    const std::vector<viame::Detections3DPositions> &left_3d_pos,
    const std::vector<kwiver::vital::detected_object_sptr> &right_detections) {
  bool do_rectify_bbox = m_pairing_method == "PAIRING_RECTIFIED_IOU";
  if (m_pairing_method == "PAIRING_3D")
    return pair_left_right_detections_using_3d_center(left_detections, left_3d_pos, right_detections);
  else
    return pair_left_right_tracks_using_bbox_iou(left_detections, right_detections, do_rectify_bbox);
}


std::string viame::ocv_pair_stereo_detections::most_likely_detection_class(
    const kwiver::vital::detected_object_sptr &detection) {
  if (!detection)
    return {};

  auto detection_type = detection->type();
  if (!detection_type)
    return {};

  std::string most_likely;
  detection_type->get_most_likely(most_likely);
  return most_likely;
}