#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "../tracks_pairing_from_stereo.h"
#include "../detections_pairing_from_stereo.h"
#include <vital/types/track.h>
#include <vital/types/timestamp.h>
#include <arrows/ocv/image_container.h>

namespace kv = kwiver::vital;
using namespace viame::core;

namespace kwiver::vital {
/// @brief Pretty print for BBox for Gtest
void PrintTo(const kv::bounding_box_d &bbox, ::std::ostream *os) {
  *os << "kv::bounding_box_d( upperLeft {" << bbox.upper_left().x() << ", " << bbox.upper_left().y()
      << "}, lowerRight {" << bbox.lower_right().x() << ", " << bbox.lower_right().y() << "})";
}

std::ostream &operator<<(std::ostream &os, const kv::bounding_box_d &bbox) {
  PrintTo(bbox, &os);
  return os;
}
} // kwiver::vital

bool is_near(double a, double b, double tol) {
  return std::abs(a - b) < tol;
}

bool is_near(const Eigen::Matrix<double, 2, 1> &actual, const Eigen::Matrix<double, 2, 1> &expected, double pix_tol) {
  return is_near(actual.x(), expected.x(), pix_tol) && is_near(actual.y(), expected.y(), pix_tol);
}

bool is_near(const kv::bounding_box_d &actual, const kv::bounding_box_d &expected, double pix_tol = 100.) {
  return is_near(actual.upper_left(), expected.upper_left(), pix_tol) &&
         is_near(actual.lower_right(), expected.lower_right(), pix_tol);
}

#define ASSERT_BBOX_NEAR(actual, expected, tol) \
  ASSERT_TRUE(is_near(actual, expected, tol)) << "\tactual : " << actual << "\n\texpected : "  << expected

// simple struct to define track information
struct DetectionInfo {
  int track_id{};
  int frame_id{};
  kv::bounding_box_d bbox{};
  double confidence{1.0};
};

constexpr double bbox_size{100};
constexpr double right_shift_pix{240};
inline std::vector<DetectionInfo> left_tracks_detections() {
  return {{1, 1, kv::bounding_box_d{0, 0, bbox_size, bbox_size}},//
          {1, 2, kv::bounding_box_d{0, 0, bbox_size, bbox_size}},//
          {2, 1, kv::bounding_box_d{1280 - bbox_size, 0, 1280, bbox_size}},//
          {2, 2, kv::bounding_box_d{1280 - bbox_size, 0, 1280, bbox_size}},//
  };
}

inline std::vector<DetectionInfo> right_tracks_detections() {
  return {{3, 1, kv::bounding_box_d{0, 0, bbox_size, bbox_size}},//
          {3, 2, kv::bounding_box_d{0, 0, bbox_size, bbox_size}},//
          {4, 1, kv::bounding_box_d{1280 - bbox_size - right_shift_pix, 0, 1280 - right_shift_pix, bbox_size}},//
          {4, 2, kv::bounding_box_d{1280 - bbox_size - right_shift_pix, 0, 1280 - right_shift_pix, bbox_size}},//
  };
}

inline tracks_pairing_from_stereo create_pairing() {
  tracks_pairing_from_stereo pairing;
  pairing.m_cameras_directory = std::string(TEST_DATA_DIR);
  pairing.load_camera_calibration();
  pairing.m_iou_pair_threshold = 0.05;
  pairing.m_detection_split_threshold = 3;
  pairing.m_verbose = true;
  return pairing;
}

inline detections_pairing_from_stereo create_detection_pairing() {
  detections_pairing_from_stereo pairing;
  pairing.m_cameras_directory = std::string(TEST_DATA_DIR);
  pairing.load_camera_calibration();
  pairing.m_iou_pair_threshold = 0.05;
  pairing.m_verbose = true;
  return pairing;
}

inline cv::Mat load_disparity_map() {
  return cv::imread(std::string(TEST_DATA_DIR) + "/test_depth_image.png", cv::IMREAD_GRAYSCALE);
}

inline cv::Mat create_uniform_image(int gray_value, const cv::Size &size = cv::Size{1280, 720}) {
  return {size, CV_8UC1, cv::Scalar(gray_value)};
}

inline kv::timestamp create_timestamp(int i_frame) {
  return kv::timestamp{i_frame, i_frame};
}

inline std::vector<kv::track_sptr> create_test_tracks(const std::vector<DetectionInfo> &detections_infos) {
  std::map<int, kv::track_sptr> tracks_map;
  std::vector<kv::track_sptr> tracks;
  for (const auto &info: detections_infos) {
    auto ts = create_timestamp(info.frame_id);
    if (tracks_map.find(info.track_id) == std::end(tracks_map)) {
      tracks_map[info.track_id] = kv::track::create();
      tracks.push_back(tracks_map[info.track_id]);
      tracks_map[info.track_id]->set_id(info.track_id);
    }

    auto track = tracks_map[info.track_id];
    auto detection = std::make_shared<kv::detected_object>(info.bbox, info.confidence);
    auto track_state = std::make_shared<kv::object_track_state>(ts, detection);
    track_state->set_frame(info.frame_id);
    track->append(track_state);
  }

  return tracks;
}

inline std::tuple<std::vector<kv::track_sptr>, std::vector<kv::track_sptr>, kv::timestamp> create_test_tracks() {
  return std::make_tuple(create_test_tracks(left_tracks_detections()), create_test_tracks(right_tracks_detections()),
                         create_timestamp(2));
}

TEST(TracksPairingFromStereoTest, test_util_sanity_check) {
  auto tracks_def_left = left_tracks_detections();
  auto [tracks_left, tracks_right, timestamp] = create_test_tracks();

  // Sanity check
  auto first_bbox = std::dynamic_pointer_cast<kv::object_track_state>(
      tracks_left[0]->front())->detection()->bounding_box();
  ASSERT_EQ(first_bbox, tracks_def_left[0].bbox);
}

TEST(TracksPairingFromStereoTest, smoke_test_pair_left_right_tracks) {
  auto tracks_def_left = left_tracks_detections();
  auto [tracks_left, tracks_right, timestamp] = create_test_tracks();
  auto cv_disparity_left = load_disparity_map();

  std::vector<std::string> pairing_methods{"PAIRING_3D", "PAIRING_IOU", "PAIRING_RECTIFIED_IOU"};
  for (const auto &method: pairing_methods) {
    // Create clean pairing
    auto pairing = create_pairing();

    // Estimate 3D positions in left image with disparity
    auto [left_tracks, left_3d_pos] = pairing.update_left_tracks_3d_position(tracks_left, cv_disparity_left, timestamp);
    ASSERT_EQ(left_tracks.size(), 2);
    ASSERT_EQ(left_tracks.size(), left_3d_pos.size());
    ASSERT_BBOX_NEAR(left_3d_pos[1].rectified_left_bbox, tracks_def_left[tracks_def_left.size() - 1].bbox, 100);

    // Filter tracks visible only in current timestamp
    auto right_tracks = pairing.keep_right_tracks_in_current_frame(tracks_right, timestamp);
    ASSERT_EQ(right_tracks.size(), 2);

    // Pair right and left tracks
    pairing.pair_left_right_tracks(left_tracks, left_3d_pos, right_tracks, timestamp);

    // Get updated pairs
    auto [updated_left_tracks, updated_right_tracks] = pairing.get_left_right_tracks_with_pairing();
    ASSERT_EQ(updated_left_tracks.size(), 2);
    ASSERT_EQ(updated_right_tracks.size(), 2);

    // Expect second track to have been paired with left track
    ASSERT_EQ(updated_left_tracks[0]->id(), updated_right_tracks[0]->id());
  }
}

TEST(TracksPairingFromStereoTest, can_be_called_with_obsolete_tracks) {
  auto pairing = create_pairing();

  auto tracks_def_left = left_tracks_detections();
  auto [tracks_left, tracks_right, timestamp] = create_test_tracks();
  auto cv_disparity_left = load_disparity_map();

  // Estimate 3D positions in left image with disparity
  pairing.update_left_tracks_3d_position(tracks_left, cv_disparity_left, timestamp);
  pairing.update_left_tracks_3d_position(tracks_left, cv_disparity_left, kv::timestamp(42, 42));
}

TEST(TracksPairingFromStereoTest, position_3d_estimation_returns_valid_bbox_value) {
  auto [tracks_left, tracks_right, timestamp] = create_test_tracks();
  auto pairing = create_detection_pairing();
  auto cv_disparity_left = load_disparity_map();
  auto cv_3d_pos_map = pairing.reproject_3d_depth_map(cv_disparity_left);

  auto state = std::dynamic_pointer_cast<kv::object_track_state>(tracks_left[1]->back());
  auto position = pairing.estimate_3d_position_from_detection(state->detection(), cv_3d_pos_map, true, 1. / 3.);

  // Check rectified left bbox is near the input
  auto tracks_def_left = left_tracks_detections();
  ASSERT_BBOX_NEAR(position.rectified_left_bbox, tracks_def_left[tracks_def_left.size() - 1].bbox, 100);

  // Verify the 3D coordinates are valid
  ASSERT_TRUE(pairing.point_is_valid(position.center3d.x, position.center3d.y, position.center3d.z));

  // Verify the projected bbox coordinates roughly match the right of the image
  auto tracks_def_right = right_tracks_detections();
  ASSERT_BBOX_NEAR(position.left_bbox_proj_to_right_image, tracks_def_right[tracks_def_left.size() - 1].bbox, 100);
}

TEST(TracksPairingFromStereoTest, unistort_point_preserves_global_image_corners) {
  auto pairing = create_detection_pairing();

  int tol = 100;
  EXPECT_NEAR(pairing.undistort_point({0, 0}, true).x, 0, tol);
  EXPECT_NEAR(pairing.undistort_point({0, 0}, true).y, 0, tol);

  EXPECT_NEAR(pairing.undistort_point({1280, 0}, true).x, 1280, tol);
  EXPECT_NEAR(pairing.undistort_point({1280, 0}, true).y, 0, tol);

  EXPECT_NEAR(pairing.undistort_point({1280, 720}, true).x, 1280, tol);
  EXPECT_NEAR(pairing.undistort_point({1280, 720}, true).y, 720, tol);
}

TEST(TracksPairingFromStereoTest, project_to_right_image_saturates_input_bbox_to_bounds) {
  auto pairing = create_detection_pairing();
  auto cv_disparity_left = load_disparity_map();
  auto cv_3d_pos_map = pairing.reproject_3d_depth_map(cv_disparity_left);

  ASSERT_BBOX_NEAR(pairing.project_to_right_image(kv::bounding_box_d{1200, 0, 1280, 100}, cv_3d_pos_map),
                   pairing.project_to_right_image(kv::bounding_box_d{1200, -50, 1400, 100}, cv_3d_pos_map), 2);
}

TEST(TracksPairingFromStereoTest, invalid_right_image_projection_returns_empty_bounding_box) {
  auto pairing = create_detection_pairing();
  auto cv_disparity_left = load_disparity_map();
  auto cv_3d_pos_map = pairing.reproject_3d_depth_map(cv_disparity_left);

  ASSERT_FALSE(pairing.project_to_right_image(kv::bounding_box_d{0, 650, 100, 720}, cv_3d_pos_map).is_valid());
}

inline void print(const cv::Vec3f &mat, const std::string &context) {
  std::cout << context << " {" << mat[0] << ", " << mat[1] << ", " << mat[2] << "}" << std::endl;
}

TEST(TracksPairingFromStereoTest, left_bounding_box_can_be_projected_to_right_image) {
  auto pairing = create_detection_pairing();
  auto cv_disparity_left = load_disparity_map();
  auto cv_3d_pos_map = pairing.reproject_3d_depth_map(cv_disparity_left);

  auto act = pairing.project_to_right_image(kv::bounding_box_d{1280 - bbox_size, 0, 1280, bbox_size}, cv_3d_pos_map);
  auto exp = kv::bounding_box_d{1280 - bbox_size - right_shift_pix, 0, 1280 - right_shift_pix, bbox_size};
  ASSERT_BBOX_NEAR(act, exp, 100);
}

TEST(TracksPairingFromStereoTest, tracks_not_fitting_detection_number_threshold_number_are_filtered_out) {
  auto pairing = create_pairing();

  kv::bounding_box_d a_bbox{0, 0, 1, 1};
  auto tracks = create_test_tracks(std::vector<DetectionInfo>{{42, 1, a_bbox},
                                                              {42, 2, a_bbox},
                                                              {2,  1, a_bbox},
                                                              {3,  1, a_bbox},
                                                              {3,  2, a_bbox},
                                                              {3,  3, a_bbox},
                                                              {3,  4, a_bbox},
                                                              {3,  5, a_bbox},
                                                              {58, 1, a_bbox},
                                                              {58, 2, a_bbox},
                                                              {58, 3, a_bbox},
                                                              {58, 4, a_bbox}});
  ASSERT_EQ(tracks.size(), 4);

  pairing.m_min_detection_number_threshold = 2;
  pairing.m_max_detection_number_threshold = 4;

  auto filtered = pairing.filter_tracks_with_threshold(tracks);
  ASSERT_EQ(filtered.size(), 2);
  ASSERT_EQ(filtered[0]->id(), 42);
  ASSERT_EQ(filtered[1]->id(), 58);
}

TEST(TracksPairingFromStereoTest, tracks_not_fitting_detection_threshold_number_are_filtered_out) {
  auto pairing = create_pairing();

  kv::bounding_box_d a_bbox{0, 0, 1, 1};
  auto tracks = create_test_tracks(std::vector<DetectionInfo>{{42, 1, {0, 0, 1, 100}},
                                                              {2,  1, {0, 0, 1, 99}},
                                                              {3,  1, {0, 0, 1, 201}},
                                                              {58, 1, {0, 0, 1, 200}}});
  ASSERT_EQ(tracks.size(), 4);

  pairing.m_min_detection_surface_threshold_pix = 100;
  pairing.m_max_detection_surface_threshold_pix = 200;


  auto filtered = pairing.filter_tracks_with_threshold(tracks);
  ASSERT_EQ(filtered.size(), 2);
  ASSERT_EQ(filtered[0]->id(), 42);
  ASSERT_EQ(filtered[1]->id(), 58);
}

TEST(TracksPairingFromStereoTest, split_pairing_for_continuous_pair_yields_one_split_range) {
  auto pairing = create_pairing();

  std::map<size_t, Pairing> left_to_right_pairing{{1, {{1, 2, 3, 8}, {1, 3}}}};
  auto ranges = pairing.create_split_ranges_from_track_pairs(left_to_right_pairing);
  ASSERT_EQ(1, ranges.size());
  ASSERT_EQ(4, ranges[0].detection_count);
  ASSERT_EQ(1, ranges[0].left_id);
  ASSERT_EQ(3, ranges[0].right_id);
  ASSERT_EQ(1, ranges[0].frame_id_first);
  ASSERT_GT(ranges[0].frame_id_last, 8);
}

TEST(TracksPairingFromStereoTest, split_pairing_for_inconclusive_pairs_yields_no_result) {
  auto pairing = create_pairing();

  std::map<size_t, Pairing> left_to_right_pairing{{1, {{1}, {1, 3}}},
                                                  {2, {{2}, {1, 4}}},
                                                  {3, {{3}, {1, 5}}}};
  auto ranges = pairing.create_split_ranges_from_track_pairs(left_to_right_pairing);
  ASSERT_EQ(0, ranges.size());
}


TEST(TracksPairingFromStereoTest, split_pairing_for_separate_pairs_keeps_both_ranges) {
  auto pairing = create_pairing();

  std::map<size_t, Pairing> left_to_right_pairing{{1, {{1, 2, 3, 4, 5}, {1, 3}}},
                                                  {2, {{1, 2, 3, 4, 5}, {2, 6}}}};
  auto ranges = pairing.create_split_ranges_from_track_pairs(left_to_right_pairing);
  ASSERT_EQ(2, ranges.size());
  ASSERT_EQ(1, ranges[0].left_id);
  ASSERT_EQ(3, ranges[0].right_id);
  ASSERT_GT(ranges[0].frame_id_last, 5);
  ASSERT_EQ(2, ranges[1].left_id);
  ASSERT_EQ(6, ranges[1].right_id);
  ASSERT_GT(ranges[1].frame_id_last, 5);
}


TEST(TracksPairingFromStereoTest, split_pairing_for_conflicting_pairs_creates_multiple_ranges) {
  auto pairing = create_pairing();

  std::map<size_t, Pairing> left_to_right_pairing{{1, {{1, 2, 3, 7, 8, 9}, {1, 3}}},
                                                  {2, {{4, 5, 6},          {1, 6}}}};

  auto ranges = pairing.create_split_ranges_from_track_pairs(left_to_right_pairing);
  ASSERT_EQ(3, ranges.size());
  ASSERT_EQ(1, ranges[0].left_id);
  ASSERT_EQ(3, ranges[0].right_id);
  ASSERT_EQ(3, ranges[0].frame_id_last);
  ASSERT_EQ(1, ranges[1].left_id);
  ASSERT_EQ(6, ranges[1].right_id);
  ASSERT_EQ(6, ranges[1].frame_id_last);
  ASSERT_EQ(1, ranges[2].left_id);
  ASSERT_EQ(3, ranges[2].right_id);
  ASSERT_GT(ranges[2].frame_id_last, 9);
}


TEST(TracksPairingFromStereoTest, split_pairing_with_continuous_pair_and_interleaved_inconclusive_yields_one_range) {
  auto pairing = create_pairing();

  std::map<size_t, Pairing> left_to_right_pairing{{1, {{1, 2, 3, 5, 7, 9}, {1, 3}}},
                                                  {2, {{4},                {1, 6}}},
                                                  {3, {{6},                {1, 7}}}};

  auto ranges = pairing.create_split_ranges_from_track_pairs(left_to_right_pairing);
  ASSERT_EQ(1, ranges.size());
  ASSERT_EQ(1, ranges[0].left_id);
  ASSERT_EQ(3, ranges[0].right_id);
  ASSERT_GT(ranges[0].frame_id_last, 9);
}

TEST(TracksPairingFromStereoTest, split_pairing_first_split_id_takes_closed_into_account) {
  auto pairing = create_pairing();

  std::map<size_t, Pairing> left_to_right_pairing{{1, {{1, 2, 3}, {1, 1}}},
                                                  {2, {{4, 5, 6}, {1, 6}}},
                                                  {3, {{7, 8, 9}, {6, 1}}}};

  auto ranges = pairing.create_split_ranges_from_track_pairs(left_to_right_pairing);
  ASSERT_EQ(3, ranges.size());
  ASSERT_EQ(1, ranges[0].frame_id_first);
  ASSERT_EQ(4, ranges[1].frame_id_first);
  ASSERT_EQ(7, ranges[2].frame_id_first);
}

inline kv::detected_object_sptr create_detection(const kv::bounding_box_d &bbox, bool do_use_mask = false) {
  auto detection = std::make_shared<kv::detected_object>(bbox, 1.0);
  if (do_use_mask) {
    auto mask_image = create_uniform_image(255, cv::Size((int) bbox.width(), (int) bbox.height()));
    using ic = kwiver::arrows::ocv::image_container;
    auto vital_img = std::make_shared<kwiver::vital::simple_image_container>(
        ic::ocv_to_vital(mask_image, ic::ColorMode::OTHER_COLOR));
    detection->set_mask(vital_img);
  }

  return std::make_shared<kv::detected_object>(bbox, 1.0);
}

TEST(TracksPairingFromStereoTest, threed_coordinates_are_consistent) {
  // Load detection pairing
  auto pairing = create_detection_pairing();

  // Create 4 corner detections based on 10 pix square
  double left = 240;
  double right = 1280;
  double top = 0;
  double bottom = 720;
  double square_size = 10;

  for (size_t i_mask = 0; i_mask < 2; i_mask++) {
    auto do_use_mask = static_cast<bool>(i_mask);

    auto top_left = create_detection({left, top, left + square_size, top + square_size}, do_use_mask);
    auto top_right = create_detection({right - square_size, top, right, top + square_size}, do_use_mask);
    auto bottom_left = create_detection({left, bottom - square_size, left + square_size, bottom}, do_use_mask);
    auto bottom_right = create_detection({right - square_size, bottom - square_size, right, bottom}, do_use_mask);

    // Process 3D positions in undistorted referential
    auto pos_3d_map = pairing.reproject_3d_depth_map(create_uniform_image(160));

    std::vector<viame::core::Detections3DPositions> positions;
    const float bbox_crop_ratio = 1;
    const bool do_undistort_points = false;
    for (const auto &detection: {top_left, top_right, bottom_left, bottom_right}) {
      positions.emplace_back(
          pairing.estimate_3d_position_from_detection(detection, pos_3d_map, do_undistort_points, bbox_crop_ratio));
    }

    // Assert every position is valid and max score (uniform disparity map)
    for (const auto &position: positions) {
      ASSERT_TRUE(position.is_valid());
      ASSERT_EQ(position.score, 1);
    }

    // Check coordinates orientation (X positive left to right, Y positive screen up to bottom)
    EXPECT_LT(positions[0].center3d.x, positions[1].center3d.x);
    EXPECT_NEAR(positions[0].center3d.x, -positions[1].center3d.x, .1);
    EXPECT_LT(positions[0].center3d.y, positions[2].center3d.y);
    EXPECT_NEAR(positions[0].center3d.y, -positions[2].center3d.y, .1);

    // Check measured screen FOV distances
    // image fov width ~ 1040 pix, image fov height = 720 pix
    // exp sardine width pix at 160 grayscale distance ~ 400 pix
    // 11cm < Sardine width < 16cm
    // 28cm < FOV width cm < 42cm
    // 18cm < FOV height cm < 29cm
    EXPECT_NEAR(std::abs(positions[0].center3d.x - positions[1].center3d.x), .35, .07);
    EXPECT_NEAR(std::abs(positions[0].center3d.y - positions[2].center3d.y), .235, .055);

    // Check measured depth. Expects Z direction to be consistent and depth between near to fishing net to be at least
    // size of a sardine.
    auto near_3d_map = pairing.reproject_3d_depth_map(create_uniform_image(255));
    auto far_3d_map = pairing.reproject_3d_depth_map(create_uniform_image(150));

    auto near_position = pairing.estimate_3d_position_from_bbox(top_left->bounding_box(), near_3d_map, bbox_crop_ratio,
                                                                do_undistort_points);
    auto far_position = pairing.estimate_3d_position_from_bbox(top_left->bounding_box(), far_3d_map, bbox_crop_ratio,
                                                               do_undistort_points);
    EXPECT_LT(near_position.center3d.z, far_position.center3d.z);
    EXPECT_GT(std::abs(near_position.center3d.z - far_position.center3d.z), .15);
  }
}


TEST(TracksPairingFromStereoTest, undistorted_coordinates_are_consistent) {
  // Check examples of coordinates given the camera parameters in data dir
  // Source and destination coordinates were measured on GIMP manually (hence 10 pix of tolerance are required)

  double tol_pix = 10;

  auto pairing = create_detection_pairing();

  bool is_left_image = true;
  std::vector<cv::Point2d> left_points{{833,  452},//
                                       {1162, 337},//
                                       {1111, 69}};
  std::vector<cv::Point2d> exp_points{{885,  473},//
                                      {1224, 335},//
                                      {1164, 38}};

  auto undistorted = pairing.undistort_point(left_points, is_left_image);

  ASSERT_EQ(undistorted.size(), exp_points.size());
  for (size_t i_pt = 0; i_pt < exp_points.size(); i_pt++) {
    EXPECT_NEAR(undistorted[i_pt].x, exp_points[i_pt].x, tol_pix);
    EXPECT_NEAR(undistorted[i_pt].y, exp_points[i_pt].y, tol_pix);
  }
}
