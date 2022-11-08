#include <gtest/gtest.h>

#include <plugins/core/tracks_pairing_from_stereo.h>
#include <vital/types/track.h>
#include <vital/types/timestamp.h>

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
}

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
  pairing.m_iou_merge_threshold = 0.5;
  pairing.m_iou_pair_threshold = 0.05;
  return pairing;
}

inline cv::Mat load_disparity_map() {
  return cv::imread(std::string(TEST_DATA_DIR) + "/test_depth_image.png", cv::IMREAD_GRAYSCALE);
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

TEST(TracksPairingFromStereoTest, smoke_test_bounding_box_pairing) {
  auto pairing = create_pairing();

  auto tracks_def_left = left_tracks_detections();
  auto [tracks_left, tracks_right, timestamp] = create_test_tracks();
  auto cv_disparity_left = load_disparity_map();

  // Estimate 3D positions in left image with disparity
  auto [left_tracks, left_3d_pos] = pairing.update_left_tracks_3d_position(tracks_left, cv_disparity_left, timestamp);
  ASSERT_EQ(left_tracks.size(), 2);
  ASSERT_EQ(left_tracks.size(), left_3d_pos.size());
  ASSERT_BBOX_NEAR(left_3d_pos[1].rectified_left_bbox, tracks_def_left[tracks_def_left.size() - 1].bbox, 100);

  // Filter tracks visible only in current timestamp
  auto right_tracks = pairing.keep_right_tracks_in_current_frame(tracks_right, timestamp);
  ASSERT_EQ(right_tracks.size(), 2);

  // Pair right and left tracks
  pairing.pair_left_right_tracks_using_bbox(left_tracks, left_3d_pos, right_tracks, timestamp);

  // Get updated pairs
  auto [updated_left_tracks, updated_right_tracks] = pairing.get_left_right_tracks_with_pairing();
  ASSERT_EQ(updated_left_tracks.size(), 2);
  ASSERT_EQ(updated_right_tracks.size(), 2);

  // Expect second track to have been paired with left track
  ASSERT_EQ(updated_left_tracks[0]->id(), updated_right_tracks[0]->id());
}


TEST(TracksPairingFromStereoTest, smoke_test_3d_center_pairing) {
  auto pairing = create_pairing();

  auto tracks_def_left = left_tracks_detections();
  auto [tracks_left, tracks_right, timestamp] = create_test_tracks();
  auto cv_disparity_left = load_disparity_map();

  // Estimate 3D positions in left image with disparity
  auto [left_tracks, left_3d_pos] = pairing.update_left_tracks_3d_position(tracks_left, cv_disparity_left, timestamp);
  ASSERT_EQ(left_tracks.size(), 2);
  ASSERT_EQ(left_tracks.size(), left_3d_pos.size());
  ASSERT_BBOX_NEAR(left_3d_pos[1].rectified_left_bbox, tracks_def_left[tracks_def_left.size() - 1].bbox, 100);

  // Filter tracks visible only in current timestamp
  auto right_tracks = pairing.keep_right_tracks_in_current_frame(tracks_right, timestamp);
  ASSERT_EQ(right_tracks.size(), 2);

  // Pair right and left tracks
  pairing.pair_left_right_tracks_using_3d_center(left_tracks, left_3d_pos, right_tracks, timestamp);

  // Get updated pairs
  auto [updated_left_tracks, updated_right_tracks] = pairing.get_left_right_tracks_with_pairing();
  ASSERT_EQ(updated_left_tracks.size(), 2);
  ASSERT_EQ(updated_right_tracks.size(), 2);

  // Expect second track to have been paired with left track
  ASSERT_EQ(updated_left_tracks[0]->id(), updated_right_tracks[0]->id());
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

TEST(TracksPairingFromStereoTest, track_group_returns_one_per_track_given_non_overlapping) {
  auto pairing = create_pairing();

  auto [tracks_left, tracks_right, timestamp] = create_test_tracks();
  auto cv_disparity_left = load_disparity_map();

  // Estimate 3D positions in left image with disparity
  auto [left_tracks, left_3d_pos] = pairing.update_left_tracks_3d_position(tracks_left, cv_disparity_left, timestamp);
  auto left_tracks_clusters_ids = pairing.group_overlapping_tracks_indexes_in_current_frame(left_tracks, timestamp);
  auto left_tracks_clusters = pairing.group_vector_by_ids(left_tracks, left_tracks_clusters_ids);
  auto left_clusters_3dpos = pairing.group_vector_by_ids(left_3d_pos, left_tracks_clusters_ids);
  ASSERT_EQ(left_tracks_clusters_ids.size(), 2);
  ASSERT_EQ(left_tracks_clusters.size(), left_tracks_clusters_ids.size());
  ASSERT_EQ(left_clusters_3dpos.size(), left_tracks_clusters_ids.size());
}

TEST(TracksPairingFromStereoTest, position_3d_estimation_returns_valid_bbox_value) {
  auto [tracks_left, tracks_right, timestamp] = create_test_tracks();
  auto pairing = create_pairing();
  auto cv_disparity_left = load_disparity_map();
  auto cv_3d_pos_map = pairing.reproject_3d_depth_map(cv_disparity_left);

  auto state = std::dynamic_pointer_cast<kv::object_track_state>(tracks_left[1]->back());
  auto position = pairing.estimate_3d_position_from_detection(state->detection(), cv_3d_pos_map);

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
  auto pairing = create_pairing();

  int tol = 100;
  EXPECT_NEAR(pairing.undistort_point({0, 0}).x, 0, tol);
  EXPECT_NEAR(pairing.undistort_point({0, 0}).y, 0, tol);

  EXPECT_NEAR(pairing.undistort_point({1280, 0}).x, 1280, tol);
  EXPECT_NEAR(pairing.undistort_point({1280, 0}).y, 0, tol);

  EXPECT_NEAR(pairing.undistort_point({1280, 720}).x, 1280, tol);
  EXPECT_NEAR(pairing.undistort_point({1280, 720}).y, 720, tol);
}

TEST(TracksPairingFromStereoTest, project_to_right_image_saturates_input_bbox_to_bounds) {
  auto pairing = create_pairing();
  auto cv_disparity_left = load_disparity_map();
  auto cv_3d_pos_map = pairing.reproject_3d_depth_map(cv_disparity_left);

  ASSERT_BBOX_NEAR(pairing.project_to_right_image(kv::bounding_box_d{1200, 0, 1280, 100}, cv_3d_pos_map),
                   pairing.project_to_right_image(kv::bounding_box_d{1200, -50, 1400, 100}, cv_3d_pos_map), 2);
}

TEST(TracksPairingFromStereoTest, invalid_right_image_projection_returns_empty_bounding_box) {
  auto pairing = create_pairing();
  auto cv_disparity_left = load_disparity_map();
  auto cv_3d_pos_map = pairing.reproject_3d_depth_map(cv_disparity_left);

  ASSERT_FALSE(pairing.project_to_right_image(kv::bounding_box_d{0, 650, 100, 720}, cv_3d_pos_map).is_valid());
}

inline void print(const cv::Vec3f &mat, const std::string &context) {
  std::cout << context << " {" << mat[0] << ", " << mat[1] << ", " << mat[2] << "}" << std::endl;
}

TEST(TracksPairingFromStereoTest, left_bounding_box_can_be_projected_to_right_image) {
  auto pairing = create_pairing();
  auto cv_disparity_left = load_disparity_map();
  auto cv_3d_pos_map = pairing.reproject_3d_depth_map(cv_disparity_left);

  auto act = pairing.project_to_right_image(kv::bounding_box_d{1280 - bbox_size, 0, 1280, bbox_size}, cv_3d_pos_map);
  auto exp = kv::bounding_box_d{1280 - bbox_size - right_shift_pix, 0, 1280 - right_shift_pix, bbox_size};
  ASSERT_BBOX_NEAR(act, exp, 100);
}

