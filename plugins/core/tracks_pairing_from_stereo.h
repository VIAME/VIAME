#ifndef VIAME_TRACKS_PAIRING_FROM_STEREO_H
#define VIAME_TRACKS_PAIRING_FROM_STEREO_H

#include <sprokit/pipeline/process.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vital/types/bounding_box.h>
#include <vital/types/track_set.h>
#include <vital/types/detected_object_set.h>

#include <vital/vital_types.h>
#include <vital/types/image_container.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/object_track_set.h>
#include <vital/types/bounding_box.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <arrows/ocv/image_container.h>
#include <sprokit/processes/kwiver_type_traits.h>
#include <plugins/core/viame_core_export.h>

#include <memory>

namespace viame {
namespace core {

/// @brief Structure containing extracted 3D information for each track in left image and mapping to right track
/// coordinates
struct VIAME_CORE_EXPORT Tracks3DPositions {
  cv::Point3f center3d{};
  cv::Point2f center3d_proj_to_right_image{};
  kwiver::vital::bounding_box_d rectified_left_bbox{};
  kwiver::vital::bounding_box_d left_bbox_proj_to_right_image{};
  float score{};

  /// @brief Validity status for the 3D track position
  /// Considered valid if at least one point used to process position center is valid
  bool is_valid() const { return score > 1e-6f; }
};


/// @brief Class responsible for the tracks stereo pairing logic
/// Uses camera calibration information, left and right tracks and disparity map to find corresponding tracks from left
/// To right.
///
/// Instance keeps track of the different tracks seen in left and right camera.
/// Otherwise, class is meant to be used as a toolkit for pairing the two track feeds.
class VIAME_CORE_EXPORT tracks_pairing_from_stereo {
public:
  tracks_pairing_from_stereo() = default;

  // Configuration settings
  std::string m_cameras_directory;
  kwiver::vital::logger_handle_t m_logger;
  float m_iou_merge_threshold{.5};
  float m_iou_pair_threshold{.05};
  int m_min_detection_number_threshold{0};
  int m_max_detection_number_threshold{std::numeric_limits<int>::max()};
  int m_min_detection_surface_threshold_pix{0};
  int m_max_detection_surface_threshold_pix{std::numeric_limits<int>::max()};


  // Camera depth information
  cv::Mat m_Q, m_K1, m_D1, m_R1, m_P1, m_K2, m_D2, m_R2, m_P2, m_R, m_Rvec, m_T;

  // Tracks status memo
  std::map<kwiver::vital::track_id_t, kwiver::vital::track_sptr> m_tracks_with_3d_left, m_right_tracks_memo;

  // Track ID pairing
  std::map<kwiver::vital::track_id_t, kwiver::vital::track_id_t> m_left_to_right_pairing;

  /// @brief Project depth map as 3 channel 3D image
  cv::Mat reproject_3d_depth_map(const cv::Mat &cv_disparity_left) const;

  /// @brief Load matrix calibration from settings camera directory
  void load_camera_calibration();

  /// @brief Compute median of input vector of values. Returns 0 if empty.
  static float compute_median(std::vector<float> values, bool is_sorted = false);

  /// @brief Convert input kwiver bounding box to OpenCV Rect format
  static cv::Rect bbox_to_mask_rect(kwiver::vital::bounding_box_d const &bbox);
  static kwiver::vital::bounding_box_d mask_rect_to_bbox(const cv::Rect &rect);

  /// @brief Get mask from input detection object
  ///     Copied from kwiver/arrows/ocv/refine_detections_util.cxx to include bbox used for the mask
  static std::tuple<cv::Mat, cv::Rect> get_standard_mask(kwiver::vital::detected_object_sptr const &det);

  /// @brief Estimate 3D position for input detection. If the detection contains a mask and that mask is contained
  /// within the left camera image, calculates position given mask bounds.
  /// Otherwise, returns detection for the input detection bounding box.
  Tracks3DPositions estimate_3d_position_from_detection(const kwiver::vital::detected_object_sptr &detection,
                                                        const cv::Mat &pos_3d_map) const;

  /// @brief Rectify bounding box positions given loaded camera properties
  kwiver::vital::bounding_box_d get_rectified_bbox(const kwiver::vital::bounding_box_d &bbox) const;

  /// @brief Rectify bounding box positions using OpenCV Rect
  cv::Rect get_rectified_bbox(const cv::Rect &bbox) const;

  /// @brief Undistort input point coordinate in left camera
  cv::Point2d undistort_point(const cv::Point2d &point) const;
  std::vector<cv::Point2d> undistort_point(const std::vector<cv::Point2d> &point) const;

  static bool point_is_valid(float x, float y, float z);
  static bool point_is_valid(const cv::Vec3f &pt);

  /// @brief Estimate the 3D position from given detection bounding box
  Tracks3DPositions
  estimate_3d_position_from_bbox(const kwiver::vital::bounding_box_d &bbox, const cv::Mat &pos_3d_map) const;

  /// @brief Estimate position given input detection mask
  Tracks3DPositions
  estimate_3d_position_from_mask(const cv::Rect &bbox, const cv::Mat &pos_3d_map, const cv::Mat &mask) const;

  static kwiver::vital::bounding_box_d bounding_box_cv_to_kv(const cv::Rect &bbox);

  Tracks3DPositions
  create_3d_position(const std::vector<float> &xs, const std::vector<float> &ys, const std::vector<float> &zs,
                     const kwiver::vital::bounding_box_d &bbox, const cv::Mat &pos_3d_map, float score) const;

  kwiver::vital::bounding_box_d
  project_to_right_image(const kwiver::vital::bounding_box_d &bbox, const cv::Mat &pos_3d_map) const;

  kwiver::vital::bounding_box_d project_to_right_image(const std::vector<cv::Vec3f> &points_3d) const;
  cv::Point2f project_to_right_image(const cv::Point3f &points_3d) const;

  /// @brief Update 3D tracks positions given a list of tracks and tracks disparity map
  std::tuple<std::vector<kwiver::vital::track_sptr>, std::vector<viame::core::Tracks3DPositions>>
  update_left_tracks_3d_position(const std::vector<kwiver::vital::track_sptr> &tracks, const cv::Mat &cv_disparity_map,
                                 const kwiver::vital::timestamp &timestamp);

  std::vector<kwiver::vital::track_sptr>
  keep_right_tracks_in_current_frame(const std::vector<kwiver::vital::track_sptr> &tracks,
                                     const kwiver::vital::timestamp &timestamp);

  /// @brief Calculates intersection over union of two tracks bounding box
  static double iou_distance(const std::shared_ptr<kwiver::vital::object_track_state> &t1,
                             const std::shared_ptr<kwiver::vital::object_track_state> &t2);

  /// @brief Calculates intersection over union for two bounding boxes
  static double iou_distance(const kwiver::vital::bounding_box_d &bbox1, const kwiver::vital::bounding_box_d &bbox2);


  std::vector<std::vector<size_t>>
  group_overlapping_tracks_indexes_in_current_frame(const std::vector<kwiver::vital::track_sptr> &tracks,
                                                    const kwiver::vital::timestamp &timestamp) const;

  template<typename T>
  static std::vector<std::vector<T>>
  group_vector_by_ids(const std::vector<T> &vector, const std::vector<std::vector<size_t>> &group_ids) {
    std::vector<std::vector<T>> groups;
    for (const auto &ids: group_ids) {
      std::vector<T> group;
      for (const auto &id: ids) {
        group.emplace_back(vector[id]);
      }
      groups.emplace_back(group);
    }
    return groups;
  }

  /// @brief Group overlapping tracks by IOU of last detection in given input track pointer.
  std::vector<std::vector<kwiver::vital::track_sptr>>
  group_overlapping_tracks_in_current_frame(const std::vector<kwiver::vital::track_sptr> &tracks,
                                            const kwiver::vital::timestamp &timestamp) const;

  /// @brief Union of all the bounding boxes defined in input clustered tracks.
  /// Stereo rectifies the bounding boxes before merging them
  static std::vector<kwiver::vital::bounding_box_d>
  merge_clustered_bbox(const std::vector<std::vector<kwiver::vital::track_sptr>> &clusters);

  static std::vector<kwiver::vital::bounding_box_d>
  merge_clustered_bbox(const std::vector<std::vector<kwiver::vital::bounding_box_d>> &clusters);

  static std::vector<kwiver::vital::bounding_box_d>
  merge_3d_left_projected_to_right_bbox(const std::vector<std::vector<Tracks3DPositions>> &clusters);

  kwiver::vital::track_id_t most_likely_paired_right_cluster(const kwiver::vital::bounding_box_d &bbox,
                                                             const std::vector<kwiver::vital::bounding_box_d> &other_bboxs) const;

  /// @brief Pair left and right clusters using IOU between each cluster
  /// IOU is processed using stereo rectification
  std::vector<std::tuple<std::vector<kwiver::vital::track_sptr>, std::vector<kwiver::vital::track_sptr>, std::vector<Tracks3DPositions>>>
  pair_left_right_clusters(const std::vector<std::vector<kwiver::vital::track_sptr>> &left_cluster,
                           const std::vector<std::vector<kwiver::vital::track_sptr>> &right_cluster,
                           const std::vector<std::vector<Tracks3DPositions>> &left_3d_pos) const;

  void pair_left_right_tracks_in_each_cluster(
      const std::vector<std::tuple<std::vector<kwiver::vital::track_sptr>, std::vector<kwiver::vital::track_sptr>, std::vector<Tracks3DPositions>>> &left_right_clusters);

  void pair_left_right_tracks_in_each_cluster(const std::vector<kwiver::vital::track_sptr> &left_tracks,
                                              const std::vector<kwiver::vital::track_sptr> &right_tracks,
                                              const std::vector<Tracks3DPositions> &left_positions);

  /// @brief returns last track id available in both left and right track map
  kwiver::vital::track_id_t last_left_right_track_id() const;

  /// @brief Update left and right tracks pairs using left and right bounding boxes
  ///     - Projects left 3D bounding box to right image
  ///     - Groups left and right overlapping bounding boxes
  ///     - Cluster corresponding left and right groups
  ///     - In each group, find best match between left and right track
  void pair_left_right_tracks_using_bbox(const std::vector<kwiver::vital::track_sptr> &left_tracks,
                                         const std::vector<viame::core::Tracks3DPositions> &left_3d_pos,
                                         const std::vector<kwiver::vital::track_sptr> &right_tracks,
                                         const kwiver::vital::timestamp &timestamp);


  /// @brief Update left and right tracks pairs using left 3D coordinates and right bounding boxes
  ///     - Project left 3D center coordinate to right image
  ///     - For each right bounding box containing projected left center, find closest
  void pair_left_right_tracks_using_3d_center(const std::vector<kwiver::vital::track_sptr> &left_tracks,
                                              const std::vector<viame::core::Tracks3DPositions> &left_3d_pos,
                                              const std::vector<kwiver::vital::track_sptr> &right_tracks,
                                              const kwiver::vital::timestamp &timestamp);

  /// @brief Apply pairing to dict of left and right tracks and pair left and right tracks given accumulated information
  std::tuple<std::vector<kwiver::vital::track_sptr>, std::vector<kwiver::vital::track_sptr>>
  get_left_right_tracks_with_pairing();

  /// @brief Return true if track is already paired to a right track
  bool is_left_track_paired(const kwiver::vital::track_sptr &left_track) const;

  /// @brief Return true if right track is already paired to a left track
  bool is_right_track_paired(const kwiver::vital::track_sptr &right_track) const;

  /// @brief Remove tracks in input list that don't match the min / max detection and min / max surface thresholds
  std::vector<kwiver::vital::track_sptr>
  filter_tracks_with_threshold(std::vector<kwiver::vital::track_sptr> tracks) const;

  /// @brief Cantor pairing function mapping N x N -> N
  /// Allow to map pairs of left / right ids to one unique natural integer
  static size_t cantor_pairing(size_t i, size_t j){
    return (1u / 2u) * (i + j) * (i + j + 1) + j;
  }
};

} // core
} // viame

#endif // VIAME_TRACKS_PAIRING_FROM_STEREO_H