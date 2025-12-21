#ifndef VIAME_OCV_PAIR_STEREO_DETECTIONS_H
#define VIAME_OCV_PAIR_STEREO_DETECTIONS_H

#include <vital/types/bounding_box.h>
#include <vital/types/detected_object.h>
#include <vital/vital_types.h>

#include <opencv2/core/core.hpp>

#include <plugins/opencv/viame_opencv_export.h>

#include <memory>

namespace viame {

/// @brief Structure containing extracted 3D information for each track in left image and mapping to right track
/// coordinates
struct VIAME_OPENCV_EXPORT Detections3DPositions
{
  cv::Point3f center3d{};
  cv::Point2f center3d_proj_to_right_image{};
  kwiver::vital::bounding_box_d rectified_left_bbox{};
  kwiver::vital::bounding_box_d left_bbox_proj_to_right_image{};
  float score{};

  /// @brief Validity status for the 3D track position
  /// Considered valid if at least one point used to process position center is valid
  bool is_valid() const { return score > 1e-6f; }
};


/// @brief Class responsible for the detection stereo pairing logic
/// Uses camera calibration information, left and right tracks and disparity map to find corresponding detection from
/// left to right.
class VIAME_OPENCV_EXPORT ocv_pair_stereo_detections
{
public:
  ocv_pair_stereo_detections() = default;

  // Configuration settings
  std::string m_cameras_directory;
  float m_iou_pair_threshold{ 0.1f };
  std::string m_pairing_method{ "PAIRING_3D" };
  bool m_verbose{}; // Set to true to activate debug print

  // Camera depth information
  cv::Mat m_Q, m_K1, m_D1, m_R1, m_P1, m_K2, m_D2, m_R2, m_P2, m_R, m_Rvec, m_T;

  /// @brief Project depth map as 3 channel 3D image
  cv::Mat reproject_3d_depth_map( const cv::Mat& cv_disparity_left ) const;

  /// @brief Load matrix calibration from settings camera directory
  void load_camera_calibration();

  /// @brief Compute median of input vector of values. Returns 0 if empty.
  static float compute_median( std::vector< float > values, bool is_sorted = false );

  /// @brief Convert input kwiver bounding box to OpenCV Rect format
  static cv::Rect bbox_to_mask_rect( kwiver::vital::bounding_box_d const& bbox );
  static kwiver::vital::bounding_box_d mask_rect_to_bbox( const cv::Rect& rect );

  /// @brief Get mask from input detection object
  ///     Copied from kwiver/arrows/ocv/refine_detections_util.cxx to include bbox used for the mask
  static cv::Mat get_standard_mask( kwiver::vital::detected_object_sptr const& det );

  /// @brief Estimate 3D position for input detection. If the detection contains a mask and that mask is contained
  /// within the left camera image, calculates position given mask bounds.
  /// Otherwise, returns detection for the input detection bounding box cropped to given bbox crop ratio.
  /// Set bbox_crop_ratio to 1.0 to use full bbox for 3D estimation.
  viame::Detections3DPositions
  estimate_3d_position_from_detection( const kwiver::vital::detected_object_sptr& detection,
                                       const cv::Mat& pos_3d_map,
                                       bool do_undistort_points,
                                       float bbox_crop_ratio ) const;

  /// @brief Rectify bounding box positions given loaded camera properties
  kwiver::vital::bounding_box_d get_rectified_bbox( const kwiver::vital::bounding_box_d& bbox,
                                                    bool is_left_image ) const;

  /// @brief Undistort input point coordinate
  cv::Point2d undistort_point( const cv::Point2d& point, bool is_left_image ) const;

  /// @brief Unistort input point coordinates
  std::vector< cv::Point2d > undistort_point( const std::vector< cv::Point2d >& point,
                                              bool is_left_image ) const;

  /// @return True if 3D point is valid (not infinite and Z positive)
  static bool point_is_valid( float x, float y, float z );

  /// @return True if 3D point is valid (not infinite and Z positive)
  static bool point_is_valid( const cv::Vec3f& pt );

  /// @brief Estimates 3D positions from input Bounding box and 3D map.
  /// @param bbox: Bounding box in left image on which to extract the 3D position
  /// @param pos_3d_map: 3D coordinate image with the same dimensions as the left image
  /// @param crop_ratio: [0 - 1] Defines the amount of the bounding box to use for the estimate.
  ///     Set to 1.f to use full bounding box.
  /// @param do_undistort_points: If true, will undistort bounding box corners by camera intrinsic. Otherwise, will use
  ///     provided bounding box coordinates directly.
  viame::Detections3DPositions
  estimate_3d_position_from_bbox( const kwiver::vital::bounding_box_d& bbox,
                                  const cv::Mat& pos_3d_map,
                                  float crop_ratio,
                                  bool do_undistort_points ) const;

  /// @brief Estimates 3D positions from input mask and 3D map.
  /// @param bbox: Bounding box in left image matching the mask
  /// @param pos_3d_map: 3D coordinate image with the same dimensions as the left image
  /// @param mask: Mask image with the same size as the bbox size. Values at 0 will be ignored and values > 0 will be
  ///     used for the 3D position evaluation.
  /// @param do_undistort_points: If true, will undistort mask coordinates by camera intrinsic. Otherwise, will use
  ///     provided mask coordinates directly.
  viame::Detections3DPositions
  estimate_3d_position_from_unrectified_mask( const kwiver::vital::bounding_box_d& bbox,
                                              const cv::Mat& pos_3d_map,
                                              const cv::Mat& mask,
                                              bool do_undistort_points ) const;


  /// @brief Estimates 3D positions from input undistorted coordinates and 3D map.
  /// @param rectified_bbox: Bounding box in left image matching the mask coordinates
  /// @param undistorted_mask_coords: Undistorted mask coordinates
  /// @param pos_3d_map: 3D coordinate image with the same dimensions as the left image
  Detections3DPositions
  estimate_3d_position_from_point_coordinates( const kwiver::vital::bounding_box_d& rectified_bbox,
                                               const std::vector< cv::Point2d >& undistorted_mask_coords,
                                               const cv::Mat& pos_3d_map ) const;

  /// @brief Creates 3D position from input X, Y, Z vectors.
  Detections3DPositions
  create_3d_position( const std::vector< float >& xs,
                      const std::vector< float >& ys,
                      const std::vector< float >& zs,
                      const kwiver::vital::bounding_box_d& bbox,
                      const cv::Mat& pos_3d_map,
                      float score ) const;

  /// @brief Projects the input left BBox to right image using 3D coordinates
  kwiver::vital::bounding_box_d
  project_to_right_image( const kwiver::vital::bounding_box_d& bbox,
                          const cv::Mat& pos_3d_map ) const;

  /// @brief Projects the input left coordinates to right image using 3D coordinates
  kwiver::vital::bounding_box_d project_to_right_image( const std::vector< cv::Vec3f >& points_3d ) const;

  /// @brief Projects the input left coordinates to right image using 3D coordinates
  cv::Point2f project_to_right_image( const cv::Point3f& points_3d ) const;

  /// @brief Update 3D tracks positions given a list of tracks and tracks disparity map
  std::vector< viame::Detections3DPositions >
  update_left_detections_3d_positions( const std::vector< kwiver::vital::detected_object_sptr >& detections,
                                       const cv::Mat& cv_disparity_map ) const;

  viame::Detections3DPositions
  update_left_detection_3d_position( const kwiver::vital::detected_object_sptr& detection,
                                     const cv::Mat& cv_pos_3d_map ) const;


  /// @brief Calculates intersection over union for two bounding boxes
  static double iou_distance( const kwiver::vital::bounding_box_d& bbox1,
                              const kwiver::vital::bounding_box_d& bbox2 );


  /// @brief Update left and right tracks pairs using left 3D coordinates and right bounding boxes
  ///     - Project left 3D center coordinate to right image
  ///     - For each right bounding box containing projected left center, find closest
  static std::vector< std::vector< size_t > >
  pair_left_right_detections_using_3d_center(
    const std::vector< kwiver::vital::detected_object_sptr >& left_detections,
    const std::vector< viame::Detections3DPositions >& left_3d_pos,
    const std::vector< kwiver::vital::detected_object_sptr >& right_detections );

  /// @brief Update left and right tracks pairs using left and right bounding boxes
  ///     - For each left / right tracks, finds the pair having the highest IOU
  std::vector< std::vector< size_t > >
  pair_left_right_tracks_using_bbox_iou(
    const std::vector< kwiver::vital::detected_object_sptr >& left_detections,
    const std::vector< kwiver::vital::detected_object_sptr >& right_detections,
    bool do_rectify_bbox );

  /// @brief Update left and right tracks pairs using the currently set pairing method.
  ///     If pairing is set to 3D, uses @ref pair_left_right_detections_using_3d_center.
  ///     Otherwise, uses @ref pair_left_right_tracks_using_bbox_iou.
  std::vector< std::vector< size_t > >
  pair_left_right_detections(
    const std::vector< kwiver::vital::detected_object_sptr >& left_detections,
    const std::vector< viame::Detections3DPositions >& left_3d_pos,
    const std::vector< kwiver::vital::detected_object_sptr >& right_detections );

  /// @brief Returns most likely detection class associated with input track
  /// Assumes the most likely detection class doesn't change in the lifetime of the input track
  static std::string most_likely_detection_class( const kwiver::vital::detected_object_sptr& detection );

};

} // viame

#endif // VIAME_OCV_PAIR_STEREO_DETECTIONS_H