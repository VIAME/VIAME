#ifndef VIAME_MEASUREMENT_OCV_PAIR_STEREO_DETECTIONS_H
#define VIAME_MEASUREMENT_OCV_PAIR_STEREO_DETECTIONS_H

#include <viame/core_types/bounding_box.h>
#include <viame/core_types/detected_object.h>
#include <viame/core_types/image.h>
#include <viame/core_types/matrix.h>
#include <viame/core_types/vector.h>
#include <viame/core_types/viame_core_types.h>

#include <viame/measurement/projection.h>

#include <viame/image_ops/windowed_utils.h>

#include "viame_measurement_export.h"

#include <memory>
#include <vector>

namespace viame {

/// @brief Structure containing extracted 3D information for each track in left image and mapping to right track
/// coordinates
struct VIAME_MEASUREMENT_EXPORT Detections3DPositions
{
  viame::vector_3d center3d{ 0.0, 0.0, 0.0 };
  viame::vector_2d center3d_proj_to_right_image{ 0.0, 0.0 };
  viame::bounding_box_d rectified_left_bbox{};
  viame::bounding_box_d left_bbox_proj_to_right_image{};
  float score{};

  /// @brief Validity status for the 3D track position
  /// Considered valid if at least one point used to process position center is valid
  bool is_valid() const { return score > 1e-6f; }
};


/// @brief Class responsible for the detection stereo pairing logic
/// Uses camera calibration information, left and right tracks and disparity map to find corresponding detection from
/// left to right.
class VIAME_MEASUREMENT_EXPORT pair_stereo_detections
{
public:
  pair_stereo_detections() = default;

  // Configuration settings
  std::string m_calibration_file;
  float m_iou_pair_threshold{ 0.1f };
  std::string m_pairing_method{ "PAIRING_3D" };
  bool m_verbose{}; // Set to true to activate debug print

  // Camera depth information. The rectification transforms are mutable because
  // single-file calibrations do not store them; they are derived on first use,
  // when an image size is finally available.
  viame::matrix_3x3d m_K1, m_K2, m_R;
  viame::measurement::distortion_t m_D1, m_D2;
  viame::vector_3d m_Rvec{ 0.0, 0.0, 0.0 };
  viame::vector_3d m_T{ 0.0, 0.0, 0.0 };

  mutable bool m_rectified{ false };
  mutable viame::matrix_4x4d m_Q;
  mutable viame::matrix_3x3d m_R1, m_R2;
  mutable viame::matrix_3x4d m_P1, m_P2;

  /// @brief Project depth map as 3 plane 3D image
  viame::image_of< float > reproject_3d_depth_map(
    const viame::image& disparity_left ) const;

  /// @brief Load matrix calibration from the configured calibration file
  void load_camera_calibration();

  /// @brief Derive the rectification transforms if the calibration lacked them
  /// No-op once they are populated. Called for the first disparity map, which
  /// is the earliest point an image size is known.
  void ensure_rectification( size_t width, size_t height ) const;

  /// @brief Compute median of input vector of values. Returns 0 if empty.
  static float compute_median( std::vector< float > values, bool is_sorted = false );

  /// @brief Convert input kwiver bounding box to an integer rectangle
  static image_rect bbox_to_mask_rect( viame::bounding_box_d const& bbox );
  static viame::bounding_box_d mask_rect_to_bbox( const image_rect& rect );

  /// @brief Get mask from input detection object
  ///     Copied from kwiver/arrows/ocv/refine_detections_util.cxx to include bbox used for the mask
  static viame::image_of< uint8_t > get_standard_mask(
    viame::detected_object_sptr const& det );

  /// @brief Estimate 3D position for input detection. If the detection contains a mask and that mask is contained
  /// within the left camera image, calculates position given mask bounds.
  /// Otherwise, returns detection for the input detection bounding box cropped to given bbox crop ratio.
  /// Set bbox_crop_ratio to 1.0 to use full bbox for 3D estimation.
  viame::Detections3DPositions
  estimate_3d_position_from_detection( const viame::detected_object_sptr& detection,
                                       const viame::image_of< float >& pos_3d_map,
                                       bool do_undistort_points,
                                       float bbox_crop_ratio ) const;

  /// @brief Rectify bounding box positions given loaded camera properties
  viame::bounding_box_d get_rectified_bbox( const viame::bounding_box_d& bbox,
                                                    bool is_left_image ) const;

  /// @brief Undistort input point coordinate
  viame::vector_2d undistort_point( const viame::vector_2d& point,
                                            bool is_left_image ) const;

  /// @brief Unistort input point coordinates
  std::vector< viame::vector_2d > undistort_point(
    const std::vector< viame::vector_2d >& point,
    bool is_left_image ) const;

  /// @return True if 3D point is valid (not infinite and Z positive)
  static bool point_is_valid( float x, float y, float z );

  /// @return True if 3D point is valid (not infinite and Z positive)
  static bool point_is_valid( const viame::vector_3d& pt );

  /// @brief Estimates 3D positions from input Bounding box and 3D map.
  /// @param bbox: Bounding box in left image on which to extract the 3D position
  /// @param pos_3d_map: 3D coordinate image with the same dimensions as the left image
  /// @param crop_ratio: [0 - 1] Defines the amount of the bounding box to use for the estimate.
  ///     Set to 1.f to use full bounding box.
  /// @param do_undistort_points: If true, will undistort bounding box corners by camera intrinsic. Otherwise, will use
  ///     provided bounding box coordinates directly.
  viame::Detections3DPositions
  estimate_3d_position_from_bbox( const viame::bounding_box_d& bbox,
                                  const viame::image_of< float >& pos_3d_map,
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
  estimate_3d_position_from_unrectified_mask( const viame::bounding_box_d& bbox,
                                              const viame::image_of< float >& pos_3d_map,
                                              const viame::image_of< uint8_t >& mask,
                                              bool do_undistort_points ) const;


  /// @brief Estimates 3D positions from input undistorted coordinates and 3D map.
  /// @param rectified_bbox: Bounding box in left image matching the mask coordinates
  /// @param undistorted_mask_coords: Undistorted mask coordinates
  /// @param pos_3d_map: 3D coordinate image with the same dimensions as the left image
  Detections3DPositions
  estimate_3d_position_from_point_coordinates( const viame::bounding_box_d& rectified_bbox,
                                               const std::vector< viame::vector_2d >& undistorted_mask_coords,
                                               const viame::image_of< float >& pos_3d_map ) const;

  /// @brief Creates 3D position from input X, Y, Z vectors.
  Detections3DPositions
  create_3d_position( const std::vector< float >& xs,
                      const std::vector< float >& ys,
                      const std::vector< float >& zs,
                      const viame::bounding_box_d& bbox,
                      const viame::image_of< float >& pos_3d_map,
                      float score ) const;

  /// @brief Projects the input left BBox to right image using 3D coordinates
  viame::bounding_box_d
  project_to_right_image( const viame::bounding_box_d& bbox,
                          const viame::image_of< float >& pos_3d_map ) const;

  /// @brief Projects the input left coordinates to right image using 3D coordinates
  viame::bounding_box_d project_to_right_image(
    const std::vector< viame::vector_3d >& points_3d ) const;

  /// @brief Projects the input left coordinates to right image using 3D coordinates
  viame::vector_2d project_to_right_image(
    const viame::vector_3d& points_3d ) const;

  /// @brief Update 3D tracks positions given a list of tracks and tracks disparity map
  std::vector< viame::Detections3DPositions >
  update_left_detections_3d_positions( const std::vector< viame::detected_object_sptr >& detections,
                                       const viame::image& disparity_map ) const;

  viame::Detections3DPositions
  update_left_detection_3d_position( const viame::detected_object_sptr& detection,
                                     const viame::image_of< float >& pos_3d_map ) const;


  /// @brief Calculates intersection over union for two bounding boxes
  static double iou_distance( const viame::bounding_box_d& bbox1,
                              const viame::bounding_box_d& bbox2 );


  /// @brief Update left and right tracks pairs using left 3D coordinates and right bounding boxes
  ///     - Project left 3D center coordinate to right image
  ///     - For each right bounding box containing projected left center, find closest
  static std::vector< std::vector< size_t > >
  pair_left_right_detections_using_3d_center(
    const std::vector< viame::detected_object_sptr >& left_detections,
    const std::vector< viame::Detections3DPositions >& left_3d_pos,
    const std::vector< viame::detected_object_sptr >& right_detections );

  /// @brief Update left and right tracks pairs using left and right bounding boxes
  ///     - For each left / right tracks, finds the pair having the highest IOU
  std::vector< std::vector< size_t > >
  pair_left_right_tracks_using_bbox_iou(
    const std::vector< viame::detected_object_sptr >& left_detections,
    const std::vector< viame::detected_object_sptr >& right_detections,
    bool do_rectify_bbox );

  /// @brief Update left and right tracks pairs using the currently set pairing method.
  ///     If pairing is set to 3D, uses @ref pair_left_right_detections_using_3d_center.
  ///     Otherwise, uses @ref pair_left_right_tracks_using_bbox_iou.
  std::vector< std::vector< size_t > >
  pair_left_right_detections(
    const std::vector< viame::detected_object_sptr >& left_detections,
    const std::vector< viame::Detections3DPositions >& left_3d_pos,
    const std::vector< viame::detected_object_sptr >& right_detections );

  /// @brief Returns most likely detection class associated with input track
  /// Assumes the most likely detection class doesn't change in the lifetime of the input track
  static std::string most_likely_detection_class( const viame::detected_object_sptr& detection );

};

} // viame

#endif // VIAME_MEASUREMENT_OCV_PAIR_STEREO_DETECTIONS_H