#ifndef VIAME_OPENCV_FILTER_STEREO_FEATURE_TRACKS_H
#define VIAME_OPENCV_FILTER_STEREO_FEATURE_TRACKS_H

#include <vital/types/feature_track_set.h>
#include <vital/types/landmark_map.h>
#include <opencv2/core/eigen.hpp>
#include "viame_opencv_export.h"

namespace viame {

using Landmarks = std::vector< kwiver::vital::landmark_map_sptr >;
using FeatureTracks = std::vector< kwiver::vital::feature_track_set_sptr >;

// Feature track states grouped by camera_index > track_id;
using CameraFeatureTrackStates = std::vector< std::vector< kwiver::vital::feature_track_state_sptr > >;

// Feature track states grouped by frame > camera_index > track_id;
using FrameFeatureTrackStates = std::vector< CameraFeatureTrackStates >;

/// @brief Structure containing only the points for single camera calibration and stereo camera calibration
struct VIAME_OPENCV_EXPORT StereoPointCoordinates
{
  /// @brief Default Ctor which always initializes images_pts for stereo cameras (ie. image_pts of size 2)
  StereoPointCoordinates();

  std::vector< std::vector< std::vector< cv::Point2f > > > image_pts; // Frame points grouped by cam > frame
  std::vector< std::vector< cv::Point3f > > world_pts;  // World points grouped by frame
  std::vector< size_t > frame_ids; // Frame ids

  static StereoPointCoordinates from_features( const FrameFeatureTrackStates& features,
                                               const Landmarks& landmarks );
};

/// @brief Class responsible for selecting the best tracks in given feature track vector for the calibration
class VIAME_OPENCV_EXPORT filter_stereo_feature_tracks
{
public:
  static StereoPointCoordinates
  select_points_maximizing_variance( const StereoPointCoordinates& coordinates,
                                     size_t frame_count_threshold );

  /// @brief Select at most N tracks from input frames with maximum variation sampling method.
  /// Allows to limit the number of frames used during calibration processing while keeping the overall calibration
  /// quality.
  static StereoPointCoordinates
  select_frames( const FeatureTracks& features,
                 const Landmarks& landmarks,
                 size_t frame_count_threshold );

private:
  /// @brief Remove empty frames from the input feature tracks vector
  static FrameFeatureTrackStates remove_empty_frames( FrameFeatureTrackStates frame_feature_tracks );

  /// @brief Group input feature tracks vector by frame ids > Cameras > Tracks
  /// Output vector only contains full frames
  static FrameFeatureTrackStates group_by_frame_id( const FeatureTracks& feature_tracks );

  /// @brief Remove frames where the number of tracks between left and right cameras is not identical to maximum number
  /// of tracks detected (function of calibration pattern)
  /// For single camera case, this method call does nothing
  static FrameFeatureTrackStates
  remove_frames_without_corresponding_left_right_match( FrameFeatureTrackStates features );

  /// @brief Returns the maximum id in the input feature track vector given a max function
  static int64_t
  max_feature_id( const FeatureTracks& feature_tracks,
                  const std::function< int64_t( const kwiver::vital::feature_track_set_sptr& ) >& max_f );

  /// @brief Returns the maximum track id in the input feature track vector
  static int64_t max_feature_tracks_track_id( const FeatureTracks& feature_tracks );

  /// @brief Returns the maximum frame id in the input feature track vector
  static int64_t max_feature_tracks_frame_id( const FeatureTracks& feature_tracks );

  static cv::Mat
  create_frames_extents_matrix( const viame::StereoPointCoordinates& coordinates, int n_frames );

  static std::array< cv::Point3f, 4 >
  get_world_point_corner_values( const std::vector< std::vector< cv::Point3f > >& world_pts );

  static std::array< int, 2 >
  get_destination_extent( const cv::Point3f& world_pt,
                          const std::array< cv::Point3f, 4 >& world_points_corner_values );

  static std::set< size_t >
  select_frames_in_cluster( const cv::Mat& frames_matrix,
                            const cv::Mat& cluster_labels,
                            const cv::Mat& cluster_centers );

  static StereoPointCoordinates
  filter_points_in_kept_frames( const StereoPointCoordinates& coordinates,
                                const std::set< size_t >& selected_frame_idx );
};

} // viame

#endif // VIAME_OPENCV_FILTER_STEREO_FEATURE_TRACKS_H
