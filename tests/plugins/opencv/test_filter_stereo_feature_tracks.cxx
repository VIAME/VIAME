#include <gtest/gtest.h>

#include "filter_stereo_feature_tracks.h"

namespace viame {
namespace testing {

TEST( filter_stereo_feature_tracks, stereo_point_coordinates_default_ctor )
{
  StereoPointCoordinates coords;

  // Default constructor should initialize image_pts for stereo (2 cameras)
  EXPECT_EQ( coords.image_pts.size(), 2 );
  EXPECT_TRUE( coords.world_pts.empty() );
  EXPECT_TRUE( coords.frame_ids.empty() );
}

TEST( filter_stereo_feature_tracks, select_points_maximizing_variance_empty_input )
{
  StereoPointCoordinates coords;
  size_t frame_count_threshold = 10;

  auto result = filter_stereo_feature_tracks::select_points_maximizing_variance(
    coords, frame_count_threshold );

  // With empty input, result should also be empty
  EXPECT_TRUE( result.world_pts.empty() );
  EXPECT_TRUE( result.frame_ids.empty() );
}

TEST( filter_stereo_feature_tracks, select_frames_empty_input )
{
  FeatureTracks features;
  Landmarks landmarks;
  size_t frame_count_threshold = 10;

  auto result = filter_stereo_feature_tracks::select_frames(
    features, landmarks, frame_count_threshold );

  // With empty input, result should also be empty
  EXPECT_TRUE( result.world_pts.empty() );
  EXPECT_TRUE( result.frame_ids.empty() );
}

} // namespace testing
} // namespace viame
