#include <gtest/gtest.h>
#include "filter_stereo_feature_tracks.h"

namespace kv = kwiver::vital;
using namespace viame;


std::vector< cv::Point3f > create_54_tracks_world_coords()
{
  float max_x{0.5}, max_y{0.25};
  size_t pattern_width{9}, pattern_height{6};
  float step_x{max_x / ( (float)pattern_width - 1.f )};
  float step_y{max_y / ( (float)pattern_height - 1.f )};

  std::vector< cv::Point3f > world_pts;
  for( size_t i_x = 0; i_x < pattern_width; i_x++ )
  {
    for( size_t i_y = 0; i_y < pattern_height; i_y++ )
    {
      auto x_world = (float)i_x * step_x;
      auto y_world = (float)i_y * step_y;

      world_pts.emplace_back( x_world, y_world, 0 );
    }
  }
  return world_pts;
}

/// @brief Generate 54 stereo point coordinates per given frame for input n frame number
StereoPointCoordinates generate_54_stereo_points_per_frame( size_t n_frames )
{
  StereoPointCoordinates coordinates;
  float img_width{1280}, img_height{720};

  auto track_world_coords = create_54_tracks_world_coords();
  for( size_t i_frame = 0; i_frame < n_frames; i_frame++ )
  {
    std::vector< cv::Point2f > img_pts_left, img_pts_right;
    std::vector< cv::Point3f > world_pts;

    for( const auto &world_coord : track_world_coords )
    {
      auto x_screen = world_coord.x * img_width;
      auto y_screen = world_coord.y * img_height;
      img_pts_left.emplace_back( cv::Point2f{ x_screen + (float)i_frame, y_screen } );
      img_pts_right.emplace_back( cv::Point2f{ x_screen + (float)( i_frame + n_frames ), y_screen } );
      world_pts.emplace_back( world_coord );
    }
    coordinates.image_pts[0].push_back( img_pts_left );
    coordinates.image_pts[1].push_back( img_pts_right );
    coordinates.world_pts.push_back( world_pts );
    coordinates.frame_ids.push_back( i_frame );
  }

  return coordinates;
}

void generated_landmarks_sanity_check( const FeatureTracks &feature_tracks, const Landmarks &landmarks, size_t n_cam,
                                       size_t n_frames, size_t n_tracks )
{
  // Sanity check on feature tracks
  ASSERT_EQ( feature_tracks.size(), n_cam );
  ASSERT_EQ( landmarks.size(), n_cam );
  for( const auto &feature_track : feature_tracks )
  {
    ASSERT_EQ( feature_track->all_frame_ids().size(), n_frames );
    ASSERT_EQ( feature_track->all_track_ids().size(), n_tracks );
  }
}

std::tuple< FeatureTracks, Landmarks >
convert_to_feature_tracks_and_landmarks( const StereoPointCoordinates &coordinates, size_t n_cam, size_t n_frames )
{
  // Initialize feature tracks with two cams by default
  FeatureTracks feature_tracks{ std::make_shared< kv::feature_track_set >(), std::make_shared< kv::feature_track_set >() };
  std::vector< kv::landmark_map::map_landmark_t > landmarks_map;
  landmarks_map.resize( 2 );

  // Initialize track world coordinates
  auto track_world_coords = create_54_tracks_world_coords();
  auto n_tracks = track_world_coords.size();

  // Helper function to create a feature track state given the camera number, frame index and track index
  auto get_feature = [&]( size_t i_cam, size_t i_frame, size_t i_track )
  {
    auto loc = coordinates.image_pts[i_cam][i_frame][i_track];
    auto feature{ std::make_shared< kv::feature_f >() };
    feature->loc().x() = loc.x;
    feature->loc().y() = loc.y;
    return std::make_shared< kv::feature_track_state >( i_frame, feature );
  };

  // For each track
  for( size_t i_track = 0; i_track < n_tracks; i_track++ )
  {
    // Create left and right tracks by default
    auto left_track{ kv::track::create() }, right_track{ kv::track::create() };
    left_track->set_id( (int)i_track );
    right_track->set_id( (int)i_track );

    // Populate landmark coordinates
    auto track_world_coord = track_world_coords[i_track];
    kv::vector_3d pt = kv::vector_3d( track_world_coord.x, track_world_coord.y, track_world_coord.z );
    landmarks_map[0][(int)i_track] = kv::landmark_sptr( new kv::landmark_d( pt ) );
    landmarks_map[1][(int)i_track] = kv::landmark_sptr( new kv::landmark_d( pt ) );

    // Populate track feature for each frame
    for( size_t i_frame = 0; i_frame < n_frames; i_frame++ )
    {
      left_track->insert( get_feature( 0, i_frame, i_track ) );
      right_track->insert( get_feature( 1, i_frame, i_track ) );
    }
    feature_tracks[0]->insert( left_track );
    feature_tracks[1]->insert( right_track );
  }

  Landmarks landmarks{ std::make_shared< kv::simple_landmark_map >( landmarks_map[0] ),
                       std::make_shared< kv::simple_landmark_map >( landmarks_map[1] ) };

  // Drop extra camera if needed
  if( n_cam == 1 )
  {
    feature_tracks.pop_back();
    landmarks.pop_back();
  }

  generated_landmarks_sanity_check( feature_tracks, landmarks, n_cam, n_frames, n_tracks );
  return { feature_tracks, landmarks };
}

TEST(filter_stereo_feature_tracksTest, with_empty_input_coordinates_does_nothing) {
  StereoPointCoordinates coordinates;
  auto out_coord = filter_stereo_feature_tracks::select_points_maximizing_variance(coordinates, 50);
  EXPECT_TRUE(out_coord.world_pts.empty());
}

TEST(filter_stereo_feature_tracksTest, returns_every_points_when_number_of_points_is_less_than_threshold) {
  size_t n_frames{45};
  auto coordinates = generate_54_stereo_points_per_frame(n_frames);

  size_t max_frame_count{50};
  auto out_coord = filter_stereo_feature_tracks::select_points_maximizing_variance(coordinates, max_frame_count);

  EXPECT_EQ(out_coord.world_pts.size(), n_frames);
  EXPECT_EQ(coordinates.image_pts[0].size(), n_frames);
  EXPECT_EQ(coordinates.image_pts[1].size(), n_frames);
}

TEST(filter_stereo_feature_tracksTest,
     returns_at_most_number_of_threshold_points_when_number_of_frames_greater_than_points) {
  size_t n_frames{100};
  auto coordinates = generate_54_stereo_points_per_frame(n_frames);

  size_t max_frame_count{50};
  auto out_coord = filter_stereo_feature_tracks::select_points_maximizing_variance(coordinates, max_frame_count);

  EXPECT_EQ(out_coord.world_pts.size(), max_frame_count);
  EXPECT_EQ(out_coord.image_pts[0].size(), max_frame_count);
  EXPECT_EQ(out_coord.image_pts[1].size(), max_frame_count);
}

TEST(filter_stereo_feature_tracksTest, filter_is_independant_from_mono_or_stereo_config) {
  size_t n_frames{100};
  auto coordinates = generate_54_stereo_points_per_frame(n_frames);

  for (size_t n_cam = 1; n_cam < 3; n_cam++) {
    auto tracks_and_landmarks = convert_to_feature_tracks_and_landmarks(coordinates, n_cam, n_frames);

    size_t max_frame_count{50};
    auto out_coord = filter_stereo_feature_tracks::select_frames(std::get<0>(tracks_and_landmarks),
                                                             std::get<1>(tracks_and_landmarks), max_frame_count);

    EXPECT_EQ(out_coord.world_pts.size(), max_frame_count);
    EXPECT_EQ(out_coord.image_pts[0].size(), max_frame_count);
    EXPECT_EQ(out_coord.image_pts[1].size(), max_frame_count);
  }
}

TEST(filter_stereo_feature_tracksTest, keeping_every_frame_returns_every_frame_point_coordinate) {
  size_t n_frames{100}, max_frame_count{100};
  size_t n_cam{2};
  auto coordinates = generate_54_stereo_points_per_frame(n_frames);
  auto tracks_and_landmarks = convert_to_feature_tracks_and_landmarks(coordinates, n_cam, n_frames);
  auto out_coord = filter_stereo_feature_tracks::select_frames(std::get<0>(tracks_and_landmarks),
                                                           std::get<1>(tracks_and_landmarks), max_frame_count);
  EXPECT_EQ(out_coord.frame_ids.size(), coordinates.frame_ids.size());
}

TEST(filter_stereo_feature_tracksTest, frames_with_incoherent_left_right_tracks_are_dropped) {
  size_t n_frames{100}, max_frame_count{100};
  size_t n_cam{2};
  auto coordinates = generate_54_stereo_points_per_frame(n_frames);
  auto tracks_and_landmarks = convert_to_feature_tracks_and_landmarks(coordinates, n_cam, n_frames);
  auto feature_tracks = std::get<0>(tracks_and_landmarks);
  auto landmarks = std::get<1>(tracks_and_landmarks);

  // Remove track from left cam
  size_t i_track_to_remove{5};
  feature_tracks[0]->remove(feature_tracks[0]->tracks()[i_track_to_remove]);
  ASSERT_NE(feature_tracks[0]->all_track_ids(), feature_tracks[1]->all_track_ids());

  // Expect output to be empty (53 tracks left vs 54 right)
  auto out_coord = filter_stereo_feature_tracks::select_frames(feature_tracks, landmarks, max_frame_count);
  EXPECT_TRUE(out_coord.frame_ids.empty());

  // Remove same track from right cam and expect output to have all frames
  feature_tracks[1]->remove(feature_tracks[1]->tracks()[i_track_to_remove]);
  out_coord = filter_stereo_feature_tracks::select_frames(feature_tracks, landmarks, max_frame_count);
  EXPECT_EQ(out_coord.frame_ids.size(), n_frames);
}

TEST(filter_stereo_feature_tracksTest, empty_frames_are_dropped) {
  size_t n_frames{100}, max_frame_count{100};
  size_t n_cam{2};
  auto coordinates = generate_54_stereo_points_per_frame(n_frames);
  auto tracks_and_landmarks = convert_to_feature_tracks_and_landmarks(coordinates, n_cam, n_frames);
  auto feature_tracks = std::get<0>(tracks_and_landmarks);
  auto landmarks = std::get<1>(tracks_and_landmarks);

  // Remove tracks for frame
  int i_frame_tracks_to_remove = 42;
  for (size_t i_cam = 0; i_cam < n_cam; i_cam++)
    for (const auto &track: feature_tracks[i_cam]->tracks())
      track->remove(i_frame_tracks_to_remove);

  auto frame_ids_left = feature_tracks[0]->all_frame_ids();
  ASSERT_TRUE(frame_ids_left.find(i_frame_tracks_to_remove) == std::end(frame_ids_left));

  // Expect output to contain n_frames - 1
  auto out_coord = filter_stereo_feature_tracks::select_frames(feature_tracks, landmarks, max_frame_count);
  EXPECT_EQ(out_coord.frame_ids.size(), (n_frames - 1));
}


TEST(filter_stereo_feature_tracksTest, using_max_frame_count_of_0_returns_all_frames) {
  size_t n_frames{100}, max_frame_count{0};
  size_t n_cam{2};
  auto coordinates = generate_54_stereo_points_per_frame(n_frames);
  auto tracks_and_landmarks = convert_to_feature_tracks_and_landmarks(coordinates, n_cam, n_frames);
  auto feature_tracks = std::get<0>(tracks_and_landmarks);
  auto landmarks = std::get<1>(tracks_and_landmarks);

  // Expect output to contain n_frames - 1
  auto out_coord = filter_stereo_feature_tracks::select_frames(feature_tracks, landmarks, max_frame_count);
  EXPECT_EQ(out_coord.frame_ids.size(), n_frames);
}
