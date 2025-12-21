#include "stereo_feature_track_filter.h"
#include "kmedians.h"
#include <cmath>

viame::StereoPointCoordinates
viame::StereoPointCoordinates
::from_features( const viame::FrameFeatureTrackStates& features,
                 const viame::Landmarks& landmarks )
{
  // Initialize point coordinates
  StereoPointCoordinates coordinates;

  if( features.empty() || landmarks.empty() )
    return coordinates;

  auto lms1 = landmarks[0]->landmarks();
  auto lms2 = landmarks.size() > 1 ? landmarks[1]->landmarks() : lms1;

  for( const auto& frame : features )
  {
    auto cam1 = frame[0];
    auto cam2 = frame.size() > 1 ? frame[1] : cam1;

    // Sanity skip of frames for which the number of tracks is different
    // This shouldn't be the case if previous filtering is correctly applied
    if( cam1.size() != cam2.size() )
      continue;

    // Initialize points vector for frame
    std::vector< cv::Point2f > pts2D1, pts2D2;
    std::vector< cv::Point3f > pts3D;
    std::set< size_t > frame_ids;
    for( const auto& state1 : cam1 )
    {
      if( !state1 )
        continue;

      auto fts1 = state1->track();
      for( const auto& state2 : cam2 )
      {
        if( !state2 )
          continue;

        auto fts2 = state2->track();
        if( fts1->id() != fts2->id() )
          continue;

        bool world1_found = lms1.count( fts1->id() );
        bool world2_found = lms2.count( fts2->id() );

        cv::Point3d w1, w2;
        if( world1_found )
        {
          w1 = cv::Point3d( lms1[fts1->id()]->loc()[0], lms1[fts1->id()]->loc()[1], lms1[fts1->id()]->loc()[2] );
        }

        if( world2_found )
        {
          w2 = cv::Point3d( lms2[fts2->id()]->loc()[0], lms2[fts2->id()]->loc()[1], lms2[fts2->id()]->loc()[2] );
        }

        // Check if world points are valid
        bool world_found = false;
        if( world1_found && world2_found )
        {
          world_found = w1 == w2;
        }
        else if( world1_found || world2_found )
        {
          world_found = true;
        }

        if( !world_found ) continue;


        auto center11 = state1->feature->loc();
        auto center21 = state2->feature->loc();
        pts2D1.push_back( cv::Point2d( center11[0], center11[1] ) );
        pts2D2.push_back( cv::Point2d( center21[0], center21[1] ) );
        pts3D.push_back( world1_found ? w1 : w2 );
        frame_ids.insert( state1->frame() );
      }
    }

    // Sanity check on frame ids content
    assert( frame_ids.size() == 1 && "Track frame ids is incoherent for given frame." );
    coordinates.frame_ids.push_back( *frame_ids.begin() );

    coordinates.image_pts[0].push_back( pts2D1 );
    coordinates.image_pts[1].push_back( pts2D2 );
    coordinates.world_pts.push_back( pts3D );
  }

  return coordinates;
}

viame::StereoPointCoordinates::StereoPointCoordinates()
{
  image_pts.resize( 2 );
}

viame::FrameFeatureTrackStates
viame::StereoFeatureTrackFilter
::remove_empty_frames( viame::FrameFeatureTrackStates frame_feature_tracks )
{
  frame_feature_tracks.erase(
    std::remove_if( std::begin( frame_feature_tracks ), std::end( frame_feature_tracks ),
                    []( const CameraFeatureTrackStates& frame ) { return frame.empty(); } ),
    std::end( frame_feature_tracks ) );
  return frame_feature_tracks;
}

viame::FrameFeatureTrackStates
viame::StereoFeatureTrackFilter
::group_by_frame_id( const viame::FeatureTracks& feature_tracks )
{
  if( feature_tracks.empty() )
    return {};

  auto max_frame_id = max_feature_tracks_frame_id( feature_tracks );
  auto max_track_id = max_feature_tracks_track_id( feature_tracks );
  if( max_frame_id < 0 || max_track_id < 0 )
    return {};

  FrameFeatureTrackStates frame_feature_tracks;
  frame_feature_tracks.resize( max_frame_id + 1 );

  size_t i_feature{ 0 };
  for( const auto& features : feature_tracks )
  {
    for( const auto& track : features->tracks() )
    {
      for( const auto& state : *track | kwiver::vital::as_feature_track )
      {
        if( frame_feature_tracks[state->frame()].empty() )
          frame_feature_tracks[state->frame()].resize( feature_tracks.size() );

        if( frame_feature_tracks[state->frame()][i_feature].empty() )
          frame_feature_tracks[state->frame()][i_feature].resize( max_track_id + 1 );

        frame_feature_tracks[state->frame()][i_feature][state->track()->id()] = state;
      }
    }
    i_feature++;
  }

  return remove_empty_frames( frame_feature_tracks );
}

viame::FrameFeatureTrackStates
viame::StereoFeatureTrackFilter
::remove_frames_without_corresponding_left_right_match( viame::FrameFeatureTrackStates features )
{
  // For single camera case, no need to check for correspondence
  bool is_single_camera = true;
  for( const auto& frame : features )
    is_single_camera &= frame.size() == 1;

  if( is_single_camera )
    return features;

  // Otherwise, check both camera tracks are present and each contain the maximum number of tracks detected
  auto doesnt_have_corresponding_tracks = []( const CameraFeatureTrackStates& frame )
  {
    if( frame.size() == 1 )
      return false;

    auto get_track_ids = [&]( size_t i_cam )
    {
      std::set< size_t > track_ids;
      for( const auto& track : frame[i_cam] )
        if( track )
          track_ids.insert( track->track()->id() );
      return track_ids;
    };

    return !( get_track_ids( 0 ) == get_track_ids( 1 ) );
  };

  features.erase( std::remove_if( std::begin( features ), std::end( features ), doesnt_have_corresponding_tracks ),
                  std::end( features ) );
  return features;
}

viame::StereoPointCoordinates
viame::StereoFeatureTrackFilter
::select_points_maximizing_variance( const StereoPointCoordinates& coordinates,
                                     size_t frame_count_threshold )
{
  // Count number of frames in input point coordinates
  auto n_frames = coordinates.frame_ids.size();

  // If frame count is less than frame count threshold, return all frames or max is deactivated (ie = 0)
  if( ( frame_count_threshold == 0 ) || ( n_frames <= frame_count_threshold ) )
    return coordinates;

  // Create matrix with N frame rows and 16 columns (8 coordinates per points)
  auto frames_matrix = create_frames_extents_matrix( coordinates, n_frames );

  // Group frames using a KMedians clustering method with frame count threshold
  int max_iter_count{ 50 }, attempts{ 5 };
  double eps{ 1.0 };
  cv::Mat labels, centers;
  kmedians( frames_matrix, (int) frame_count_threshold, labels,
            cv::TermCriteria( cv::TermCriteria::EPS + cv::TermCriteria::COUNT, max_iter_count, eps ), attempts,
            cv::KMEANS_PP_CENTERS, centers );

  // Keep closest frame to center for each cluster
  auto kept_frame_idx = select_frames_in_cluster( frames_matrix, labels, centers );

  // Filter points in kept frames
  return filter_points_in_kept_frames( coordinates, kept_frame_idx );
}

viame::StereoPointCoordinates
viame::StereoFeatureTrackFilter
::select_frames( const viame::FeatureTracks& features,
                 const viame::Landmarks& landmarks,
                 size_t frame_count_threshold )
{
  // Group tracks by frame
  auto frame_features = group_by_frame_id( features );

  // Remove frames without matches in both left / right cameras
  frame_features = remove_frames_without_corresponding_left_right_match( frame_features );

  // Convert and return frame features to 2D & 3D point coordinates
  auto stereo_points = StereoPointCoordinates::from_features( frame_features, landmarks );

  // Limit the number of stereo points to the ones maximizing the variance
  return select_points_maximizing_variance( stereo_points, frame_count_threshold );
}

int64_t
viame::StereoFeatureTrackFilter
::max_feature_id( const viame::FeatureTracks& feature_tracks,
                  const std::function< int64_t( const kwiver::vital::feature_track_set_sptr& ) >& max_f )
{
  int64_t max_id{ -1 };
  for( const auto& feature : feature_tracks )
  {
    max_id = std::max( max_id, max_f( feature ) );
  }

  return max_id;
}

int64_t
viame::StereoFeatureTrackFilter
::max_feature_tracks_track_id( const viame::FeatureTracks& feature_tracks )
{
  auto max_track_id_f = []( const kwiver::vital::feature_track_set_sptr& f ) { return *f->all_track_ids().rbegin(); };
  return max_feature_id( feature_tracks, max_track_id_f );
}

int64_t
viame::StereoFeatureTrackFilter
::max_feature_tracks_frame_id( const viame::FeatureTracks& feature_tracks )
{
  auto max_frame_id_f = []( const kwiver::vital::feature_track_set_sptr& f ) { return *f->all_frame_ids().rbegin(); };
  return max_feature_id( feature_tracks, max_frame_id_f );
}

cv::Mat
viame::StereoFeatureTrackFilter
::create_frames_extents_matrix( const viame::StereoPointCoordinates& coordinates, int n_frames )
{
  // Initialize extent matrix to 0
  constexpr size_t extent_size{ 16 };
  constexpr size_t half_extent_size{ extent_size / 2 };
  cv::Mat frames_matrix( (int) n_frames, extent_size, CV_32FC1 );
  frames_matrix = 0;

  // Find world corner coordinates from list of world points
  auto world_points_corner_values = get_world_point_corner_values( coordinates.world_pts );

  // For each frame, fill the extent matrix with 2D corner position
  for( size_t i_frame = 0; i_frame < coordinates.world_pts.size(); i_frame++ )
  {
    for( size_t i_point = 0; i_point < coordinates.world_pts[i_frame].size(); i_point++ )
    {
      // Find if the current point matches a world point corner. If it's the case, save the point image coordinates
      // to the frame extents matrix.
      auto world_pts = coordinates.world_pts[i_frame][i_point];
      auto dest_extent = get_destination_extent( world_pts, world_points_corner_values );
      if( dest_extent[0] == -1 )
        continue;

      frames_matrix.at< float >( i_frame, dest_extent[0] ) = coordinates.image_pts[0][i_frame][i_point].x;
      frames_matrix.at< float >( i_frame, dest_extent[1] ) = coordinates.image_pts[0][i_frame][i_point].y;
      frames_matrix.at< float >( i_frame,
                                 dest_extent[0] + (int) half_extent_size ) = coordinates.image_pts[1][i_frame][i_point].x;
      frames_matrix.at< float >( i_frame,
                                 dest_extent[1] + (int) half_extent_size ) = coordinates.image_pts[1][i_frame][i_point].y;
    }
  }
  return frames_matrix;
}

std::array< cv::Point3f, 4 >
viame::StereoFeatureTrackFilter
::get_world_point_corner_values( const std::vector< std::vector< cv::Point3f > >& world_pts )
{
  // Find world X and Y bounds while ignoring Z (using flat calibration pattern)
  auto min_float{ std::numeric_limits< float >::lowest() };
  auto max_float{ std::numeric_limits< float >::max() };

  float min_x{ max_float }, min_y{ max_float };
  float max_x{ min_float }, max_y{ min_float };
  float z{};

  for( const auto& pt_vect : world_pts )
  {
    auto pt = pt_vect[0];
    min_x = std::min( min_x, pt.x );
    max_x = std::max( max_x, pt.x );

    min_y = std::min( min_y, pt.y );
    max_y = std::max( max_y, pt.y );
  }

  return std::array< cv::Point3f, 4 >{ cv::Point3f{ min_x, min_y, z }, cv::Point3f{ min_x, max_y, z },
                                       cv::Point3f{ max_x, min_y, z }, cv::Point3f{ max_x, max_y, z } };
}

std::array< int, 2 >
viame::StereoFeatureTrackFilter
::get_destination_extent( const cv::Point3f& world_pt,
                          const std::array< cv::Point3f, 4 >& world_points_corner_values )
{
  auto is_close = []( float a, float b ) { return std::abs( a - b ) < 1e-6; };

  for( size_t i_corner = 0; i_corner < world_points_corner_values.size(); i_corner++ )
  {
    auto world_corner = world_points_corner_values[i_corner];
    if( is_close( world_corner.x, world_pt.x ) && is_close( world_corner.y, world_pt.y ) )
      return { (int) i_corner * 2, (int) i_corner * 2 + 1 };
  }

  return { -1, -1 };
}

std::set< size_t >
viame::StereoFeatureTrackFilter
::select_frames_in_cluster( const cv::Mat& frames_matrix,
                            const cv::Mat& cluster_labels,
                            const cv::Mat& cluster_centers )
{
  std::set< size_t > selected_frames_idx{};

  for( int i_cluster = 0; i_cluster < cluster_centers.rows; i_cluster++ )
  {
    auto i_frame = find_closest_frame_id_to_center( frames_matrix, cluster_labels, cluster_centers, i_cluster );
    selected_frames_idx.insert( i_frame );
  }

  return selected_frames_idx;
}

viame::StereoPointCoordinates
viame::StereoFeatureTrackFilter
::filter_points_in_kept_frames( const viame::StereoPointCoordinates& coordinates,
                                const std::set< size_t >& selected_frame_idx )
{
  StereoPointCoordinates output;
  for( const auto& i_frame : selected_frame_idx )
  {
    output.image_pts[0].push_back( coordinates.image_pts[0][i_frame] );
    output.image_pts[1].push_back( coordinates.image_pts[1][i_frame] );
    output.world_pts.push_back( coordinates.world_pts[i_frame] );
    output.frame_ids.push_back( coordinates.frame_ids[i_frame] );
  }

  return output;
}
