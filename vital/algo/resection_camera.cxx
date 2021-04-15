// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief resection_camera instantiation

#include <vital/algo/algorithm.txx>
#include <vital/algo/resection_camera.h>

#include <vital/range/iota.h>

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF( kwiver::vital::algo::resection_camera );
/// \endcond

namespace kwiver {

namespace vital {

namespace algo {

// ----------------------------------------------------------------------------
resection_camera
::resection_camera()
{
  attach_logger( "algo.resection_camera" );
}

// ----------------------------------------------------------------------------
camera_perspective_sptr
resection_camera
::resection( frame_id_t frame_id,
             landmark_map_sptr landmarks,
             feature_track_set_sptr tracks,
             unsigned width, unsigned height,
             std::unordered_set< landmark_id_t >* inliers ) const
{
  // Generate calibration guess from image dimensions.
  auto const principal_point = vector_2d{ width * 0.5, height * 0.5 };
  auto cal = std::make_shared< simple_camera_intrinsics >(
    ( width + height ) * 0.5,
    principal_point, 1.0, 0.0,
    Eigen::VectorXd(), width, height );

  // Resection using guessed calibration.
  return resection( frame_id, landmarks, tracks, cal, inliers );
}

// ----------------------------------------------------------------------------
camera_perspective_sptr
resection_camera
::resection( frame_id_t frame_id,
             landmark_map_sptr landmarks,
             feature_track_set_sptr tracks,
             kwiver::vital::camera_intrinsics_sptr cal,
             std::unordered_set< landmark_id_t >* inliers ) const
{
  auto world_points = std::vector< vector_3d >{};
  auto camera_points = std::vector< vector_2d >{};
  auto inliers_mask = std::vector< bool >{};
  auto ids_used = std::vector< landmark_id_t >{};

  auto const inliers_pointer = ( inliers ? &inliers_mask : nullptr );

  auto const& real_landmarks = landmarks->landmarks();
  for( auto const& fts : tracks->frame_feature_track_states( frame_id ) )
  {
    auto lmi = real_landmarks.find( fts->track()->id() );
    if( lmi != real_landmarks.end() )
    {
      world_points.emplace_back( lmi->second->loc() );
      camera_points.emplace_back( fts->feature->loc() );
      if( inliers )
      {
        ids_used.push_back( lmi->first );
      }
    }
  }

  // Resection camera using point correspondences and initial calibration guess
  auto result = resection( camera_points, world_points, cal, inliers_pointer );
  if( inliers )
  {
    for( auto const i : range::iota( ids_used.size() ) )
    {
      if( inliers_mask[ i ] )
      {
        inliers->insert( ids_used[ i ] );
      }
    }
  }
  return result;
}

} // namespace algo

} // namespace vital

} // namespace kwiver
