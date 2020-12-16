// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <test_scene.h>

#include <arrows/mvg/projected_track_set.h>
#include <arrows/mvg/epipolar_geometry.h>

#include <gtest/gtest.h>

#include <algorithm>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
// Apply transform to elements in a vector; return vector of transformed
// elements
template <typename T, typename Func>
static
auto transform( std::vector<T> const& in, Func xf )
  -> std::vector< decltype( xf( std::declval<T>() ) ) >
{
  std::vector< decltype( xf( std::declval<T>() ) ) > result;
  result.reserve( in.size() );

  std::transform(
    in.begin(), in.end(), std::back_inserter( result ), xf );

  return result;
}

// ----------------------------------------------------------------------------
// Print epipolar distance of pairs of points given a fundamental matrix
void print_epipolar_distances(
  kwiver::vital::matrix_3x3d const& F,
  std::vector<kwiver::vital::vector_2d> const& right_pts,
  std::vector<kwiver::vital::vector_2d> const& left_pts)
{
  using namespace kwiver::arrows;

  matrix_3x3d Ft = F.transpose();
  for ( unsigned i = 0; i < right_pts.size(); ++i )
  {
    auto const& pr = right_pts[i];
    auto const& pl = left_pts[i];

    vector_3d vr( pr.x(), pr.y(), 1.0 );
    vector_3d vl( pl.x(), pl.y(), 1.0 );
    vector_3d lr = F * vr;
    vector_3d ll = Ft * vl;

    double sr = 1.0 / sqrt( lr.x() * lr.x() + lr.y() * lr.y() );
    double sl = 1.0 / sqrt( ll.x() * ll.x() + ll.y() * ll.y() );

    // Sum of point to epipolar line distance in both images
    double d = vr.dot( ll );

    std::cout << " dist right = " << d * sr
              << "  dist left = " << d * sl
              << std::endl;
  }
}

// ----------------------------------------------------------------------------
// Test essential matrix estimation with ideal points
TEST(epipolar_geometry, ideal_points)
{
  using namespace kwiver::arrows::mvg;

  // create landmarks at the random locations
  auto landmarks = kwiver::testing::init_landmarks( 100 );
  landmarks = kwiver::testing::noisy_landmarks( landmarks, 1.0 );

  // create a camera sequence (elliptical path)
  auto const& cameras = kwiver::testing::camera_seq();

  // create tracks from the projections
  auto const& tracks = projected_tracks( landmarks, cameras );

  const frame_id_t frame1 = 0;
  const frame_id_t frame2 = 10;

  camera_map::map_camera_t cams = cameras->cameras();
  auto cam1 = std::dynamic_pointer_cast<camera_perspective>(cams[frame1]);
  auto cam2 = std::dynamic_pointer_cast<camera_perspective>(cams[frame2]);
  camera_intrinsics_sptr cal1 = cam1->intrinsics();
  camera_intrinsics_sptr cal2 = cam2->intrinsics();

  // compute the true essential matrix from the cameras
  auto const& em = essential_matrix_from_cameras( *cam1, *cam2 );
  auto const& fm = fundamental_matrix_from_cameras( *cam1, *cam2 );

  // Extract corresponding image points
  std::vector<vector_2d> pts1, pts2;
  for ( auto const& track : tracks->tracks() )
  {
    auto const fts1 =
      std::dynamic_pointer_cast<feature_track_state>( *track->find( frame1 ) );
    auto const fts2 =
      std::dynamic_pointer_cast<feature_track_state>( *track->find( frame2 ) );
    pts1.push_back( fts1->feature->loc() );
    pts2.push_back( fts2->feature->loc() );
  }

  using std::placeholders::_1;
  auto const unmap = &camera_intrinsics::unmap;
  auto const& norm_pts1 = transform( pts1, std::bind( unmap, cal1, _1 ) );
  auto const& norm_pts2 = transform( pts2, std::bind( unmap, cal2, _1 ) );

  // Print the epipolar distances using this fundamental matrix
  print_epipolar_distances( fm->matrix(), pts1, pts2 );

  // Compute the inliers with a small scale
  auto const& inliers = mark_fm_inliers( *fm, pts1, pts2, 1e-8 );
  EXPECT_EQ( pts1.size(), inliers.size() );

  // Compute the inliers with a small scale
  auto const& norm_inliers =
    mark_fm_inliers( fundamental_matrix_d( em->matrix() ),
                     norm_pts1, norm_pts2, 1e-8 );
  EXPECT_EQ( norm_pts1.size(), norm_inliers.size() );
}
