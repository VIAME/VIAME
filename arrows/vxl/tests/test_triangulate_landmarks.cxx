// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <test_scene.h>

#include <arrows/tests/test_triangulate_landmarks.h>
#include <arrows/vxl/triangulate_landmarks.h>

#include <vital/plugin_loader/plugin_manager.h>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(triangulate_landmarks, create)
{
  plugin_manager::instance().load_all_plugins();

  EXPECT_NE(nullptr, algo::triangulate_landmarks::create("vxl"));
}

// ----------------------------------------------------------------------------
// Input to triangulation is the ideal solution, make sure it doesn't diverge
TEST(triangulate_landmarks, from_solution)
{
  kwiver::arrows::vxl::triangulate_landmarks tri_lm;
  kwiver::testing::test_from_solution(tri_lm);
}

// ----------------------------------------------------------------------------
// Add noise to landmarks before input to triangulation
TEST(triangulate_landmarks, noisy_landmarks)
{
  kwiver::arrows::vxl::triangulate_landmarks tri_lm;
  kwiver::testing::test_noisy_landmarks(tri_lm);
}

// ----------------------------------------------------------------------------
// Initialize all landmarks to the origin as input to triangulation
TEST(triangulate_landmarks, zero_landmarks)
{
  kwiver::arrows::vxl::triangulate_landmarks tri_lm;
  kwiver::testing::test_zero_landmarks(tri_lm);
}

// ----------------------------------------------------------------------------
// Select a subset of cameras to triangulation from
TEST(triangulate_landmarks, subset_cameras)
{
  kwiver::arrows::vxl::triangulate_landmarks tri_lm;
  kwiver::testing::test_subset_cameras(tri_lm);
}

// ----------------------------------------------------------------------------
// Select a subset of landmarks to triangulate
TEST(triangulate_landmarks, subset_landmarks)
{
  kwiver::arrows::vxl::triangulate_landmarks tri_lm;
  kwiver::testing::test_subset_landmarks(tri_lm);
}

// ----------------------------------------------------------------------------
// Select a subset of tracks/track_states to constrain the problem
TEST(triangulate_landmarks, subset_tracks)
{
  kwiver::arrows::vxl::triangulate_landmarks tri_lm;
  kwiver::testing::test_subset_tracks(tri_lm);
}

// ----------------------------------------------------------------------------
// Select a subset of tracks/track_states and add noise
TEST(triangulate_landmarks, noisy_tracks)
{
  kwiver::arrows::vxl::triangulate_landmarks tri_lm;
  kwiver::testing::test_noisy_tracks(tri_lm);
}
