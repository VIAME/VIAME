// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <test_scene.h>

#include <arrows/mvg/algo/integrate_depth_maps.h>
#include <arrows/core/render_mesh_depth_map.h>
#include <arrows/core/mesh_operations.h>

#include <vital/plugin_loader/plugin_manager.h>
#include <vital/types/camera_perspective_map.h>
#include <vital/util/transform_image.h>

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
TEST(integrate_depth_maps, create)
{
  using namespace kwiver::vital;

  plugin_manager::instance().load_all_plugins();

  EXPECT_NE(nullptr, algo::integrate_depth_maps::create("mvg"));
}

// ----------------------------------------------------------------------------
// Test depth map integration
TEST(integrate_depth_maps, integrate)
{
  namespace mvg = kwiver::arrows::mvg;
  namespace core = kwiver::arrows::core;

  // create two stacked boxes on a ground plane
  auto cube = kwiver::testing::cube_mesh( 1.0 );
  cube->merge(*kwiver::testing::cube_mesh(0.5, {0.0, 0.0, 0.75}));
  cube->merge(*kwiver::testing::grid_mesh(100, 20, 1.0, {-10.0, -10.0, -0.5}));

  // convert to triangles for rendering
  core::mesh_triangulate(*cube);

  // create a camera sequence (elliptical path)
  auto K = simple_camera_intrinsics(
    200, { 80, 60 }, 1.0, 0.0, {}, 160, 120);
  auto cameras = kwiver::testing::camera_seq(20, K, 1.0, 360.0);
  camera_perspective_map pcameras;
  pcameras.set_from_base_cams(cameras);

  std::vector< image_container_sptr > depth_maps;
  std::vector< camera_perspective_sptr > cams;
  for (auto const& camera : pcameras.T_cameras())
  {
    depth_maps.push_back(core::render_mesh_depth_map(cube, camera.second));
    cams.push_back(camera.second);
  }

  mvg::integrate_depth_maps algorithm;
  config_block_sptr config = algorithm.get_configuration();
  config->set_value("voxel_spacing_factor", 1.0);
  algorithm.set_configuration(config);
  image_container_sptr volume = nullptr;
  vector_3d spacing{ 1.0, 1.0, 1.0 };
  vector_3d min_pt{ -1.0, -1.0, -0.7 };
  vector_3d max_pt{ 1.0, 1.0, 1.2 };
  algorithm.integrate(min_pt, max_pt,
                      depth_maps, {}, cams, volume, spacing);
  EXPECT_EQ(volume->width(), 89);
  EXPECT_EQ(volume->height(), 89);
  EXPECT_EQ(volume->depth(), 84);
  vector_3d sizes = max_pt - min_pt;
  EXPECT_NEAR(spacing[0] * volume->width(), sizes[0], spacing[0]);
  EXPECT_NEAR(spacing[1] * volume->height(), sizes[1], spacing[1]);
  EXPECT_NEAR(spacing[2] * volume->depth(), sizes[2], spacing[2]);

  // helper function to look up volume values in global coordinates
  image_of<double> vol_data(volume->get_image());
  auto world_value = [&vol_data, spacing, min_pt](vector_3d const& v)
  {
    Eigen::Vector3i index = ((v - min_pt).array() / spacing.array()).cast<int>();
    if (index[0] < 0 || index[0] >= vol_data.width() ||
      index[1] < 0 || index[1] >= vol_data.height() ||
      index[2] < 0 || index[2] >= vol_data.depth())
    {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return vol_data(index[0], index[1], index[2]);
  };

  // values inside the structure should have positive values
  EXPECT_GT(world_value({ 0.0, 0.0, 0.0 }), 0.0);
  EXPECT_GT(world_value({ 0.0, 0.0, -0.6 }), 0.0);
  EXPECT_GT(world_value({ 0.0, 0.0, 1.0 }), 0.0);
  EXPECT_GT(world_value({ -0.75, -0.75, -0.6 }), 0.0);

  // values near the boundary should have small values
  EXPECT_NEAR(world_value({ 0.5, 0.0, 0.0 }), 0.0, 1.0);
  EXPECT_NEAR(world_value({ 0.0, 0.5, 0.0 }), 0.0, 1.0);
  EXPECT_NEAR(world_value({ 0.5, 0.5, 0.0 }), 0.0, 1.0);
  EXPECT_NEAR(world_value({ 0.0, 0.0, 1.0 }), 0.0, 1.0);

  // values inside the structure should have positive values
  EXPECT_LT(world_value({ 0.0, 0.0, 1.1 }), 0.0);
  EXPECT_LT(world_value({ 0.5, 0.5, 0.6 }), 0.0);
  EXPECT_LT(world_value({ -0.75, -0.75, -0.4 }), 0.0);
}
