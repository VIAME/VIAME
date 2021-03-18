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
#include <vital/util/cpu_timer.h>

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

void make_test_data(std::vector< image_container_sptr >& depth_maps,
                    std::vector< camera_perspective_sptr >& cameras,
                    vector_3d& min_pt, vector_3d& max_pt,
                    simple_camera_intrinsics const& K)
{
  namespace core = kwiver::arrows::core;

  // create two stacked boxes on a ground plane
  auto cube = kwiver::testing::cube_mesh(1.0);
  cube->merge(*kwiver::testing::cube_mesh(0.5, { 0.0, 0.0, 0.75 }));
  cube->merge(*kwiver::testing::grid_mesh(20, 20, 1.0, { -10.0, -10.0, -0.5 }));

  min_pt = vector_3d{ -1.0, -1.0, -0.7 };
  max_pt = vector_3d{ 1.0, 1.0, 1.2 };

  // convert to triangles for rendering
  core::mesh_triangulate(*cube);

  // create a camera sequence (elliptical path)
  auto cams = kwiver::testing::camera_seq(10, K, 1.0, 360.0);
  camera_perspective_map pcameras;
  pcameras.set_from_base_cams(cams);

  depth_maps.clear();
  cameras.clear();
  for (auto const& camera : pcameras.T_cameras())
  {
    depth_maps.push_back(core::render_mesh_depth_map(cube, camera.second));
    cameras.push_back(camera.second);
  }
}

void evaluate_volume(image_container_sptr volume,
                     vector_3d const& min_pt,
                     vector_3d const& max_pt,
                     vector_3d const& spacing)
{
  vector_3d sizes = max_pt - min_pt;
  EXPECT_NEAR(spacing[0] * volume->width(), sizes[0], spacing[0]);
  EXPECT_NEAR(spacing[1] * volume->height(), sizes[1], spacing[1]);
  EXPECT_NEAR(spacing[2] * volume->depth(), sizes[2], spacing[2]);

  // helper function to look up volume values in global coordinates
  image_of<double> vol_data(volume->get_image());
  auto world_value = [&vol_data, spacing, min_pt](vector_3d const& v)
  {
    Eigen::Vector3i index = ((v - min_pt).array() / spacing.array()).cast<int>();
    if (index[0] < 0 || index[0] >= static_cast<int>(vol_data.width()) ||
        index[1] < 0 || index[1] >= static_cast<int>(vol_data.height()) ||
        index[2] < 0 || index[2] >= static_cast<int>(vol_data.depth()))
    {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return vol_data(index[0], index[1], index[2]);
  };

  // values inside the structure should have positive values
  EXPECT_GT(world_value({ 0.0, 0.0, 0.0 }), 0.0);
  EXPECT_GT(world_value({ 0.0, 0.0, -0.6 }), 0.0);
  EXPECT_GT(world_value({ 0.0, 0.0, 0.9 }), 0.0);
  EXPECT_GT(world_value({ -0.75, -0.75, -0.6 }), 0.0);

  // values near the boundary should have small values
  EXPECT_NEAR(world_value({ 0.5, 0.0, 0.0 }), 0.0, 1.0);
  EXPECT_NEAR(world_value({ 0.0, 0.5, 0.0 }), 0.0, 1.0);
  EXPECT_NEAR(world_value({ 0.49, 0.49, 0.0 }), 0.0, 1.0);
  EXPECT_NEAR(world_value({ 0.0, 0.0, 1.0 }), 0.0, 1.0);

  // values inside the structure should have positive values
  EXPECT_LT(world_value({ 0.0, 0.0, 1.1 }), 0.0);
  EXPECT_LT(world_value({ 0.5, 0.5, 0.6 }), 0.0);
  EXPECT_LT(world_value({ -0.75, -0.75, -0.4 }), 0.0);
}

// ----------------------------------------------------------------------------
// Test depth map integration
TEST(integrate_depth_maps, integrate)
{
  namespace mvg = kwiver::arrows::mvg;

  std::vector< image_container_sptr > depth_maps;
  std::vector< camera_perspective_sptr > cams;
  vector_3d min_pt, max_pt;
  auto K = simple_camera_intrinsics(
    200, { 80, 60 }, 1.0, 0.0, {}, 160, 120);
  make_test_data(depth_maps, cams, min_pt, max_pt, K);

  mvg::integrate_depth_maps algorithm;
  config_block_sptr config = algorithm.get_configuration();
  config->set_value("voxel_spacing_factor", 1.0);
  algorithm.set_configuration(config);
  image_container_sptr volume = nullptr;
  vector_3d spacing{ 1.0, 1.0, 1.0 };
  cpu_timer timer;
  timer.start();
  algorithm.integrate(min_pt, max_pt,
                      depth_maps, {}, cams, volume, spacing);
  timer.stop();
  std::cout << "integration time: " << timer.elapsed() << std::endl;

  evaluate_volume(volume, min_pt, max_pt, spacing);
}

// ----------------------------------------------------------------------------
// Test depth map integration
TEST(integrate_depth_maps, integrate_weighted)
{
  namespace mvg = kwiver::arrows::mvg;

  std::vector< image_container_sptr > depth_maps;
  std::vector< image_container_sptr > weights;
  std::vector< camera_perspective_sptr > cams;
  vector_3d min_pt, max_pt;
  auto K = simple_camera_intrinsics(
    200, { 80, 60 }, 1.0, 0.0, {}, 160, 120);
  make_test_data(depth_maps, cams, min_pt, max_pt, K);
  image_of<double> weight(depth_maps[0]->width(), depth_maps[0]->height());
  transform_image(weight, [](double) { return 1.0; });
  for (unsigned i = 0; i < depth_maps.size(); ++i)
  {
    weights.push_back(std::make_shared<simple_image_container>(weight));
  }

  mvg::integrate_depth_maps algorithm;
  config_block_sptr config = algorithm.get_configuration();
  config->set_value("voxel_spacing_factor", 1.0);
  algorithm.set_configuration(config);
  image_container_sptr volume = nullptr;
  vector_3d spacing{ 1.0, 1.0, 1.0 };
  cpu_timer timer;
  timer.start();
  algorithm.integrate(min_pt, max_pt,
                      depth_maps, weights, cams, volume, spacing);
  timer.stop();
  std::cout << "integration time: " << timer.elapsed() << std::endl;

  evaluate_volume(volume, min_pt, max_pt, spacing);
}

// ----------------------------------------------------------------------------
// Test depth map integration
TEST(integrate_depth_maps, integrate_distorted)
{
  namespace mvg = kwiver::arrows::mvg;

  std::vector< image_container_sptr > depth_maps;
  std::vector< camera_perspective_sptr > cams;
  vector_3d min_pt, max_pt;
  Eigen::VectorXd dist(1);
  dist[0] = 0.0;
  auto K = simple_camera_intrinsics(
    200, { 80, 60 }, 1.0, 0.0, dist, 160, 120);
  make_test_data(depth_maps, cams, min_pt, max_pt, K);

  mvg::integrate_depth_maps algorithm;
  config_block_sptr config = algorithm.get_configuration();
  config->set_value("voxel_spacing_factor", 1.0);
  algorithm.set_configuration(config);
  image_container_sptr volume = nullptr;
  vector_3d spacing{ 1.0, 1.0, 1.0 };
  cpu_timer timer;
  timer.start();
  algorithm.integrate(min_pt, max_pt,
    depth_maps, {}, cams, volume, spacing);
  timer.stop();
  std::cout << "integration time: " << timer.elapsed() << std::endl;

  evaluate_volume(volume, min_pt, max_pt, spacing);
}
