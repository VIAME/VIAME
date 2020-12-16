// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <test_gtest.h>

#include <tuple>

#include <arrows/core/transfer_bbox_with_depth_map.h>

using namespace kwiver::vital;

path_t g_data_dir;

static std::string src_cam_file_name = "src_camera.krtd";
static std::string dest_cam_file_name = "dest_camera.krtd";

// ----------------------------------------------------------------------------
int main( int argc, char* argv[] )
{
  ::testing::InitGoogleTest( &argc, argv );

  GET_ARG(1, g_data_dir);

  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
class transfer_bbox_with_depth_map : public ::testing::Test
{
  TEST_ARG(data_dir);
};

// ----------------------------------------------------------------------------
TEST_F(transfer_bbox_with_depth_map, backproject_to_depth_map)
{
  path_t src_cam_file_path = data_dir + "/" + src_cam_file_name;

  auto const src_cam_sptr = read_krtd_file(src_cam_file_path);

  // Stub out our depth map
  auto img = image(1920, 1080, 1, false, image_pixel_traits_of<float>());
  img.at<float>(740, 260) = 158.8108367919922;

  auto img_ptr = std::make_shared<simple_image_container>(img);

  auto img_point = vector_2d(740.0, 260.0);

  vector_3d world_point =
    kwiver::arrows::core::backproject_to_depth_map
    (src_cam_sptr, img_ptr, img_point);

  EXPECT_NEAR(world_point(0), 8.21985243, 1e-6);
  EXPECT_NEAR(world_point(1), -75.49992019, 1e-6);
  EXPECT_NEAR(world_point(2), 1.74507372, 1e-6);

  auto bad_img_point = vector_2d(2000, 2000);

  EXPECT_THROW
    (kwiver::arrows::core::backproject_to_depth_map
     (src_cam_sptr, img_ptr, bad_img_point),
     std::invalid_argument);
}

// ----------------------------------------------------------------------------
TEST_F(transfer_bbox_with_depth_map, backproject_wrt_height)
{
  path_t src_cam_file_path = data_dir + "/" + src_cam_file_name;

  auto const src_cam_sptr = read_krtd_file(src_cam_file_path);

  // Stub out our depth map
  auto img = image(1920, 1080, 1, false, image_pixel_traits_of<float>());
  img.at<float>(920, 301) = 124.2246322631836;

  auto img_ptr = std::make_shared<simple_image_container>(img);

  auto img_point_bottom = vector_2d(920.0, 301.0);
  auto img_point_top = vector_2d(924.0, 158.0);

  vector_3d world_point_top;
  std::tie (std::ignore, world_point_top) =
    kwiver::arrows::core::backproject_wrt_height
    (src_cam_sptr, img_ptr, img_point_bottom, img_point_top);

  EXPECT_NEAR(world_point_top(0), -2.54535866, 1e-6);
  EXPECT_NEAR(world_point_top(1), -39.21040916, 1e-6);
  EXPECT_NEAR(world_point_top(2), 9.34753604, 1e-6);
}

// ----------------------------------------------------------------------------
TEST_F(transfer_bbox_with_depth_map,
       transfer_bbox_with_depth_map_stationary_camera)
{
  path_t src_cam_file_path = data_dir + "/" + src_cam_file_name;
  path_t dest_cam_file_path = data_dir + "/" + dest_cam_file_name;

  auto const src_cam_sptr = read_krtd_file(src_cam_file_path);
  auto const dest_cam_sptr = read_krtd_file(dest_cam_file_path);

  // Stub out our depth map
  auto img = image(1920, 1080, 1, false, image_pixel_traits_of<float>());
  img.at<float>(920, 301) = 124.2246322631836;

  auto img_ptr = std::make_shared<simple_image_container>(img);

  auto bbox = kwiver::vital::bounding_box<double>(900.0, 154.0, 940.0, 301.0);

  kwiver::vital::bounding_box<double> out_bbox =
    kwiver::arrows::core::transfer_bbox_with_depth_map_stationary_camera
    (src_cam_sptr, dest_cam_sptr, img_ptr, bbox);

  EXPECT_NEAR(out_bbox.min_x(), 586.1738903996884, 1e-6);
  EXPECT_NEAR(out_bbox.min_y(), 10.778501924859142, 1e-6);
  EXPECT_NEAR(out_bbox.max_x(), 700.6970751394713, 1e-6);
  EXPECT_NEAR(out_bbox.max_y(), 431.65120584356146, 1e-6);
}
