// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test VXL color commonality filtering
 */

#include <test_gtest.h>

#include <arrows/vxl/color_commonality_filter.h>
#include <arrows/vxl/image_io.h>

#include <vital/plugin_loader/plugin_manager.h>

#include <gtest/gtest.h>

namespace kv = kwiver::vital;
namespace ka = kwiver::arrows;

kv::path_t g_data_dir;
static std::string test_image_name = "images/small_grey_logo.png";
static std::string test_color_image_name = "images/small_color_logo.png";
static std::string expected_commonality_default_color =
  "images/commonality_filter_default_color.png";
static std::string expected_commonality_default_gray =
  "images/commonality_filter_default_gray.png";

// ----------------------------------------------------------------------------
int
main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  TEST_LOAD_PLUGINS();

  GET_ARG( 1, g_data_dir );

  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
class color_commonality_filter : public ::testing::Test
{
  TEST_ARG( data_dir );
};

// ----------------------------------------------------------------------------
TEST_F(color_commonality_filter, color)
{
  ka::vxl::image_io io;

  std::string filename = data_dir + "/" + test_color_image_name;
  auto const& image_ptr = io.load( filename );

  ka::vxl::color_commonality_filter filter;

  auto const filtered_image_ptr = filter.filter( image_ptr );

  kv::path_t expected = data_dir + "/" + expected_commonality_default_color;
  auto const& expected_image_ptr = io.load( expected );
  EXPECT_TRUE( equal_content( filtered_image_ptr->get_image(),
                              expected_image_ptr->get_image() ) );
}

// ----------------------------------------------------------------------------
TEST_F(color_commonality_filter, gray)
{
  ka::vxl::image_io io;

  std::string filename = data_dir + "/" + test_image_name;
  auto const& image_ptr = io.load( filename );

  ka::vxl::color_commonality_filter filter;

  auto const filtered_image_ptr = filter.filter( image_ptr );

  kv::path_t expected = data_dir + "/" + expected_commonality_default_gray;
  auto const& expected_image_ptr = io.load( expected );
  EXPECT_TRUE( equal_content( filtered_image_ptr->get_image(),
                              expected_image_ptr->get_image() ) );
}
