// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test VXL high pass filtering
 */

#include <test_gtest.h>

#include <arrows/vxl/high_pass_filter.h>
#include <arrows/vxl/image_container.h>
#include <arrows/vxl/image_io.h>

#include <vital/plugin_loader/plugin_manager.h>

#include <gtest/gtest.h>

namespace kv = kwiver::vital;
namespace ka = kwiver::arrows;

kv::path_t g_data_dir;
static std::string test_image_name = "images/small_grey_logo.png";
static std::string test_color_image_name = "images/small_color_logo.png";
static std::string expected_box = "images/box.png";
static std::string expected_box_wide = "images/box_wide.png";
static std::string expected_bidir = "images/bidir.png";
static std::string expected_bidir_wide = "images/bidir_wide.png";
static std::string expected_bidir_color = "images/bidir_color.png";

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
class high_pass_filter : public ::testing::Test
{
  TEST_ARG( data_dir );
};

// ----------------------------------------------------------------------------
TEST_F(high_pass_filter, color)
{
  ka::vxl::image_io io;

  std::string filename = data_dir + "/" + test_color_image_name;
  auto const& image_ptr = io.load( filename );

  ka::vxl::high_pass_filter filter;
  auto config = kv::config_block::empty_config();
  config->set_value( "mode", "bidir" );
  filter.set_configuration( config );

  auto const filtered_image_ptr = filter.filter( image_ptr );
  kv::path_t expected = data_dir + "/" + expected_bidir_color;
  auto const& expected_image_ptr = io.load( expected );
  EXPECT_TRUE( equal_content( filtered_image_ptr->get_image(),
                              expected_image_ptr->get_image() ) );
}

// ----------------------------------------------------------------------------
TEST_F(high_pass_filter, box)
{
  ka::vxl::image_io io;

  std::string filename = data_dir + "/" + test_image_name;
  auto const& image_ptr = io.load( filename );

  ka::vxl::high_pass_filter filter;
  auto config = kv::config_block::empty_config();
  config->set_value( "mode", "box" );
  filter.set_configuration( config );

  auto const filtered_image_ptr = filter.filter( image_ptr );
  kv::path_t expected = data_dir + "/" + expected_box;
  auto const& expected_image_ptr = io.load( expected );
  EXPECT_TRUE( equal_content( filtered_image_ptr->get_image(),
                              expected_image_ptr->get_image() ) );
}

// ----------------------------------------------------------------------------
TEST_F(high_pass_filter, box_wide)
{
  ka::vxl::image_io io;

  std::string filename = data_dir + "/" + test_image_name;
  auto const& image_ptr = io.load( filename );

  ka::vxl::high_pass_filter filter;
  auto config = kv::config_block::empty_config();
  config->set_value( "mode", "box" );
  config->set_value( "kernel_width", 15 );
  config->set_value( "kernel_height", 15 );
  filter.set_configuration( config );

  auto const filtered_image_ptr = filter.filter( image_ptr );
  kv::path_t expected = data_dir + "/" + expected_box_wide;
  auto const& expected_image_ptr = io.load( expected );
  EXPECT_TRUE( equal_content( filtered_image_ptr->get_image(),
                              expected_image_ptr->get_image() ) );
}

// ----------------------------------------------------------------------------
TEST_F(high_pass_filter, bidir)
{
  ka::vxl::image_io io;

  std::string filename = data_dir + "/" + test_image_name;
  auto const& image_ptr = io.load( filename );

  ka::vxl::high_pass_filter filter;
  auto config = kv::config_block::empty_config();
  config->set_value( "mode", "bidir" );
  filter.set_configuration( config );

  auto const filtered_image_ptr = filter.filter( image_ptr );
  kv::path_t expected = data_dir + "/" + expected_bidir;
  auto const& expected_image_ptr = io.load( expected );
  EXPECT_TRUE( equal_content( filtered_image_ptr->get_image(),
                              expected_image_ptr->get_image() ) );
}

// ----------------------------------------------------------------------------
TEST_F(high_pass_filter, bidir_wide)
{
  ka::vxl::image_io io;

  std::string filename = data_dir + "/" + test_image_name;
  auto const image_ptr = io.load( filename );

  ka::vxl::high_pass_filter filter;
  auto config = kv::config_block::empty_config();
  config->set_value( "mode", "bidir" );
  config->set_value( "kernel_width", 15 );
  config->set_value( "kernel_height", 15 );
  filter.set_configuration( config );

  kv::image_container_sptr filtered_image_ptr = filter.filter( image_ptr );
  kv::path_t expected = data_dir + "/" + expected_bidir_wide;
  auto const expected_image_ptr = io.load( expected );
  EXPECT_TRUE( equal_content( filtered_image_ptr->get_image(),
                              expected_image_ptr->get_image() ) );
}
