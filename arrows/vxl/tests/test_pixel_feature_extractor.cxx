// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <test_gtest.h>

#include <arrows/vxl/pixel_feature_extractor.h>
#include <arrows/vxl/image_container.h>
#include <arrows/vxl/image_io.h>

#include <vital/plugin_loader/plugin_manager.h>

#include <vil/vil_plane.h>

#include <gtest/gtest.h>

#include <vector>

namespace kv = kwiver::vital;
namespace ka = kwiver::arrows;

kv::path_t g_data_dir;
static std::string test_color_image_name = "images/small_color_logo.png";
static std::string expected_name = "images/features_expected.png";

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
class pixel_feature_extractor : public ::testing::Test
{
  TEST_ARG( data_dir );
};

// ----------------------------------------------------------------------------
TEST_F(pixel_feature_extractor, compute_all)
{
  std::string input_filename = data_dir + "/" + test_color_image_name;
  std::string expected_filename = data_dir + "/" + expected_name;

  ka::vxl::pixel_feature_extractor filter;
  ka::vxl::image_io io;

  auto const input_image = io.load( input_filename );
  auto const filtered = filter.filter( input_image );

  // Many-plane images are saved in a per-channel format
  auto io_config = kv::config_block::empty_config();
  io_config->set_value( "split_channels", true );
  io.set_configuration( io_config );

  auto const expected = io.load( expected_filename );
  EXPECT_TRUE( equal_content( filtered->get_image(),
                              expected->get_image() ) );
}
