// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <test_gtest.h>

#include <arrows/vxl/hashed_image_classifier_filter.h>
#include <arrows/vxl/image_container.h>
#include <arrows/vxl/image_io.h>

#include <vital/plugin_loader/plugin_manager.h>

#include <vil/vil_convert.h>
#include <vil/vil_math.h>

#include <gtest/gtest.h>

namespace kv = kwiver::vital;
namespace ka = kwiver::arrows;

kv::path_t g_data_dir;
static std::string feature_images = "images/features_expected.png";
static std::string expected_classified_image =
  "images/classified_expected.png";
static std::string model_file = "default_burnout_600_iters";

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
class hashed_image_classifier_filter : public ::testing::Test
{
  TEST_ARG( data_dir );
};

// ----------------------------------------------------------------------------
TEST_F(hashed_image_classifier_filter, compute_all)
{
  std::string input_filename = data_dir + "/" + feature_images;
  std::string expected_filename = data_dir + "/" + expected_classified_image;
  std::string model_filepath = data_dir + "/" + model_file;

  ka::vxl::hashed_image_classifier_filter filter;
  ka::vxl::image_io io;

  auto io_config = kv::config_block::empty_config();
  io_config->set_value( "split_channels", true );
  io.set_configuration( io_config );
  auto const input_image = io.load( input_filename );

  auto filter_config = kv::config_block::empty_config();
  filter_config->set_value( "model_file", model_filepath );
  filter.set_configuration( filter_config );
  auto const filtered = filter.filter( input_image );

  // Only byte values can be saved to disk, so convert to a reasonable range
  auto vxl_filtered =
    static_cast< vil_image_view< double > >(
      ka::vxl::image_container::vital_to_vxl(
        filtered->get_image() ) );
  vil_math_scale_and_offset_values( vxl_filtered, 512.0, 128.0 );
  vil_image_view< vxl_byte > filtered_byte;
  vil_convert_cast( vxl_filtered, filtered_byte );
  auto const filtered_byte_vital =
    std::make_shared< ka::vxl::image_container >( filtered_byte );

  auto const expected = io.load( expected_filename );

  EXPECT_TRUE( equal_content( filtered_byte_vital->get_image(),
                              expected->get_image() ) );
}
