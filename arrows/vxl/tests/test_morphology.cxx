// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <test_gtest.h>

#include <arrows/vxl/image_container.h>
#include <arrows/vxl/image_io.h>
#include <arrows/vxl/morphology.h>

#include <vil/algo/vil_threshold.h>
#include <vil/vil_convert.h>

#include <vital/plugin_loader/plugin_manager.h>

#include <gtest/gtest.h>

namespace kv = kwiver::vital;
namespace ka = kwiver::arrows;
namespace kav = kwiver::arrows::vxl;

kv::path_t g_data_dir;
static std::string test_color_image_name = "images/small_color_logo.png";
static std::string expected_morphology_erode =
  "images/morphology_erode.png";
static std::string expected_morphology_dilate =
  "images/morphology_dilate.png";
static std::string expected_morphology_union =
  "images/morphology_union.png";
static std::string expected_morphology_intersection =
  "images/morphology_intersection.png";

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
class morphology : public ::testing::Test
{
  void SetUp();

  TEST_ARG( data_dir );

  ka::vxl::image_io io;
  ka::vxl::morphology filter;
  kv::image_container_sptr input_image;
  vil_image_view< vxl_byte > vxl_byte_image;

public:
  void test_morphology_type( kv::config_block_sptr const& config,
                             kv::path_t expected_basename );
};

// ----------------------------------------------------------------------------
void
morphology
::SetUp()
{
  std::string test_file = data_dir + "/" + test_color_image_name;
  ka::vxl::image_io io;
  input_image = io.load( test_file );

  vxl_byte_image =
    kav::image_container::vital_to_vxl( input_image->get_image() );

  vil_image_view< bool > thresholded{
    vxl_byte_image.ni(), vxl_byte_image.nj(), vxl_byte_image.nplanes() };
  vil_threshold_below< vxl_byte >( vxl_byte_image, thresholded, 128 );
  input_image = std::make_shared< kav::image_container >( thresholded );
}

// ----------------------------------------------------------------------------
void
morphology
::test_morphology_type( kv::config_block_sptr const& config,
                        kv::path_t expected_basename )
{
  filter.set_configuration( config );

  auto const filtered_image_ptr = filter.filter( input_image );
  vil_image_view< bool > filtered_vxl_image =
    kav::image_container::vital_to_vxl( filtered_image_ptr->get_image() );
  vil_convert_cast( filtered_vxl_image, vxl_byte_image );
  vil_math_scale_values( vxl_byte_image, 255 );

  auto const filtered_byte_image_ptr =
   std::make_shared< ka::vxl::image_container >( vxl_byte_image );

  auto expected_filename = data_dir + "/" + expected_basename;

  auto const expected_image_ptr = io.load( expected_filename );
  EXPECT_TRUE( equal_content( filtered_byte_image_ptr->get_image(),
                              expected_image_ptr->get_image() ) );
}

// ----------------------------------------------------------------------------
TEST_F(morphology, erode)
{
  auto config = kv::config_block::empty_config();
  config->set_value( "morphology", "erode" );
  config->set_value( "element_size", "32" );

  test_morphology_type( config, expected_morphology_erode );
}

// ----------------------------------------------------------------------------
TEST_F(morphology, dilate)
{
  auto config = kv::config_block::empty_config();
  config->set_value( "morphology", "dilate" );
  config->set_value( "element_size", "32" );

  test_morphology_type( config, expected_morphology_dilate );
}

// ----------------------------------------------------------------------------
TEST_F(morphology, union)
{
  auto config = kv::config_block::empty_config();
  config->set_value( "channel_combination", "union" );
  config->set_value( "element_size", "128" );

  test_morphology_type( config, expected_morphology_union );
}

// ----------------------------------------------------------------------------
TEST_F(morphology, intersection)
{
  auto config = kv::config_block::empty_config();
  config->set_value( "channel_combination", "intersection" );
  config->set_value( "element_size", "128" );

  test_morphology_type( config, expected_morphology_intersection );
}
