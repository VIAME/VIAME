// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test aligned edge detection
 */

#include <test_gtest.h>

#include <arrows/vxl/aligned_edge_detection.h>

#include <arrows/vxl/image_container.h>
#include <arrows/vxl/image_io.h>

#include <vital/plugin_loader/plugin_manager.h>

#include <vil/vil_plane.h>

#include <gtest/gtest.h>

#include <vector>

namespace kv = kwiver::vital;
namespace ka = kwiver::arrows;

kv::path_t g_data_dir;
static std::string test_image = "images/small_grey_logo.png";

static std::string expected_seperate_edges =
  "images/expected_edge_seperate.png";
static std::string expected_combined_edges =
  "images/expected_edge_combined.png";

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
class aligned_edge_detection : public ::testing::Test
{
public:
  void SetUp();

  TEST_ARG( data_dir );

  ka::vxl::image_io io;
  ka::vxl::aligned_edge_detection filter;
  kv::image_container_sptr input_image;
};

void
aligned_edge_detection
::SetUp()
{
  std::string test_file = data_dir + "/" + test_image;
  ka::vxl::image_io io;
  input_image = io.load( test_file );
}

TEST_F(aligned_edge_detection, seperate)
{
  auto const expected_filename = data_dir + "/" + expected_seperate_edges;

  auto config = kv::config_block::empty_config();
  config->set_value( "produce_joint_output", false );
  filter.set_configuration( config );

  auto const filtered_image_ptr = filter.filter( input_image );
  auto const expected_image_ptr = io.load( expected_filename );

  EXPECT_TRUE( equal_content( filtered_image_ptr->get_image(),
                              expected_image_ptr->get_image() ) );
}

TEST_F(aligned_edge_detection, combined)
{
  auto const expected_filename = data_dir + "/" + expected_combined_edges;

  auto const filtered_image_ptr = filter.filter( input_image );
  auto const expected_image_ptr = io.load( expected_filename );

  EXPECT_TRUE( equal_content( filtered_image_ptr->get_image(),
                              expected_image_ptr->get_image() ) );
}
